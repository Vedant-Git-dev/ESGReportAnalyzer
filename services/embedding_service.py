"""
services/embedding_service.py

Embedding Service — computes and stores dense vector embeddings for all
DocumentChunks using a local sentence-transformers model.

Design decisions:
  - Model runs entirely locally, zero API cost
  - Embeddings computed once at parse time, reused forever
  - Batched inference for efficiency (default batch_size=64)
  - Falls back gracefully if model unavailable (keyword retrieval still works)
  - is_embedded flag prevents redundant re-computation

Default model: BAAI/bge-small-en-v1.5
  - 384 dimensions, ~130MB, runs on CPU
  - Better ESG domain accuracy than all-MiniLM-L6-v2
  - Instruction-aware: prepend "Represent this sentence for searching" to queries
  - Swappable: change EMBEDDING_MODEL in .env, matches EMBEDDING_DIM in db_models.py

Cosine similarity is used for retrieval (pgvector's <=> operator).
"""
from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import Float

from core.config import get_settings
from core.logging_config import get_logger
from models.db_models import DocumentChunk, EMBEDDING_DIM

logger = get_logger(__name__)

# Instruction prefix improves retrieval accuracy for bge models
_ENCODE_PREFIX = ""           # for document chunks
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingService:
    """
    Singleton-friendly embedding service.
    Model is loaded lazily on first call — startup cost ~2s, then cached.
    """

    _instance: Optional["EmbeddingService"] = None
    _model = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_name: str = getattr(self.settings, "embedding_model", "BAAI/bge-small-en-v1.5")
        self.batch_size: int = getattr(self.settings, "embedding_batch_size", 64)
        self._model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("embedding_service.loading_model", model=self.model_name)
                self._model = SentenceTransformer(self.model_name)
                actual_dim = self._model.get_sentence_embedding_dimension()
                if actual_dim != EMBEDDING_DIM:
                    raise ValueError(
                        f"Model produces {actual_dim}-dim embeddings "
                        f"but EMBEDDING_DIM={EMBEDDING_DIM}. "
                        f"Update EMBEDDING_DIM in db_models.py to match."
                    )
                logger.info("embedding_service.model_ready", dim=actual_dim)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model

    def is_available(self) -> bool:
        """Return True if the embedding model can be loaded."""
        try:
            self._get_model()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_texts(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Args:
            texts:    List of strings to encode
            is_query: If True, prepend query instruction prefix (bge models)

        Returns:
            numpy array of shape (len(texts), EMBEDDING_DIM), float32
        """
        model = self._get_model()

        if is_query:
            texts = [_QUERY_PREFIX + t for t in texts]

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single retrieval query with instruction prefix."""
        return self.encode_texts([query], is_query=True)[0]

    # ------------------------------------------------------------------
    # Batch embedding for a parsed document
    # ------------------------------------------------------------------

    def embed_document(
        self,
        parsed_document_id: uuid.UUID,
        db: Session,
        force: bool = False,
    ) -> int:
        """
        Compute and store embeddings for all unembedded chunks of a parsed document.

        Args:
            parsed_document_id: UUID of ParsedDocument
            db:    Active session
            force: Re-embed even if is_embedded=True

        Returns:
            Number of chunks newly embedded
        """
        query = db.query(DocumentChunk).filter(
            DocumentChunk.parsed_document_id == parsed_document_id,
        )
        if not force:
            query = query.filter(DocumentChunk.is_embedded == False)

        chunks = query.order_by(DocumentChunk.chunk_index).all()

        if not chunks:
            logger.info("embedding_service.nothing_to_embed",
                       parsed_document_id=str(parsed_document_id))
            return 0

        logger.info("embedding_service.embed_start",
                   parsed_document_id=str(parsed_document_id),
                   chunks=len(chunks))

        # Batch encode all chunk texts
        texts = [c.content for c in chunks]
        embeddings = self.encode_texts(texts, is_query=False)

        # Store back into DB
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
            chunk.is_embedded = True

        db.flush()

        logger.info("embedding_service.embed_complete",
                   parsed_document_id=str(parsed_document_id),
                   embedded=len(chunks))
        return len(chunks)

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search(
        self,
        parsed_document_id: uuid.UUID,
        query: str,
        db: Session,
        top_k: int = 7,
        chunk_types: Optional[list[str]] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Find the top-K most semantically similar chunks to a query string.

        Uses pgvector's cosine distance operator (<=>) for efficient ANN search.

        Args:
            parsed_document_id: Scope search to this document
            query:  Natural-language query string (e.g. "Scope 1 GHG emissions tCO2e")
            db:     Active session
            top_k:  Number of results to return
            chunk_types: Optional filter e.g. ["table", "text"]

        Returns:
            List of (DocumentChunk, similarity_score) sorted by similarity desc.
            similarity_score is in [0, 1] — higher is more similar.
        """
        from sqlalchemy import func, cast
        from pgvector.sqlalchemy import Vector

        query_vec = self.encode_query(query)
        query_vec_list = query_vec.tolist()

        # pgvector cosine distance: 1 - cosine_similarity
        # <=> operator returns distance (0=identical, 2=opposite)
        distance_expr = cast(
            DocumentChunk.embedding.op("<=>")(
                cast(query_vec_list, Vector(EMBEDDING_DIM))
            ),
            Float
        ).label("distance")

        q = db.query(DocumentChunk, distance_expr.label("distance")).filter(
            DocumentChunk.parsed_document_id == parsed_document_id,
            DocumentChunk.is_embedded == True,
        )

        if chunk_types:
            q = q.filter(DocumentChunk.chunk_type.in_(chunk_types))

        results = q.order_by(distance_expr).limit(top_k).all()

        # Convert distance to similarity score
        return [
            (chunk, round(1.0 - float(distance), 4))
            for chunk, distance in results
        ]

    def search_multi_query(
        self,
        parsed_document_id: uuid.UUID,
        queries: list[str],
        db: Session,
        top_k: int = 7,
        chunk_types: Optional[list[str]] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Run multiple query strings and merge results by max similarity.
        Useful for KPIs that may be phrased multiple different ways.

        Returns deduplicated top-K chunks sorted by best similarity score.
        """
        seen: dict[uuid.UUID, tuple[DocumentChunk, float]] = {}

        for query in queries:
            results = self.search(
                parsed_document_id=parsed_document_id,
                query=query,
                db=db,
                top_k=top_k,
                chunk_types=chunk_types,
            )
            for chunk, score in results:
                if chunk.id not in seen or score > seen[chunk.id][1]:
                    seen[chunk.id] = (chunk, score)

        merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        return merged[:top_k]