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

FIXES (v2)
----------
1. True module-level singleton so the model is loaded exactly once per process
   and survives across multiple EmbeddingService() instantiations.

2. Explicit, actionable logging at every failure point:
   - model load attempts, success, and failure (with full error message)
   - per-batch encode timing and shape
   - per-chunk embedding assignment
   - post-flush DB verification (count of rows where is_embedded=True)

3. Robust numpy→Python-float conversion: always call .astype(numpy.float64)
   before .tolist() so pgvector receives a list[float] never list[numpy.float32].
   pgvector's SQLAlchemy adapter can silently store None or raise for numpy scalars.

4. Post-insert validation: after db.flush(), re-query the DB to confirm the
   expected number of rows has is_embedded=True and embedding IS NOT NULL.

5. Graceful offline-mode: on first load failure the service logs the exact
   exception, then retries with local_files_only=True (uses the HuggingFace
   disk cache if the model was previously downloaded). If both fail, is_available()
   returns False and keyword-only retrieval continues working.
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import Float, func

from core.config import get_settings
from core.logging_config import get_logger
from models.db_models import DocumentChunk, EMBEDDING_DIM

logger = get_logger(__name__)

# Instruction prefix improves retrieval accuracy for bge models
_ENCODE_PREFIX = ""           # for document chunks (no prefix needed for bge-small)
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# Module-level singleton — survives multiple EmbeddingService() instantiations
# ---------------------------------------------------------------------------
# This is intentionally at module scope so that even if the caller creates
# a fresh EmbeddingService() object on every parse, the expensive model load
# only happens once per Python process.
_GLOBAL_MODEL = None
_GLOBAL_MODEL_NAME: Optional[str] = None
_GLOBAL_MODEL_AVAILABLE: Optional[bool] = None   # None = not yet tried


def _load_model(model_name: str):
    """
    Load the sentence-transformers model.

    Attempt order:
      1. Normal load (downloads from HuggingFace or uses local cache).
      2. local_files_only=True (uses disk cache, no network; works after
         the model has been downloaded at least once).

    Returns the loaded model or raises the last exception.
    Logs the exact failure reason so the operator can act on it.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("embedding_service.model_load_attempt", model=model_name)

    # Attempt 1: normal load (uses HF cache if available, else downloads)
    try:
        t0 = time.time()
        model = SentenceTransformer(model_name)
        elapsed = round(time.time() - t0, 2)
        dim = model.get_sentence_embedding_dimension()
        logger.info(
            "embedding_service.model_loaded",
            model=model_name,
            dim=dim,
            load_time_s=elapsed,
            source="normal",
        )
        return model
    except Exception as exc_online:
        logger.warning(
            "embedding_service.model_load_online_failed",
            model=model_name,
            error=str(exc_online),
            hint=(
                "HuggingFace Hub unreachable or model not yet cached. "
                "Retrying with local_files_only=True (uses disk cache)."
            ),
        )

    # Attempt 2: offline mode — only works if model was previously downloaded
    try:
        t0 = time.time()
        model = SentenceTransformer(model_name, local_files_only=True)
        elapsed = round(time.time() - t0, 2)
        dim = model.get_sentence_embedding_dimension()
        logger.info(
            "embedding_service.model_loaded",
            model=model_name,
            dim=dim,
            load_time_s=elapsed,
            source="local_cache",
        )
        return model
    except Exception as exc_offline:
        logger.error(
            "embedding_service.model_load_failed_both_attempts",
            model=model_name,
            online_error=str(exc_online),
            offline_error=str(exc_offline),
            action=(
                "Semantic embeddings are DISABLED for this run. "
                "Retrieval will fall back to keyword-only mode. "
                "To enable embeddings: ensure HuggingFace is reachable, "
                "or pre-download the model with: "
                f"python -c \"from sentence_transformers import SentenceTransformer; "
                f"SentenceTransformer('{model_name}')\""
            ),
        )
        raise exc_offline


class EmbeddingService:
    """
    Singleton-friendly embedding service.
    Model is loaded lazily on first call — startup cost ~2s, then cached
    at module level so subsequent instantiations reuse it for free.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_name: str = getattr(
            self.settings, "embedding_model", "BAAI/bge-small-en-v1.5"
        )
        self.batch_size: int = getattr(self.settings, "embedding_batch_size", 64)
        # NOTE: we deliberately do NOT set self._model here.
        # _get_model() always reads from the module-level singleton so the
        # model persists even when the caller creates a new EmbeddingService().

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_model(self):
        """
        Return the cached model, loading it on first call.
        Uses a module-level singleton so the model is loaded once per process.
        Raises RuntimeError if sentence-transformers is not installed.
        Raises OSError / similar if the model cannot be loaded.
        """
        global _GLOBAL_MODEL, _GLOBAL_MODEL_NAME, _GLOBAL_MODEL_AVAILABLE

        # Already loaded (and the model name hasn't changed)
        if _GLOBAL_MODEL is not None and _GLOBAL_MODEL_NAME == self.model_name:
            return _GLOBAL_MODEL

        # Already tried and failed for this model
        if _GLOBAL_MODEL_AVAILABLE is False and _GLOBAL_MODEL_NAME == self.model_name:
            raise RuntimeError(
                f"Embedding model '{self.model_name}' failed to load earlier "
                "in this process. See previous log entries for details."
            )

        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        try:
            model = _load_model(self.model_name)
        except Exception:
            _GLOBAL_MODEL_AVAILABLE = False
            _GLOBAL_MODEL_NAME = self.model_name
            raise

        # Validate dimension matches DB schema
        actual_dim = model.get_sentence_embedding_dimension()
        if actual_dim != EMBEDDING_DIM:
            raise ValueError(
                f"Model '{self.model_name}' produces {actual_dim}-dim embeddings "
                f"but EMBEDDING_DIM={EMBEDDING_DIM} in db_models.py. "
                f"Update EMBEDDING_DIM to {actual_dim} and re-run migrations."
            )

        _GLOBAL_MODEL = model
        _GLOBAL_MODEL_NAME = self.model_name
        _GLOBAL_MODEL_AVAILABLE = True
        logger.info(
            "embedding_service.model_ready",
            model=self.model_name,
            dim=actual_dim,
        )
        return _GLOBAL_MODEL

    def is_available(self) -> bool:
        """Return True if the embedding model can be loaded."""
        global _GLOBAL_MODEL_AVAILABLE
        if _GLOBAL_MODEL_AVAILABLE is True:
            return True
        if _GLOBAL_MODEL_AVAILABLE is False:
            logger.debug("embedding_service.is_available=False (cached failure)")
            return False
        # Not yet tried
        try:
            self._get_model()
            return True
        except Exception as exc:
            logger.warning(
                "embedding_service.is_available_check_failed",
                error=str(exc),
            )
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

        logger.debug(
            "embedding_service.encode_start",
            n_texts=len(texts),
            is_query=is_query,
            batch_size=self.batch_size,
        )

        t0 = time.time()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
            convert_to_numpy=True,
        )
        elapsed = round(time.time() - t0, 3)

        # Guarantee float32 regardless of model output dtype
        embeddings = embeddings.astype(np.float32)

        logger.debug(
            "embedding_service.encode_done",
            shape=list(embeddings.shape),
            dtype=str(embeddings.dtype),
            elapsed_s=elapsed,
        )

        # Sanity: dimension must match DB column
        if embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"encode_texts produced {embeddings.shape[1]}-dim embeddings "
                f"but EMBEDDING_DIM={EMBEDDING_DIM}. "
                "This should have been caught in _get_model(); investigate."
            )

        return embeddings

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
            Number of chunks newly embedded (0 if nothing to do or on error).
        """
        logger.info(
            "embedding_service.embed_document_start",
            parsed_document_id=str(parsed_document_id),
            force=force,
        )

        # ------------------------------------------------------------------
        # 1. Load chunks to embed
        # ------------------------------------------------------------------
        query = db.query(DocumentChunk).filter(
            DocumentChunk.parsed_document_id == parsed_document_id,
        )
        if not force:
            query = query.filter(DocumentChunk.is_embedded == False)

        chunks = query.order_by(DocumentChunk.chunk_index).all()

        if not chunks:
            already_embedded = db.query(func.count(DocumentChunk.id)).filter(
                DocumentChunk.parsed_document_id == parsed_document_id,
                DocumentChunk.is_embedded == True,
            ).scalar() or 0
            logger.info(
                "embedding_service.nothing_to_embed",
                parsed_document_id=str(parsed_document_id),
                already_embedded=already_embedded,
            )
            return 0

        logger.info(
            "embedding_service.embed_start",
            parsed_document_id=str(parsed_document_id),
            chunks_to_embed=len(chunks),
            force=force,
        )

        # ------------------------------------------------------------------
        # 2. Batch encode
        # ------------------------------------------------------------------
        texts = [c.content for c in chunks]

        logger.debug(
            "embedding_service.encoding_texts",
            n_chunks=len(texts),
            sample_preview=texts[0][:80].replace("\n", " ") if texts else "",
        )

        try:
            embeddings = self.encode_texts(texts, is_query=False)
        except Exception as exc:
            logger.error(
                "embedding_service.encode_failed",
                parsed_document_id=str(parsed_document_id),
                n_chunks=len(texts),
                error=str(exc),
                action="Embeddings NOT stored. Keyword retrieval will be used.",
            )
            return 0

        logger.info(
            "embedding_service.encode_success",
            parsed_document_id=str(parsed_document_id),
            shape=list(embeddings.shape),
            dtype=str(embeddings.dtype),
        )

        # ------------------------------------------------------------------
        # 3. Store embeddings — convert numpy float32 → list[float] for pgvector
        # ------------------------------------------------------------------
        stored_count = 0
        failed_count = 0

        for idx, (chunk, emb_row) in enumerate(zip(chunks, embeddings)):
            try:
                # Explicit conversion: numpy float32 scalar → Python native float
                # pgvector's SQLAlchemy adapter expects list[float] (Python builtins),
                # NOT list[numpy.float32]. Using .astype(float64).tolist() guarantees
                # Python float objects even if numpy dtypes change between versions.
                emb_list: list[float] = emb_row.astype(np.float64).tolist()

                # Validate before storing
                if len(emb_list) != EMBEDDING_DIM:
                    logger.error(
                        "embedding_service.wrong_dim_on_chunk",
                        chunk_index=chunk.chunk_index,
                        got=len(emb_list),
                        expected=EMBEDDING_DIM,
                    )
                    failed_count += 1
                    continue

                if not all(isinstance(v, float) for v in emb_list[:5]):
                    logger.error(
                        "embedding_service.non_float_in_embedding",
                        chunk_index=chunk.chunk_index,
                        sample_types=[type(v).__name__ for v in emb_list[:5]],
                    )
                    failed_count += 1
                    continue

                chunk.embedding = emb_list
                chunk.is_embedded = True
                stored_count += 1

                if idx < 3 or idx % 500 == 0:
                    logger.debug(
                        "embedding_service.chunk_embedding_set",
                        chunk_index=chunk.chunk_index,
                        page=chunk.page_number,
                        dim=len(emb_list),
                        first3=emb_list[:3],
                    )

            except Exception as exc:
                logger.error(
                    "embedding_service.chunk_store_error",
                    chunk_index=chunk.chunk_index,
                    error=str(exc),
                )
                failed_count += 1
                continue

        logger.info(
            "embedding_service.assignments_done",
            parsed_document_id=str(parsed_document_id),
            stored=stored_count,
            failed=failed_count,
            total_chunks=len(chunks),
        )

        # ------------------------------------------------------------------
        # 4. Flush to DB
        # ------------------------------------------------------------------
        if stored_count == 0:
            logger.error(
                "embedding_service.zero_embeddings_stored",
                parsed_document_id=str(parsed_document_id),
                action="db.flush() skipped — nothing was assigned.",
            )
            return 0

        try:
            db.flush()
            logger.info(
                "embedding_service.flush_success",
                parsed_document_id=str(parsed_document_id),
                flushed=stored_count,
            )
        except Exception as exc:
            logger.error(
                "embedding_service.flush_failed",
                parsed_document_id=str(parsed_document_id),
                error=str(exc),
                action=(
                    "DB flush failed — embeddings are NOT persisted. "
                    "Check pgvector extension is installed and the embedding "
                    f"column is Vector({EMBEDDING_DIM}). "
                    "Run: CREATE EXTENSION IF NOT EXISTS vector;"
                ),
            )
            return 0

        # ------------------------------------------------------------------
        # 5. Post-flush validation: re-query DB to confirm storage
        # ------------------------------------------------------------------
        try:
            confirmed = db.query(func.count(DocumentChunk.id)).filter(
                DocumentChunk.parsed_document_id == parsed_document_id,
                DocumentChunk.is_embedded == True,
                DocumentChunk.embedding.isnot(None),
            ).scalar() or 0

            total_chunks = db.query(func.count(DocumentChunk.id)).filter(
                DocumentChunk.parsed_document_id == parsed_document_id,
            ).scalar() or 0

            coverage_pct = round(100 * confirmed / total_chunks, 1) if total_chunks else 0

            if confirmed == 0:
                logger.error(
                    "embedding_service.validation_zero_embeddings",
                    parsed_document_id=str(parsed_document_id),
                    expected=stored_count,
                    confirmed=0,
                    action=(
                        "CRITICAL: flush reported success but DB shows 0 embedded chunks. "
                        "Check that the embedding column is of type vector and pgvector "
                        "extension is active on the database."
                    ),
                )
            elif confirmed < stored_count:
                logger.warning(
                    "embedding_service.validation_partial",
                    parsed_document_id=str(parsed_document_id),
                    expected=stored_count,
                    confirmed=confirmed,
                    missing=stored_count - confirmed,
                )
            else:
                logger.info(
                    "embedding_service.validation_ok",
                    parsed_document_id=str(parsed_document_id),
                    confirmed=confirmed,
                    total_chunks=total_chunks,
                    coverage_pct=coverage_pct,
                )

        except Exception as exc:
            logger.warning(
                "embedding_service.validation_query_failed",
                parsed_document_id=str(parsed_document_id),
                error=str(exc),
                note="Could not verify storage — embeddings may or may not be persisted.",
            )

        logger.info(
            "embedding_service.embed_complete",
            parsed_document_id=str(parsed_document_id),
            embedded=stored_count,
        )
        return stored_count

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
        from sqlalchemy import func as sa_func, cast
        from pgvector.sqlalchemy import Vector

        query_vec = self.encode_query(query)
        query_vec_list = query_vec.astype(np.float64).tolist()

        logger.debug(
            "embedding_service.search_start",
            parsed_document_id=str(parsed_document_id),
            query_preview=query[:80],
            top_k=top_k,
        )

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

        logger.debug(
            "embedding_service.search_done",
            parsed_document_id=str(parsed_document_id),
            returned=len(results),
            top_distance=round(float(results[0][1]), 4) if results else None,
        )

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