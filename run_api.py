"""
run_api.py

Convenience launcher for the ESG API server.

Usage:
    python run_api.py                    # defaults: host=0.0.0.0 port=8000
    python run_api.py --port 9000
    python run_api.py --no-reload

API docs available at: http://localhost:8000/api/docs
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG Competitive Intelligence API")
    parser.add_argument("--host",      default="0.0.0.0")
    parser.add_argument("--port",      type=int, default=8000)
    parser.add_argument("--no-reload", action="store_true")
    args = parser.parse_args()

    print(f"\n  ESG Intelligence API")
    print(f"  ─────────────────────────────────────────")
    print(f"  Server   : http://{args.host}:{args.port}")
    print(f"  API docs : http://localhost:{args.port}/api/docs")
    print(f"  React    : http://localhost:5173  (run: cd frontend && npm run dev)")
    print(f"  ─────────────────────────────────────────\n")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        reload_dirs=[str(ROOT / "api")],
        log_level="info",
    )