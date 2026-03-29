"""Vercel serverless entrypoint for FastAPI.

Exports ``handler`` (Mangum-wrapped ASGI app) which Vercel's Python runtime
invokes for every incoming request.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so src.* modules resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mangum import Mangum  # noqa: E402
from api.main import app  # noqa: E402

handler = Mangum(app, lifespan="off")
