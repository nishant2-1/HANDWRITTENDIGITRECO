"""Vercel serverless entrypoint for FastAPI.

Mangum wraps the FastAPI ASGI app into an AWS Lambda-compatible handler,
which is the invocation model Vercel's Python runtime uses internally.
Vercel looks for a module-level ``handler`` variable in this file.
"""

from __future__ import annotations

from mangum import Mangum

from api.main import app

# ``lifespan="off"`` disables the ASGI lifespan protocol which is not
# supported in Vercel's serverless environment.
handler = Mangum(app, lifespan="off")
