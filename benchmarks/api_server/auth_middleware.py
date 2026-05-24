"""Bearer-token auth dependency for the MaxText API server.

Behavior is governed by the `MAXTEXT_API_KEY` environment variable:

  - If unset (or empty): the dependency is a no-op. Useful for dev /
    in-VPC deployments behind IAP where IAM is the primary boundary.
  - If set: every request to a protected route must carry an
    `Authorization: Bearer <key>` header whose token matches exactly.
    A missing header returns 401; a mismatched token returns 403.

Also accepts the Anthropic-style `x-api-key` header so Claude Code and
the `anthropic` Python SDK work without extra config.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


async def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> None:
  expected = os.environ.get("MAXTEXT_API_KEY", "").strip()
  if not expected:
    return

  presented: str | None = None
  if authorization:
    parts = authorization.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
      presented = parts[1].strip()
  if presented is None and x_api_key:
    presented = x_api_key.strip()

  if not presented:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )
  if presented != expected:
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="invalid bearer token",
    )
