"""Password hashing (PBKDF2-SHA256) and signed access tokens (JWT, HS256)."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import secrets

import jwt

ITERATIONS = 600_000  # OWASP 2023+ guidance for PBKDF2-SHA256
ALGORITHM = "HS256"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def hash_password(password: str, iterations: int | None = None) -> str:
    iterations = iterations or ITERATIONS
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        algo, iterations, salt, digest = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt), int(iterations))
    return hmac.compare_digest(candidate, base64.b64decode(digest))


def create_token(user_id: int, secret: str, ttl: dt.timedelta) -> str:
    now = dt.datetime.now(dt.UTC)
    return jwt.encode({"sub": str(user_id), "iat": now, "exp": now + ttl}, secret, algorithm=ALGORITHM)


def read_token(token: str, secret: str) -> int | None:
    """User id from a valid, unexpired token; None otherwise."""
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM], options={"require": ["sub", "exp"]})
        return int(payload["sub"])
    except (jwt.PyJWTError, ValueError, KeyError):
        return None
