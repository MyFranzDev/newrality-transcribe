"""API Key authentication for protected endpoints."""

from typing import Optional
from fastapi import Header, HTTPException, status
from app.config import settings


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Verify API key from X-API-Key header.

    Args:
        x_api_key: API key from request header

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    allowed_keys = settings.allowed_api_keys_list

    if x_api_key not in allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return x_api_key
