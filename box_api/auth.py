"""Shared authentication for early admission and the OpenAPI security scheme."""
import secrets

from box_api.errors import ApiError


def validate_api_key(settings, supplied):
    if settings.api_key is None:
        return
    if len(supplied) != 1 or not secrets.compare_digest(
        supplied[0], settings.api_key.encode("ascii")
    ):
        raise ApiError(401, "Missing or invalid API key")
