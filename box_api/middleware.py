"""Admission runs before FastAPI parses or buffers multipart uploads."""
import asyncio
import time

from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse

from box_api.errors import ApiError
from box_api.auth import validate_api_key


class AdmissionMiddleware:
    def __init__(self, app, gate, settings):
        self.app = app
        self.gate = gate
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].rstrip("/") != "/v1/predict":
            await self.app(scope, receive, send)
            return
        if scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        response_started = False

        async def tracked_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        lease = None
        start = time.monotonic()
        try:
            validate_api_key(self.settings, [
                value for name, value in scope.get("headers", [])
                if name.lower() == b"x-api-key"
            ])
            headers = dict(scope.get("headers", []))
            length = headers.get(b"content-length")
            if length is not None:
                try:
                    length = int(length)
                except ValueError:
                    raise ApiError(400, "Invalid Content-Length")
                if length < 0:
                    raise ApiError(400, "Invalid Content-Length")
                if length > self.settings.max_body_bytes:
                    raise ApiError(413, "Request body exceeds the configured byte limit")
            try:
                lease = await self.gate.acquire(self.settings.queue_timeout)
            except asyncio.TimeoutError:
                raise ApiError(504, "Queue wait exceeded the configured timeout")
            state = scope.setdefault("state", {})
            state["lease"] = lease
            state["admission_wait"] = time.monotonic() - start
            consumed = 0
            upload_deadline = time.monotonic() + self.settings.upload_timeout
            body_complete = False

            async def limited_receive():
                nonlocal consumed, body_complete
                if body_complete:
                    return await receive()
                remaining = upload_deadline - time.monotonic()
                if remaining <= 0:
                    raise MultiPartException("box-upload-timeout")
                try:
                    message = await asyncio.wait_for(receive(), remaining)
                except asyncio.TimeoutError:
                    raise MultiPartException("box-upload-timeout")
                if message["type"] == "http.request":
                    consumed += len(message.get("body", b""))
                    if consumed > self.settings.max_body_bytes:
                        raise MultiPartException("box-upload-size-limit")
                    body_complete = not message.get("more_body", False)
                return message

            await self.app(scope, limited_receive, tracked_send)
        except ApiError as exc:
            if response_started:
                raise
            await JSONResponse({"detail": exc.detail}, status_code=exc.status_code)(
                scope, receive, tracked_send
            )
        finally:
            if lease is not None:
                lease.release()
