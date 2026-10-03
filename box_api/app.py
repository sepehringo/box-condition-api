"""Run with: uvicorn box_api.app:app --workers 1"""
import asyncio
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, RedirectResponse

from box_api.admission import AdmissionGate
from box_api.auth import validate_api_key
from box_api.errors import ApiError, InvalidImage
from box_api.middleware import AdmissionMiddleware
from box_api.scheduler import Scheduler
from box_api.settings import Settings
from box_api.worker import check_image

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    id: int
    class_id: int
    confidence: float
    box: BoundingBox

    # The public wire name matches the existing CLI's "class" field.
    class_name: str = Field(alias="class")
    model_config = ConfigDict(populate_by_name=True)


class ImageResult(BaseModel):
    index: int
    filename: str
    width: int
    height: int
    predictions: list[Detection]


class PredictionResponse(BaseModel):
    request_id: str
    results: list[ImageResult]


async def complete_io(function, *args):
    """Finish disk operations before cancellation can remove their files."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def save_uploads(form, settings, directory):
    files = form.getlist("files")
    if not files or len(files) > settings.max_images:
        raise ApiError(422, f"Supply 1 to {settings.max_images} files")
    if any(not isinstance(upload, UploadFile) for upload in files):
        raise ApiError(422, "The files field must contain image uploads")
    if any(key not in ("files", "confidence") for key in form):
        raise ApiError(422, "Only files and confidence fields are supported")
    confidences = form.getlist("confidence")
    if len(confidences) > 1:
        raise ApiError(422, "Supply confidence once")
    try:
        confidence = float(confidences[0]) if confidences else 0.5
    except (TypeError, ValueError):
        raise ApiError(422, "Confidence must be a number between 0 and 1")
    if not 0 <= confidence <= 1:
        raise ApiError(422, "Confidence must be a number between 0 and 1")
    paths, names = [], []
    for index, upload in enumerate(files):
        if upload.size is not None and upload.size > settings.max_file_bytes:
            raise ApiError(413, f"Image {index} exceeds the configured byte limit")
        # Internal paths never use a client-controlled filename.
        path = str(Path(directory.name) / f"{index}.upload")
        size = 0
        with open(path, "wb") as output:
            while chunk := await upload.read(64 * 1024):
                size += len(chunk)
                if size > settings.max_file_bytes:
                    raise ApiError(413, f"Image {index} exceeds the configured byte limit")
                await complete_io(output.write, chunk)
        try:
            await complete_io(check_image, path, settings.max_pixels)
        except InvalidImage as exc:
            raise ApiError(422, f"Image {index}: {exc}") from exc
        paths.append(path)
        names.append(upload.filename or f"image-{index}")
    return paths, names, confidence


async def wait_for_job(request, scheduler, job):
    try:
        while not job.future.done():
            if await request.is_disconnected():
                raise ApiError(499, "Client disconnected")
            await asyncio.wait({job.future}, timeout=0.1)
        return job.future.result()
    finally:
        # Worker tasks are never cancelled when the HTTP handler stops awaiting.
        scheduler.cancel(job)


def create_app(settings=None, scheduler_factory=Scheduler):
    settings = settings or Settings.from_env()
    gate = AdmissionGate(settings.max_requests)
    scheduler = scheduler_factory(settings)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await scheduler.start()
            yield
        finally:
            gate.close()
            await scheduler.close()

    application = FastAPI(title="Box condition inference", version="1.0.0", lifespan=lifespan)
    application.state.scheduler = scheduler
    application.state.gate = gate
    application.state.settings = settings
    application.add_middleware(AdmissionMiddleware, gate=gate, settings=settings)
    key_header = APIKeyHeader(name="X-API-Key", scheme_name="DemoAPIKey", auto_error=False)

    async def authenticate(request: Request, key=Depends(key_header)):
        # Middleware has already checked this before admission or multipart reads.
        # The dependency also declares the Swagger Authorize button.
        validate_api_key(settings, [
            value for name, value in request.scope.get("headers", [])
            if name.lower() == b"x-api-key"
        ])

    @application.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse("/docs")

    @application.exception_handler(ApiError)
    async def api_error(request, exc):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    @application.get("/ready")
    async def readiness():
        return JSONResponse({
            "ready": scheduler.ready and not scheduler.closing,
            "active_workers": scheduler.active,
            "admitted_requests": gate.in_use,
            "waiting_admission": sum(not future.done() for future in gate.waiters),
            "queued_requests": sum(job.started is None for job in scheduler.jobs),
        }, status_code=200 if scheduler.ready and not scheduler.closing else 503)

    @application.post(
        "/v1/predict", response_model=PredictionResponse,
        dependencies=[Depends(authenticate)],
        responses={
            401: {"description": "Missing or invalid API key"},
            408: {"description": "Upload timeout"},
            413: {"description": "Upload limit exceeded"},
            422: {"description": "Invalid input"},
            500: {"description": "Inference service failure"},
            504: {"description": "Queue or processing timeout"},
        },
        openapi_extra={"requestBody": {"required": True, "content": {
            "multipart/form-data": {"schema": {
                "type": "object", "required": ["files"],
                "properties": {
                    "files": {"type": "array", "minItems": 1,
                              "maxItems": settings.max_images,
                              "items": {"type": "string", "format": "binary"}},
                    "confidence": {"type": "number", "minimum": 0,
                                   "maximum": 1, "default": 0.5},
                },
            }},
        }}},
    )
    async def predict(request: Request):
        if not scheduler.ready:
            raise ApiError(500, "Inference workers are unavailable")
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type != "multipart/form-data":
            raise ApiError(422, "Use multipart/form-data with a files field")
        directory = tempfile.TemporaryDirectory(prefix="box-api-", dir=settings.temp_root)
        job = None
        try:
            try:
                async with request.form(max_files=settings.max_images, max_fields=1) as form:
                    paths, names, confidence = await save_uploads(form, settings, directory)
            except HTTPException as exc:
                if exc.detail == "box-upload-size-limit":
                    raise ApiError(413, "Request body exceeds the configured byte limit") from exc
                if exc.detail == "box-upload-timeout":
                    raise ApiError(408, "Upload exceeded the configured timeout") from exc
                if exc.status_code == 400:
                    raise ApiError(422, str(exc.detail)) from exc
                raise
            request_id = uuid.uuid4().hex
            job = scheduler.submit(
                paths, names, confidence, directory, request.state.lease, request_id,
                max(0, settings.queue_timeout - request.state.admission_wait),
                admission_wait=request.state.admission_wait,
            )
            results = await wait_for_job(request, scheduler, job)
            return {"request_id": request_id, "results": results}
        finally:
            if job is None:
                # Synchronous cleanup finishes before releasing upload admission.
                directory.cleanup()

    return application


app = create_app()
