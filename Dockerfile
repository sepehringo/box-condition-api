FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/matplotlib YOLO_CONFIG_DIR=/tmp/ultralytics \
    BOX_DEVICE=cpu BOX_WORKERS=2 BOX_TORCH_THREADS=1 BOX_REQUIRE_API_KEY=true

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY requirements.txt requirements-api.txt ./
# CPU wheels avoid the CUDA dependencies of Linux's default PyTorch distribution.
RUN python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements-api.txt
COPY box_api ./box_api
COPY inference_core.py ./
COPY models/best.pt ./models/best.pt
USER app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "box_api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "120"]
