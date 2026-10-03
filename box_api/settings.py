"""Environment configuration for the inference service."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model_path: str = str(ROOT / "models" / "best.pt")
    device: str = "cpu"
    workers: int = 2
    torch_threads: int = 1
    max_requests: int = 32
    max_images: int = 16
    max_file_bytes: int = 10 * 1024 * 1024
    max_body_bytes: int = 50 * 1024 * 1024
    max_pixels: int = 20_000_000
    queue_timeout: float = 60.0
    processing_timeout: float = 60.0
    upload_timeout: float = 60.0
    startup_timeout: float = 180.0
    temp_root: str | None = None
    api_key: str | None = field(default=None, repr=False)
    require_api_key: bool = False

    def __post_init__(self):
        if self.api_key is not None and (
            not self.api_key or not self.api_key.isascii() or any(c.isspace() for c in self.api_key)
        ):
            raise ValueError("BOX_API_KEY must be nonempty ASCII without whitespace")
        if self.require_api_key and (
            not self.api_key or len(self.api_key) < 32 or self.api_key.startswith("REPLACE_")
        ):
            raise ValueError("Hosted mode requires a random BOX_API_KEY of at least 32 characters")
        for name in (
            "workers", "torch_threads", "max_requests", "max_images",
            "max_file_bytes", "max_body_bytes", "max_pixels",
            "queue_timeout", "processing_timeout", "upload_timeout", "startup_timeout",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.device not in ("cpu", "mps"):
            raise ValueError("device must be cpu or mps")
        if self.device == "mps" and self.workers != 1:
            raise ValueError("MPS mode requires BOX_WORKERS=1")

    @classmethod
    def from_env(cls):
        values = {}
        for name, field in cls.__dataclass_fields__.items():
            value = os.getenv(f"BOX_{name.upper()}")
            if value is not None:
                if isinstance(field.default, bool):
                    if value.lower() not in ("true", "false", "1", "0"):
                        raise ValueError(f"BOX_{name.upper()} must be true or false")
                    value = value.lower() in ("true", "1")
                elif isinstance(field.default, int):
                    value = int(value)
                elif isinstance(field.default, float):
                    value = float(value)
                values[name] = value
        return cls(**values)
