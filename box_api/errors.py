class ApiError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code, detail)
        self.status_code = status_code
        self.detail = detail


class InvalidImage(ValueError):
    """Picklable validation error raised by an inference worker."""
