from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

class AppError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

async def app_error_handler(request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.message}
    )

async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": "Validation error", "detail": exc.errors()}
    )