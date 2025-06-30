from fastapi import FastAPI
from app.appError import AppError, app_error_handler, validation_exception_handler
from fastapi.exceptions import RequestValidationError
from app.routes import router as match_router
import json

app = FastAPI(
    title="Peer Tutor Microservice",
    description="A microservice that identifies the best peer tutors to help a student with a given academic topic using a trained AI model.",
    
)

app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

app.include_router(match_router)


