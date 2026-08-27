from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import AppException
from app.models.schemas import HealthResponse, ErrorDetail
from app.api.routes_document import router as document_router
from app.api.routes_verification import router as verification_router
from app.api.routes_camera import router as camera_router
from app.services.camera_service import camera_service

# Base static dir
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: initial setup and graceful shutdown."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Configuration: FACE_MODEL={settings.FACE_MODEL}, SIM_THRESHOLD={settings.FACE_SIMILARITY_THRESHOLD}")
    logger.info(f"OCR_LANGUAGE={settings.OCR_LANGUAGE}, DEFAULT_CAMERA={settings.CAMERA_INDEX}")
    yield
    # Shutdown
    logger.info("Releasing camera and resources...")
    camera_service.release()
    logger.info("Shutdown complete.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Python-based Identity Verification MVP with InsightFace, PaddleOCR, and OpenCV.",
    lifespan=lifespan,
)

# CORS middleware for local development and future JavaFX desktop client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Domain Exception Handler
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code.value,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )


# Unhandled Exception Handler (prevent stack traces leaking)
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error: {str(exc)}", exc_info=False)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred. Please check server logs.",
                "details": {},
            },
        },
    )


# Health Check
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Healthcheck endpoint returning system diagnostics and configurations."""
    cams = camera_service.list_available_cameras()
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        face_model=settings.FACE_MODEL,
        similarity_threshold=settings.FACE_SIMILARITY_THRESHOLD,
        ocr_language=settings.OCR_LANGUAGE,
        cameras_available=len(cams),
    )


# Register API Routers
app.include_router(document_router)
app.include_router(verification_router)
app.include_router(camera_router)

# Mount static files for web UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def serve_ui():
        return FileResponse(str(STATIC_DIR / "index.html"))
