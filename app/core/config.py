from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment configuration."""

    # Project metadata
    PROJECT_NAME: str = "Identity Verification System MVP"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Face Recognition
    FACE_MODEL: str = "buffalo_l"
    FACE_SIMILARITY_THRESHOLD: float = 0.40
    INSIGHTFACE_ROOT: str = str(Path.home() / ".insightface")
    ENFORCE_STRICT_FACE_QUALITY: bool = False
    CROWDED_MODE_ENABLED: bool = True
    ALLOW_MULTIPLE_DOCUMENT_FACES: bool = True
    MIN_FACE_HEIGHT_RATIO_LIVE: float = 0.08

    # OCR
    OCR_LANGUAGE: str = "en"
    OCR_USE_ANGLE_CLS: bool = True

    # Camera
    CAMERA_INDEX: int = 0
    MAX_CAMERAS_TO_CHECK: int = 5

    # Storage & Temporary Files
    DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    TEMP_DIR: Path = DATA_DIR / "temporary"

    # Temporary verification session TTL in seconds (e.g. 15 minutes)
    SESSION_TTL_SECONDS: int = 900

     # General
    app_name: str = Field(default="Reusable-OCR-Service", description="API application name")
    app_env: str = Field(default="development", description="Environment: development, production, test")
    app_debug: bool = Field(default=True, description="Debug mode")
    port: int = Field(default=8000, description="Port to listen on")
    host: str = Field(default="0.0.0.0", description="Host binding")

    # OCR Engine
    ocr_engine: str = Field(default="paddle", description="Selected OCR engine: paddle, mock")
    paddle_use_angle_cls: bool = Field(default=True, description="Enable text orientation angle classification")
    paddle_lang: str = Field(default="en", description="Default OCR language")
    paddle_use_gpu: bool = Field(default=False, description="Use GPU acceleration if available")
    paddle_det_db_thresh: float = Field(default=0.3, description="DB detection threshold")
    paddle_det_db_box_thresh: float = Field(default=0.5, description="DB box detection threshold")

    # Image Preprocessing
    enable_preprocessing: bool = Field(default=True, description="Enable OpenCV image preprocessing pipeline")
    deskew_enabled: bool = Field(default=True, description="Enable automatic document deskewing")
    clahe_enabled: bool = Field(default=True, description="Enable Contrast Limited Adaptive Histogram Equalization")
    denoise_enabled: bool = Field(default=False, description="Enable Gaussian / bilateral denoising")

    # Document Validation
    max_image_size_mb: int = Field(default=20, description="Max allowed file size in MB")
    allowed_extensions_str: str = Field(default="jpg,jpeg,png,webp,bmp,tiff,tif", alias="ALLOWED_EXTENSIONS")

    @property
    def allowed_extensions(self) -> List[str]:
        """Return list of allowed file extensions."""
        return [ext.strip().lower() for ext in self.allowed_extensions_str.split(",") if ext.strip()]


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
