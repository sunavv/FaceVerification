from pathlib import Path
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
