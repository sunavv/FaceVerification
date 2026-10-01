"""Pydantic response models for OCR REST API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Document metadata extracted from the uploaded file."""

    filename: str = Field(..., description="Original filename of the uploaded document")
    type: str = Field(..., description="Document MIME type or category, e.g. 'image/jpeg'")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    channels: Optional[int] = Field(None, description="Number of color channels")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")


class TextLine(BaseModel):
    """Individual recognized text line/region with spatial geometry and confidence."""

    text: str = Field(..., description="Recognized text string for this region")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score between 0.0 and 1.0")
    bbox: List[int] = Field(
        ...,
        description="Axis-aligned bounding box coordinates formatted as [x_min, y_min, x_max, y_max]",
    )
    polygon: Optional[List[List[int]]] = Field(
        None,
        description="4-point polygon bounding box coordinates formatted as [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]",
    )


class PreprocessingInfo(BaseModel):
    """Details about preprocessing steps performed on the image."""

    applied: bool = Field(..., description="Whether preprocessing was applied")
    deskew_angle: Optional[float] = Field(None, description="Deskew angle in degrees applied to the image")
    clahe_applied: Optional[bool] = Field(None, description="Whether CLAHE contrast enhancement was applied")
    denoise_applied: Optional[bool] = Field(None, description="Whether denoising filter was applied")


class OCRResponse(BaseModel):
    """Structured response returned by POST /api/v1/ocr."""

    success: bool = Field(..., description="True if OCR extraction completed successfully")
    processing_time_ms: float = Field(..., description="Total server-side processing time in milliseconds")
    document: DocumentMetadata = Field(..., description="Document metadata")
    text: str = Field(..., description="Complete combined recognized document text")
    lines: List[TextLine] = Field(default_factory=list, description="List of recognized text lines with spatial data")
    preprocessing: Optional[PreprocessingInfo] = Field(None, description="Image preprocessing telemetry")
    engine: str = Field(..., description="OCR engine identifier used for extraction")
    error: Optional[str] = Field(None, description="Error message if success is false")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status, e.g. 'healthy'")
    version: str = Field(..., description="API version")
    active_engine: str = Field(..., description="Active OCR engine backend")
    supported_engines: List[str] = Field(..., description="List of available OCR engine implementations")
    supported_formats: List[str] = Field(..., description="List of supported file extensions")
