"""FastAPI Router for OCR endpoints."""

import logging
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from app.models.response import OCRResponse, DocumentMetadata
from app.core.config import settings
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["OCR"])
ocr_service = OCRService()


@router.post("/ocr", response_model=OCRResponse)
async def extract_text(file: UploadFile = File(..., description="Document image file (PNG, JPG, WebP, BMP, TIFF)")):
    """Extract text, regions, bounding boxes, and confidence scores from an uploaded document image."""
    filename = file.filename or "unknown_document"

    try:
        content = await file.read()
        response = ocr_service.process_document(filename=filename, content=content)
        return response
    except ValueError as val_err:
        logger.warning("Validation error processing '%s': %s", filename, str(val_err))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "processing_time_ms": 0.0,
                "document": {
                    "filename": filename,
                    "type": file.content_type or "unknown",
                    "width": None,
                    "height": None,
                    "channels": None,
                    "size_bytes": 0,
                },
                "text": "",
                "lines": [],
                "engine": ocr_service.engine.get_name(),
                "error": str(val_err),
            },
        )
    except Exception as exc:
        logger.error("Unexpected error processing document '%s': %s", filename, str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "processing_time_ms": 0.0,
                "document": {
                    "filename": filename,
                    "type": file.content_type or "unknown",
                    "width": None,
                    "height": None,
                    "channels": None,
                    "size_bytes": 0,
                },
                "text": "",
                "lines": [],
                "engine": ocr_service.engine.get_name(),
                "error": f"OCR extraction failed: {str(exc)}",
            },
        )
