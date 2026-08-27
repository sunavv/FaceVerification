from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any

from app.models.schemas import DocumentAnalysisResponse, DocumentAnalysisData, ErrorDetail
from app.utils.image import decode_image_bytes, validate_image_extension
from app.services.verification_service import verification_service
from app.core.exceptions import AppException
from app.core.logging import logger

router = APIRouter(prefix="/document", tags=["Document Analysis"])


@router.post("/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyze an uploaded identity document:
    1. Validates image file format (JPG, JPEG, PNG, WEBP).
    2. Detects face count (must be exactly 1).
    3. Extracts normalized facial embedding.
    4. Runs PaddleOCR to extract text and candidate name.
    5. Saves temporary verification session and returns extracted data with session_id.
    """
    try:
        # Validate extension
        validate_image_extension(file.filename)

        # Read file bytes into memory
        contents = await file.read()
        image = decode_image_bytes(contents)

        # Process document through verification service
        result = verification_service.process_document(image)

        return DocumentAnalysisResponse(
            success=True,
            document=DocumentAnalysisData(
                face_detected=result["face_detected"],
                face_count=result["face_count"],
                name=result["name"],
                all_extracted_text=result["all_extracted_text"],
                session_id=result["session_id"],
            ),
        )
    except AppException as e:
        logger.warning(f"Document analysis domain exception: [{e.error_code.value}] {e.message}")
        return DocumentAnalysisResponse(
            success=False,
            error=ErrorDetail(code=e.error_code.value, message=e.message, details=e.details),
        )
    except Exception as e:
        logger.error(f"Unexpected error in /document/analyze: {str(e)}")
        return DocumentAnalysisResponse(
            success=False,
            error=ErrorDetail(
                code="DOCUMENT_INVALID",
                message="Failed to process identity document.",
                details={"info": str(e)},
            ),
        )
