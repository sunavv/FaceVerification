from fastapi import APIRouter, UploadFile, File, Form
from typing import Dict, Any, Optional

from app.models.schemas import (
    VerificationResponse,
    VerificationResultData,
    VerificationStartRequest,
    BaseAPIResponse,
    ErrorDetail,
)
from app.utils.image import decode_image_bytes, validate_image_extension
from app.services.verification_service import verification_service
from app.services.camera_service import camera_service
from app.core.exceptions import AppException
from app.core.logging import logger

router = APIRouter(prefix="/verification", tags=["Verification"])


@router.post("/start", response_model=BaseAPIResponse)
async def start_verification(request: VerificationStartRequest):
    """
    Validate that a session exists and prepare for camera comparison.
    """
    try:
        session = verification_service.get_session(request.session_id)
        if request.expected_name:
            session.extracted_name = request.expected_name.strip()
        return BaseAPIResponse(success=True)
    except AppException as e:
        return BaseAPIResponse(
            success=False,
            error=ErrorDetail(code=e.error_code.value, message=e.message, details=e.details),
        )
    except Exception as e:
        logger.error(f"Error in /verification/start: {str(e)}")
        return BaseAPIResponse(
            success=False,
            error=ErrorDetail(code="SESSION_NOT_FOUND", message="Session validation failed."),
        )


@router.post("/compare", response_model=VerificationResponse)
async def compare_verification(
    session_id: str = Form(...),
    expected_name: Optional[str] = Form(None),
    threshold: Optional[float] = Form(None),
    live_image: Optional[UploadFile] = File(None),
):
    """
    Compare a live camera frame (uploaded or captured on demand from active webcam)
    against the active session's document face embedding and extracted name.
    """
    try:
        # Obtain live frame either from upload or from active webcam
        if live_image is not None and live_image.filename:
            validate_image_extension(live_image.filename)
            bytes_data = await live_image.read()
            frame = decode_image_bytes(bytes_data)
        else:
            # Capture frame from active camera
            frame = camera_service.capture_single_frame()

        result = verification_service.verify_live_against_session(
            session_id=session_id,
            live_image=frame,
            expected_name=expected_name,
            threshold=threshold,
        )

        return VerificationResponse(
            success=True,
            document={
                "face_detected": result["reference_face_detected"],
                "face_count": result["reference_face_count"],
                "name": result["extracted_document_name"],
                "face_quality": result["reference_face_quality"],
            },
            verification=VerificationResultData(
                reference_face_detected=result["reference_face_detected"],
                reference_face_count=result["reference_face_count"],
                reference_face_quality=result["reference_face_quality"],
                reference_face_quality_details=result.get("reference_face_quality_details", {}),
                reference_face_image_base64=result.get("reference_face_image_base64"),
                reference_raw_face_image_base64=result.get("reference_raw_face_image_base64"),
                reference_face_resolution=result.get("reference_face_resolution", {}),
                live_face_detected=result["live_face_detected"],
                live_face_count=result["live_face_count"],
                live_face_quality=result["live_face_quality"],
                live_face_quality_details=result.get("live_face_quality_details", {}),
                live_face_image_base64=result.get("live_face_image_base64"),
                live_raw_face_image_base64=result.get("live_raw_face_image_base64"),
                live_face_resolution=result.get("live_face_resolution", {}),
                ocr_success=result["ocr_success"],
                extracted_document_name=result["extracted_document_name"],
                expected_name=result["expected_name"],
                name_match=result["name_match"],
                similarity=result["similarity"],
                threshold=result["threshold"],
                face_match=result["face_match"],
                document_valid=result["document_valid"],
                document_face_detected=result["reference_face_detected"],
                document_face_count=result["reference_face_count"],
                verified=result["verified"],
            ),
        )
    except AppException as e:
        logger.warning(f"Verification exception: [{e.error_code.value}] {e.message}")
        return VerificationResponse(
            success=False,
            error=ErrorDetail(code=e.error_code.value, message=e.message, details=e.details),
        )
    except Exception as e:
        logger.error(f"Unexpected error in /verification/compare: {str(e)}")
        return VerificationResponse(
            success=False,
            error=ErrorDetail(
                code="FACE_COMPARISON_FAILED",
                message="Verification comparison failed.",
                details={"info": str(e)},
            ),
        )
