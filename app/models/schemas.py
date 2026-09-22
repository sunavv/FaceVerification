from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class BaseAPIResponse(BaseModel):
    success: bool
    error: Optional[ErrorDetail] = None


class DocumentAnalysisData(BaseModel):
    face_detected: bool = Field(..., description="Whether a face was detected in the document")
    face_count: int = Field(..., description="Number of faces found in the document")
    name: Optional[str] = Field(None, description="Extracted full name from document")
    face_quality: Optional[str] = Field("GOOD", description="Quality assessment of document face (GOOD/ACCEPTABLE/POOR)")
    quality_details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Detailed quality metrics")
    all_extracted_text: Optional[List[str]] = Field(default_factory=list, description="All OCR text lines extracted")
    session_id: Optional[str] = Field(None, description="Verification session token for next step")
    face_image_base64: Optional[str] = Field(None, description="Enhanced upscaled (256x256) face crop Data URI")
    raw_face_image_base64: Optional[str] = Field(None, description="Raw face crop Data URI")
    face_crop_width: Optional[int] = Field(None, description="Original face crop pixel width")
    face_crop_height: Optional[int] = Field(None, description="Original face crop pixel height")
    enhanced_resolution: Optional[List[int]] = Field(default_factory=lambda: [256, 256], description="Resolution of enhanced crop")


class DocumentAnalysisResponse(BaseAPIResponse):
    document: Optional[DocumentAnalysisData] = None


class VerificationStartRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID from document analysis")
    expected_name: Optional[str] = Field(None, description="Optional expected name if overriding OCR name")


class VerificationResultData(BaseModel):
    # Reference / Document face checks
    reference_face_detected: bool = Field(..., description="Whether reference document contains a face")
    reference_face_count: int = Field(1, description="Count of faces in reference document")
    reference_face_quality: str = Field("GOOD", description="Quality rating of reference face (GOOD/ACCEPTABLE/POOR)")
    reference_face_quality_details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Quality metrics for reference face")
    reference_face_image_base64: Optional[str] = Field(None, description="Enhanced reference face crop Data URI")
    reference_raw_face_image_base64: Optional[str] = Field(None, description="Raw reference face crop Data URI")
    reference_face_resolution: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Dimensions of reference face crop")

    # Live face checks
    live_face_detected: bool = Field(..., description="Whether live camera frame contains a face")
    live_face_count: int = Field(1, description="Count of faces in live camera frame")
    live_face_quality: str = Field("GOOD", description="Quality rating of live face (GOOD/ACCEPTABLE/POOR)")
    live_face_quality_details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Quality metrics for live face")
    live_face_image_base64: Optional[str] = Field(None, description="Enhanced live camera face crop Data URI")
    live_raw_face_image_base64: Optional[str] = Field(None, description="Raw live camera face crop Data URI")
    live_face_resolution: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Dimensions of live camera face crop")

    # OCR and Name checks
    ocr_success: bool = Field(..., description="Whether OCR extraction succeeded")
    extracted_document_name: Optional[str] = Field(None, description="Extracted document name")
    expected_name: Optional[str] = Field(None, description="Expected or queried name")
    name_match: bool = Field(..., description="Whether extracted name matches expected name")

    # Biometric similarity checks
    similarity: float = Field(..., description="Cosine similarity score between reference & live face embeddings")
    threshold: float = Field(..., description="Configured similarity threshold")
    face_match: bool = Field(..., description="Whether similarity meets or exceeds threshold")

    # Backward compatibility aliases
    document_valid: bool = Field(True, description="Document format and readability validity")
    document_face_detected: bool = Field(True, description="Alias for reference_face_detected")
    document_face_count: int = Field(1, description="Alias for reference_face_count")

    # Final decision: name_match == True AND face_match == True
    verified: bool = Field(..., description="Final verification result: name_match == True AND face_match == True")


class VerificationResponse(BaseAPIResponse):
    document: Optional[Dict[str, Any]] = None
    verification: Optional[VerificationResultData] = None


class CameraInfo(BaseModel):
    index: int
    name: str
    is_default: bool = False


class CameraListResponse(BaseAPIResponse):
    cameras: List[CameraInfo] = []
    selected_camera: int = 0


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    face_model: str
    similarity_threshold: float
    ocr_language: str
    cameras_available: int
