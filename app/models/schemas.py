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
    all_extracted_text: Optional[List[str]] = Field(default_factory=list, description="All OCR text lines extracted")
    session_id: Optional[str] = Field(None, description="Verification session token for next step")


class DocumentAnalysisResponse(BaseAPIResponse):
    document: Optional[DocumentAnalysisData] = None


class VerificationStartRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID from document analysis")
    expected_name: Optional[str] = Field(None, description="Optional expected name if overriding OCR name")


class VerificationResultData(BaseModel):
    # Document checks
    document_valid: bool = Field(..., description="Document format and readability validity")
    document_face_detected: bool = Field(..., description="Whether document contains a face")
    document_face_count: int = Field(..., description="Count of faces detected in document")
    ocr_success: bool = Field(..., description="Whether OCR extraction succeeded")
    extracted_document_name: Optional[str] = Field(None, description="Extracted document name")
    
    # Name matching
    expected_name: Optional[str] = Field(None, description="Expected or queried name")
    name_match: bool = Field(..., description="Whether extracted name matches expected name")
    
    # Live face checks
    live_face_detected: bool = Field(..., description="Whether live camera frame contains a face")
    live_face_count: int = Field(..., description="Count of faces detected in live camera frame")
    
    # Face matching
    similarity: float = Field(..., description="Cosine similarity score between document & live face embeddings")
    threshold: float = Field(..., description="Configured similarity threshold")
    face_match: bool = Field(..., description="Whether similarity meets or exceeds threshold")
    
    # Final decision: name_match AND face_match
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
