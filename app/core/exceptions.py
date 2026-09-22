from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(str, Enum):
    DOCUMENT_INVALID = "DOCUMENT_INVALID"
    DOCUMENT_FACE_NOT_FOUND = "DOCUMENT_FACE_NOT_FOUND"
    MULTIPLE_DOCUMENT_FACES = "MULTIPLE_DOCUMENT_FACES"
    LIVE_FACE_NOT_FOUND = "LIVE_FACE_NOT_FOUND"
    MULTIPLE_LIVE_FACES = "MULTIPLE_LIVE_FACES"
    FACE_QUALITY_TOO_LOW = "FACE_QUALITY_TOO_LOW"
    OCR_FAILED = "OCR_FAILED"
    NAME_NOT_FOUND = "NAME_NOT_FOUND"
    FACE_COMPARISON_FAILED = "FACE_COMPARISON_FAILED"
    FACE_BELOW_THRESHOLD = "FACE_BELOW_THRESHOLD"
    NAME_MISMATCH = "NAME_MISMATCH"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


class AppException(Exception):
    """Base application exception with error code, message, and HTTP status code."""

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class FaceQualityTooLowException(AppException):
    def __init__(self, message: str = "Detected face quality is too low for reliable verification.", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.FACE_QUALITY_TOO_LOW, message, status_code=422, details=details)


class DocumentInvalidException(AppException):
    def __init__(self, message: str = "Invalid or unreadable document format.", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.DOCUMENT_INVALID, message, status_code=400, details=details)


class DocumentFaceNotFoundException(AppException):
    def __init__(self, message: str = "No face detected in the identity document.", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.DOCUMENT_FACE_NOT_FOUND, message, status_code=422, details=details)


class MultipleDocumentFacesException(AppException):
    def __init__(self, message: str = "Multiple faces detected in the identity document. Exactly one face is required.", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.MULTIPLE_DOCUMENT_FACES, message, status_code=422, details=details)


class LiveFaceNotFoundException(AppException):
    def __init__(self, message: str = "No face detected in the camera frame. Please position your face clearly in view.", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.LIVE_FACE_NOT_FOUND, message, status_code=422, details=details)


class MultipleLiveFacesException(AppException):
    def __init__(self, message: str = "Multiple faces detected in camera frame. Please ensure only one person is in frame.", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.MULTIPLE_LIVE_FACES, message, status_code=422, details=details)


class OCRFailedException(AppException):
    def __init__(self, message: str = "OCR text extraction failed.", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.OCR_FAILED, message, status_code=422, details=details)


class NameNotFoundException(AppException):
    def __init__(self, message: str = "No name could be extracted from document text.", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.NAME_NOT_FOUND, message, status_code=422, details=details)


class SessionNotFoundException(AppException):
    def __init__(self, message: str = "Verification session not found or expired.", details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.SESSION_NOT_FOUND, message, status_code=404, details=details)
