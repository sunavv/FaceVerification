"""
Face Service module providing FaceRecognitionService and backward-compatible FaceService alias.
"""

from app.services.face_recognition_service import (
    FaceRecognitionService,
    face_recognition_service,
)

# Aliases for backward compatibility
FaceService = FaceRecognitionService
face_service = face_recognition_service

__all__ = [
    "FaceRecognitionService",
    "face_recognition_service",
    "FaceService",
    "face_service",
]
