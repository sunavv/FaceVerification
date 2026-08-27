import time
import uuid
from typing import Dict, Any, Optional
import numpy as np

from app.core.config import settings
from app.core.exceptions import (
    SessionNotFoundException,
    AppException,
    ErrorCode,
)
from app.core.logging import logger
from app.services.face_service import face_service, FaceService
from app.services.ocr_service import ocr_service, OCRService
from app.utils.text import compare_names


class VerificationSession:
    """Represents a temporary verification state holder in memory."""

    def __init__(
        self,
        session_id: str,
        document_face_embedding: np.ndarray,
        extracted_name: Optional[str],
        all_ocr_lines: list,
        doc_face_bbox: list,
    ):
        self.session_id: str = session_id
        self.document_face_embedding: np.ndarray = document_face_embedding
        self.extracted_name: Optional[str] = extracted_name
        self.all_ocr_lines: list = all_ocr_lines
        self.doc_face_bbox: list = doc_face_bbox
        self.created_at: float = time.time()

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.created_at) > ttl_seconds


class VerificationService:
    """
    Verification orchestrator combining FaceService and OCRService.
    Evaluates both biometric similarity and textual name match independently.
    """

    def __init__(
        self,
        face_svc: Optional[FaceService] = None,
        ocr_svc: Optional[OCRService] = None,
    ):
        self.face_service = face_svc or face_service
        self.ocr_service = ocr_svc or ocr_service
        self._sessions: Dict[str, VerificationSession] = {}

    def _cleanup_expired_sessions(self) -> None:
        """Purge sessions exceeding configured TTL."""
        now = time.time()
        expired_ids = [
            sid for sid, sess in self._sessions.items()
            if (now - sess.created_at) > settings.SESSION_TTL_SECONDS
        ]
        for sid in expired_ids:
            self._sessions.pop(sid, None)

    def create_session(
        self,
        document_face_embedding: np.ndarray,
        extracted_name: Optional[str],
        all_ocr_lines: list,
        doc_face_bbox: list,
    ) -> str:
        """Create and store a temporary verification session."""
        self._cleanup_expired_sessions()
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = VerificationSession(
            session_id=session_id,
            document_face_embedding=document_face_embedding,
            extracted_name=extracted_name,
            all_ocr_lines=all_ocr_lines,
            doc_face_bbox=doc_face_bbox,
        )
        return session_id

    def get_session(self, session_id: str) -> VerificationSession:
        """Retrieve an active session or raise SessionNotFoundException."""
        self._cleanup_expired_sessions()
        session = self._sessions.get(session_id)
        if not session or session.is_expired(settings.SESSION_TTL_SECONDS):
            raise SessionNotFoundException("Verification session has expired or does not exist.")
        return session

    def process_document(self, document_image: np.ndarray) -> Dict[str, Any]:
        """
        Execute document analysis:
        1. Run Face detection & Embedding extraction (FaceService)
        2. Run OCR & Candidate name extraction (OCRService)
        3. Register temporary session
        """
        # Step 1: Face analysis on document
        face_result = self.face_service.process_document_face(document_image)

        # Step 2: OCR analysis on document
        ocr_result = self.ocr_service.extract_text(document_image)

        # Step 3: Create session for subsequent live camera verification
        session_id = self.create_session(
            document_face_embedding=face_result["embedding"],
            extracted_name=ocr_result.get("extracted_name"),
            all_ocr_lines=ocr_result.get("lines", []),
            doc_face_bbox=face_result.get("bbox", []),
        )

        return {
            "face_detected": face_result["face_detected"],
            "face_count": face_result["face_count"],
            "bbox": face_result["bbox"],
            "det_score": face_result["det_score"],
            "name": ocr_result.get("extracted_name"),
            "all_extracted_text": ocr_result.get("lines", []),
            "session_id": session_id,
        }

    def verify_live_against_session(
        self,
        session_id: str,
        live_image: np.ndarray,
        expected_name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Verify a live camera image against an active document session:
        1. Extract live face embedding
        2. Compute cosine similarity
        3. Match extracted document name against expected name
        4. Return full multi-factor verification breakdown
        """
        session = self.get_session(session_id)
        
        # 1. Process live face
        live_face_result = self.face_service.process_live_face(live_image)

        # 2. Compute face similarity
        sim_result = self.face_service.compute_cosine_similarity(
            emb1=session.document_face_embedding,
            emb2=live_face_result["embedding"],
            threshold=threshold,
        )

        # 3. Name comparison logic
        target_name = expected_name if (expected_name and expected_name.strip()) else session.extracted_name
        
        if session.extracted_name and target_name:
            name_match, name_sim = compare_names(session.extracted_name, target_name)
        elif not session.extracted_name and not target_name:
            # Both empty/unspecified
            name_match = True
        else:
            # One exists, other does not
            name_match = False

        # 4. Final verification logic: name_match == True AND face_match == True
        face_match = sim_result["face_match"]
        final_verified = bool(name_match and face_match)

        return {
            "document_valid": True,
            "document_face_detected": True,
            "document_face_count": 1,
            "ocr_success": bool(session.extracted_name or session.all_ocr_lines),
            "extracted_document_name": session.extracted_name,
            
            "expected_name": target_name,
            "name_match": name_match,
            
            "live_face_detected": live_face_result["face_detected"],
            "live_face_count": live_face_result["face_count"],
            
            "similarity": sim_result["similarity"],
            "threshold": sim_result["threshold"],
            "face_match": face_match,
            
            "verified": final_verified,
        }

    def verify_direct_images(
        self,
        document_image: np.ndarray,
        live_image: np.ndarray,
        expected_name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Direct verification comparing two static images (used by CLI and test scripts).
        """
        # Process document
        doc_face = self.face_service.process_document_face(document_image)
        ocr_result = self.ocr_service.extract_text(document_image)
        extracted_name = ocr_result.get("extracted_name")

        # Process live
        live_face = self.face_service.process_live_face(live_image)

        # Compare face embeddings
        sim_result = self.face_service.compute_cosine_similarity(
            emb1=doc_face["embedding"],
            emb2=live_face["embedding"],
            threshold=threshold,
        )

        # Name match
        target_name = expected_name if (expected_name and expected_name.strip()) else extracted_name
        if extracted_name and target_name:
            name_match, _ = compare_names(extracted_name, target_name)
        elif not extracted_name and not target_name:
            name_match = True
        else:
            name_match = False

        face_match = sim_result["face_match"]
        final_verified = bool(name_match and face_match)

        return {
            "document_valid": True,
            "document_face_detected": True,
            "document_face_count": doc_face["face_count"],
            "ocr_success": ocr_result["success"],
            "extracted_document_name": extracted_name,
            
            "expected_name": target_name,
            "name_match": name_match,
            
            "live_face_detected": live_face["face_detected"],
            "live_face_count": live_face["face_count"],
            
            "similarity": sim_result["similarity"],
            "threshold": sim_result["threshold"],
            "face_match": face_match,
            
            "verified": final_verified,
        }


# Singleton instance
verification_service = VerificationService()
