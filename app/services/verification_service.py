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
from app.services.face_recognition_service import face_recognition_service, FaceRecognitionService
from app.services.ocr_service import ocr_service, OCRService
from app.utils.text import compare_names


class VerificationSession:
    """
    Ephemeral in-memory session holding the reference face embedding and OCR data
    strictly for the duration of a single 1:1 verification transaction.
    """

    def __init__(
        self,
        session_id: str,
        document_face_embedding: np.ndarray,
        extracted_name: Optional[str],
        all_ocr_lines: list,
        doc_face_bbox: list,
        quality: str = "GOOD",
        quality_details: Optional[dict] = None,
        face_image_base64: Optional[str] = None,
        raw_face_image_base64: Optional[str] = None,
        face_crop_width: int = 0,
        face_crop_height: int = 0,
    ):
        self.session_id: str = session_id
        self.document_face_embedding: np.ndarray = document_face_embedding
        self.extracted_name: Optional[str] = extracted_name
        self.all_ocr_lines: list = all_ocr_lines
        self.doc_face_bbox: list = doc_face_bbox
        self.quality: str = quality
        self.quality_details: dict = quality_details or {}
        self.face_image_base64: Optional[str] = face_image_base64
        self.raw_face_image_base64: Optional[str] = raw_face_image_base64
        self.face_crop_width: int = face_crop_width
        self.face_crop_height: int = face_crop_height
        self.created_at: float = time.time()

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.created_at) > ttl_seconds


class VerificationService:
    """
    Identity Verification Orchestrator performing strict 1:1 identity verification.
    
    Orchestration:
    1. Extracts reference embedding & name from uploaded identity document.
    2. Holds session in volatile memory (strictly ephemeral, zero persistent gallery).
    3. Performs 1:1 comparison against live camera face using deep cosine similarity.
    4. Evaluates independent name matching.
    5. Returns unified result: verified = (name_match == True) AND (face_match == True).
    """

    def __init__(
        self,
        face_svc: Optional[FaceRecognitionService] = None,
        ocr_svc: Optional[OCRService] = None,
    ):
        self.face_service = face_svc or face_recognition_service
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
        quality: str = "GOOD",
        quality_details: Optional[dict] = None,
        face_image_base64: Optional[str] = None,
        raw_face_image_base64: Optional[str] = None,
        face_crop_width: int = 0,
        face_crop_height: int = 0,
    ) -> str:
        """Create and store a temporary verification session in memory."""
        self._cleanup_expired_sessions()
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = VerificationSession(
            session_id=session_id,
            document_face_embedding=document_face_embedding,
            extracted_name=extracted_name,
            all_ocr_lines=all_ocr_lines,
            doc_face_bbox=doc_face_bbox,
            quality=quality,
            quality_details=quality_details,
            face_image_base64=face_image_base64,
            raw_face_image_base64=raw_face_image_base64,
            face_crop_width=face_crop_width,
            face_crop_height=face_crop_height,
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
        1. Run Face detection, quality evaluation & 512-D embedding extraction.
        2. Run OCR & candidate name extraction.
        3. Register fresh ephemeral session for 1:1 live comparison.
        """
        # Step 1: Face analysis & quality check on document
        face_result = self.face_service.process_document_face(
            document_image,
            enforce_quality=settings.ENFORCE_STRICT_FACE_QUALITY,
        )

        # Step 2: OCR analysis on document
        ocr_result = self.ocr_service.extract_text(document_image)

        # Step 3: Create session for subsequent live camera verification
        session_id = self.create_session(
            document_face_embedding=face_result["embedding"],
            extracted_name=ocr_result.get("extracted_name"),
            all_ocr_lines=ocr_result.get("lines", []),
            doc_face_bbox=face_result.get("bbox", []),
            quality=face_result.get("quality", "GOOD"),
            quality_details=face_result.get("quality_details", {}),
            face_image_base64=face_result.get("face_image_base64"),
            raw_face_image_base64=face_result.get("raw_face_image_base64"),
            face_crop_width=face_result.get("face_crop_width", 0),
            face_crop_height=face_result.get("face_crop_height", 0),
        )

        return {
            "face_detected": face_result["face_detected"],
            "face_count": face_result["face_count"],
            "bbox": face_result["bbox"],
            "det_score": face_result["det_score"],
            "face_quality": face_result.get("quality", "GOOD"),
            "quality_details": face_result.get("quality_details", {}),
            "name": ocr_result.get("extracted_name"),
            "all_extracted_text": ocr_result.get("lines", []),
            "session_id": session_id,
            "face_image_base64": face_result.get("face_image_base64"),
            "raw_face_image_base64": face_result.get("raw_face_image_base64"),
            "face_crop_width": face_result.get("face_crop_width"),
            "face_crop_height": face_result.get("face_crop_height"),
            "enhanced_resolution": face_result.get("enhanced_resolution", [256, 256]),
        }

    def verify_live_against_session(
        self,
        session_id: str,
        live_image: np.ndarray,
        expected_name: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Verify a live camera image strictly 1:1 against an active document session:
        1. Extract live face embedding & evaluate live face quality.
        2. Compute age-robust cosine similarity against the session document embedding.
        3. Match extracted document name against expected name.
        4. Return full multi-factor verification breakdown.
        """
        session = self.get_session(session_id)
        
        # 1. Process live face & check quality
        live_face_result = self.face_service.process_live_face(
            live_image,
            enforce_quality=settings.ENFORCE_STRICT_FACE_QUALITY,
        )

        # 2. Compute face similarity against reference embedding ONLY
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
            name_match = True
        else:
            name_match = False

        # 4. Final verification logic: name_match == True AND face_match == True
        face_match = sim_result["face_match"]
        final_verified = bool(name_match and face_match)

        return {
            "reference_face_detected": True,
            "reference_face_count": 1,
            "reference_face_quality": session.quality,
            "reference_face_quality_details": session.quality_details,
            "reference_face_image_base64": session.face_image_base64,
            "reference_raw_face_image_base64": session.raw_face_image_base64,
            "reference_face_resolution": {
                "width": session.face_crop_width,
                "height": session.face_crop_height,
            },

            "live_face_detected": live_face_result["face_detected"],
            "live_face_count": live_face_result["face_count"],
            "live_face_quality": live_face_result.get("quality", "GOOD"),
            "live_face_quality_details": live_face_result.get("quality_details", {}),
            "live_face_image_base64": live_face_result.get("face_image_base64"),
            "live_raw_face_image_base64": live_face_result.get("raw_face_image_base64"),
            "live_face_resolution": {
                "width": live_face_result.get("face_crop_width", 0),
                "height": live_face_result.get("face_crop_height", 0),
            },

            "ocr_success": bool(session.extracted_name or session.all_ocr_lines),
            "extracted_document_name": session.extracted_name,
            "expected_name": target_name,
            "name_match": name_match,

            "similarity": sim_result["similarity"],
            "threshold": sim_result["threshold"],
            "face_match": face_match,

            "document_valid": True,
            "document_face_detected": True,
            "document_face_count": 1,
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
        Direct 1:1 verification comparing two static images (used by CLI and benchmark tests).
        """
        # Process document
        doc_face = self.face_service.process_document_face(document_image, enforce_quality=False)
        ocr_result = self.ocr_service.extract_text(document_image)
        extracted_name = ocr_result.get("extracted_name")

        # Process live
        live_face = self.face_service.process_live_face(live_image, enforce_quality=False)

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
            "reference_face_detected": doc_face["face_detected"],
            "reference_face_count": doc_face["face_count"],
            "reference_face_quality": doc_face.get("quality", "GOOD"),
            "reference_face_quality_details": doc_face.get("quality_details", {}),
            "reference_face_image_base64": doc_face.get("face_image_base64"),
            "reference_raw_face_image_base64": doc_face.get("raw_face_image_base64"),
            "reference_face_resolution": {
                "width": doc_face.get("face_crop_width", 0),
                "height": doc_face.get("face_crop_height", 0),
            },

            "live_face_detected": live_face["face_detected"],
            "live_face_count": live_face["face_count"],
            "live_face_quality": live_face.get("quality", "GOOD"),
            "live_face_quality_details": live_face.get("quality_details", {}),
            "live_face_image_base64": live_face.get("face_image_base64"),
            "live_raw_face_image_base64": live_face.get("raw_face_image_base64"),
            "live_face_resolution": {
                "width": live_face.get("face_crop_width", 0),
                "height": live_face.get("face_crop_height", 0),
            },

            "ocr_success": ocr_result["success"],
            "extracted_document_name": extracted_name,
            "expected_name": target_name,
            "name_match": name_match,

            "similarity": sim_result["similarity"],
            "threshold": sim_result["threshold"],
            "face_match": face_match,

            "document_valid": True,
            "document_face_detected": True,
            "document_face_count": doc_face["face_count"],
            "verified": final_verified,
        }


# Singleton instance
verification_service = VerificationService()
