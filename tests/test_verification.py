import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.verification_service import VerificationService
from app.services.face_service import FaceService
from app.services.ocr_service import OCRService
from app.core.exceptions import SessionNotFoundException


class MockFace:
    def __init__(self, embedding):
        self.bbox = [50.0, 50.0, 200.0, 200.0]
        self.embedding = embedding
        self.det_score = 0.99


@pytest.fixture
def mock_verification_service():
    face_svc = FaceService()
    face_svc._initialized = True
    
    ocr_svc = OCRService()
    ocr_svc._initialized = True

    service = VerificationService(face_svc=face_svc, ocr_svc=ocr_svc)
    return service, face_svc, ocr_svc


class TestVerificationOrchestration:
    def test_full_verification_success(self, mock_verification_service):
        service, face_svc, ocr_svc = mock_verification_service
        
        shared_embedding = np.random.randn(512).astype(np.float32)
        
        face_svc.process_document_face = MagicMock(return_value={
            "face_detected": True,
            "face_count": 1,
            "bbox": [50, 50, 200, 200],
            "det_score": 0.99,
            "embedding": FaceService.normalize_embedding(shared_embedding),
        })
        ocr_svc.extract_text = MagicMock(return_value={
            "success": True,
            "lines": ["NAME: SUNAV SHARMA", "DOB: 1995-01-01"],
            "extracted_name": "SUNAV SHARMA",
        })

        face_svc.process_live_face = MagicMock(return_value={
            "face_detected": True,
            "face_count": 1,
            "bbox": [60, 60, 210, 210],
            "det_score": 0.98,
            "embedding": FaceService.normalize_embedding(shared_embedding),
        })

        # Process document
        doc_result = service.process_document(np.zeros((100, 100, 3), dtype=np.uint8))
        session_id = doc_result["session_id"]

        # Verify live
        ver_result = service.verify_live_against_session(
            session_id=session_id,
            live_image=np.zeros((100, 100, 3), dtype=np.uint8),
            expected_name="SUNAV SHARMA",
            threshold=0.65,
        )

        assert ver_result["document_valid"] is True
        assert ver_result["name_match"] is True
        assert ver_result["face_match"] is True
        assert ver_result["similarity"] >= 0.99
        assert ver_result["verified"] is True

    def test_verification_name_mismatch_fails(self, mock_verification_service):
        service, face_svc, ocr_svc = mock_verification_service
        shared_embedding = np.random.randn(512).astype(np.float32)

        face_svc.process_document_face = MagicMock(return_value={
            "face_detected": True,
            "face_count": 1,
            "bbox": [50, 50, 200, 200],
            "det_score": 0.99,
            "embedding": FaceService.normalize_embedding(shared_embedding),
        })
        ocr_svc.extract_text = MagicMock(return_value={
            "success": True,
            "lines": ["NAME: SUNAV SHARMA"],
            "extracted_name": "SUNAV SHARMA",
        })
        face_svc.process_live_face = MagicMock(return_value={
            "face_detected": True,
            "face_count": 1,
            "bbox": [60, 60, 210, 210],
            "det_score": 0.98,
            "embedding": FaceService.normalize_embedding(shared_embedding),
        })

        doc_result = service.process_document(np.zeros((100, 100, 3), dtype=np.uint8))
        session_id = doc_result["session_id"]

        # Expecting different name
        ver_result = service.verify_live_against_session(
            session_id=session_id,
            live_image=np.zeros((100, 100, 3), dtype=np.uint8),
            expected_name="JOHN DOE",
            threshold=0.65,
        )

        assert ver_result["name_match"] is False
        assert ver_result["face_match"] is True
        assert ver_result["verified"] is False  # Must be False because name_match is False

    def test_verification_face_mismatch_fails(self, mock_verification_service):
        service, face_svc, ocr_svc = mock_verification_service
        
        doc_emb = np.zeros(512, dtype=np.float32)
        doc_emb[0] = 1.0
        live_emb = np.zeros(512, dtype=np.float32)
        live_emb[1] = 1.0

        face_svc.process_document_face = MagicMock(return_value={
            "face_detected": True,
            "face_count": 1,
            "bbox": [50, 50, 200, 200],
            "det_score": 0.99,
            "embedding": doc_emb,
        })
        ocr_svc.extract_text = MagicMock(return_value={
            "success": True,
            "lines": ["NAME: SUNAV SHARMA"],
            "extracted_name": "SUNAV SHARMA",
        })
        face_svc.process_live_face = MagicMock(return_value={
            "face_detected": True,
            "face_count": 1,
            "bbox": [60, 60, 210, 210],
            "det_score": 0.98,
            "embedding": live_emb,
        })

        doc_result = service.process_document(np.zeros((100, 100, 3), dtype=np.uint8))
        session_id = doc_result["session_id"]

        ver_result = service.verify_live_against_session(
            session_id=session_id,
            live_image=np.zeros((100, 100, 3), dtype=np.uint8),
            expected_name="SUNAV SHARMA",
            threshold=0.65,
        )

        assert ver_result["name_match"] is True
        assert ver_result["face_match"] is False
        assert ver_result["similarity"] < 0.65
        assert ver_result["verified"] is False

    def test_invalid_session_raises_exception(self, mock_verification_service):
        service, _, _ = mock_verification_service
        with pytest.raises(SessionNotFoundException):
            service.get_session("non-existent-uuid")


class TestFastAPIRoutes:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_endpoint(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["face_model"] == "buffalo_l"
        assert data["similarity_threshold"] == 0.65

    def test_cameras_endpoint(self, client):
        res = client.get("/cameras")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert isinstance(data["cameras"], list)
