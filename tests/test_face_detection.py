import pytest
import numpy as np
from unittest.mock import MagicMock

from app.services.face_service import FaceService
from app.core.exceptions import (
    DocumentFaceNotFoundException,
    MultipleDocumentFacesException,
    LiveFaceNotFoundException,
    MultipleLiveFacesException,
)


class MockFace:
    def __init__(self, bbox=None, embedding=None, det_score=0.99):
        self.bbox = bbox if bbox is not None else np.array([50, 50, 200, 200])
        self.embedding = embedding if embedding is not None else np.random.randn(512).astype(np.float32)
        self.det_score = det_score


class TestFaceDetectionRules:
    @pytest.fixture
    def face_service(self):
        svc = FaceService()
        svc._initialized = True
        return svc

    def test_document_with_no_face_raises_exception(self, face_service):
        face_service.detect_all_faces = MagicMock(return_value=[])
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)

        with pytest.raises(DocumentFaceNotFoundException) as exc_info:
            face_service.process_document_face(dummy_img)

        assert "DOCUMENT_FACE_NOT_FOUND" in str(exc_info.value)

    def test_document_with_one_face_succeeds(self, face_service):
        mock_face = MockFace()
        face_service.detect_all_faces = MagicMock(return_value=[mock_face])
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)

        result = face_service.process_document_face(dummy_img)
        assert result["face_detected"] is True
        assert result["face_count"] == 1
        assert len(result["embedding"]) == 512
        # Check normalized L2 norm is ~1.0
        norm = np.linalg.norm(result["embedding"])
        assert np.isclose(norm, 1.0, atol=1e-4)

    def test_document_with_multiple_faces_raises_exception(self, face_service):
        mock_faces = [MockFace(), MockFace()]
        face_service.detect_all_faces = MagicMock(return_value=mock_faces)
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)

        with pytest.raises(MultipleDocumentFacesException) as exc_info:
            face_service.process_document_face(dummy_img)

        assert "MULTIPLE_DOCUMENT_FACES" in str(exc_info.value)

    def test_camera_frame_with_no_face_raises_exception(self, face_service):
        face_service.detect_all_faces = MagicMock(return_value=[])
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)

        with pytest.raises(LiveFaceNotFoundException) as exc_info:
            face_service.process_live_face(dummy_img)

        assert "LIVE_FACE_NOT_FOUND" in str(exc_info.value)

    def test_camera_frame_with_multiple_faces_raises_exception(self, face_service):
        mock_faces = [MockFace(), MockFace(), MockFace()]
        face_service.detect_all_faces = MagicMock(return_value=mock_faces)
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)

        with pytest.raises(MultipleLiveFacesException) as exc_info:
            face_service.process_live_face(dummy_img)

        assert "MULTIPLE_LIVE_FACES" in str(exc_info.value)

    def test_camera_frame_with_one_face_succeeds(self, face_service):
        mock_face = MockFace()
        face_service.detect_all_faces = MagicMock(return_value=[mock_face])
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)

        result = face_service.process_live_face(dummy_img)
        assert result["face_detected"] is True
        assert result["face_count"] == 1
        assert len(result["embedding"]) == 512
