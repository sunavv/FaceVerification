import pytest
import numpy as np

from app.services.face_service import FaceService


class TestFaceSimilarity:
    @pytest.fixture
    def face_service(self):
        svc = FaceService()
        svc._initialized = True
        return svc

    def test_same_person_embedding_similarity(self, face_service):
        base_emb = np.random.randn(512).astype(np.float32)
        # Small perturbation simulating different lighting / pose
        noise = np.random.randn(512).astype(np.float32) * 0.1
        same_person_emb = base_emb + noise

        res = face_service.compute_cosine_similarity(base_emb, same_person_emb, threshold=0.65)
        
        assert res["similarity"] > 0.85
        assert res["face_match"] is True
        assert res["threshold"] == 0.65

    def test_identical_embedding_similarity(self, face_service):
        emb = np.random.randn(512).astype(np.float32)
        res = face_service.compute_cosine_similarity(emb, emb, threshold=0.65)
        assert np.isclose(res["similarity"], 1.0, atol=1e-3)
        assert res["face_match"] is True

    def test_different_person_embedding_similarity(self, face_service):
        np.random.seed(42)
        emb_person_a = np.random.randn(512).astype(np.float32)
        emb_person_b = np.random.randn(512).astype(np.float32)

        res = face_service.compute_cosine_similarity(emb_person_a, emb_person_b, threshold=0.65)
        
        # Uncorrelated 512-D Gaussian vectors have cosine similarity around 0.0 (+- 0.15)
        assert res["similarity"] < 0.65
        assert res["face_match"] is False

    def test_custom_threshold_override(self, face_service):
        base = np.ones(512, dtype=np.float32)
        target = np.ones(512, dtype=np.float32)

        # Identical vector has similarity 1.0
        res = face_service.compute_cosine_similarity(base, target, threshold=0.99)
        assert res["face_match"] is True
        assert res["threshold"] == 0.99

        # If we supply a perpendicular vector
        ortho = np.zeros(512, dtype=np.float32)
        ortho[0] = 1.0
        base_other = np.zeros(512, dtype=np.float32)
        base_other[1] = 1.0

        res_ortho = face_service.compute_cosine_similarity(ortho, base_other, threshold=0.5)
        assert np.isclose(res_ortho["similarity"], 0.0, atol=1e-3)
        assert res_ortho["face_match"] is False
