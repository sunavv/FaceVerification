from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import (
    DocumentFaceNotFoundException,
    MultipleDocumentFacesException,
    LiveFaceNotFoundException,
    MultipleLiveFacesException,
    AppException,
    ErrorCode,
)
from app.core.logging import logger


class FaceService:
    """
    Decoupled face recognition and analysis service using InsightFace, SCRFD, and ArcFace.
    Provides face detection, normalized embedding extraction, and cosine similarity comparison.
    """

    def __init__(self, model_name: Optional[str] = None, root_dir: Optional[str] = None):
        self.model_name = model_name or settings.FACE_MODEL
        self.root_dir = root_dir or settings.INSIGHTFACE_ROOT
        self._app = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the InsightFace FaceAnalysis pipeline."""
        if self._initialized:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis

            logger.info(f"Initializing InsightFace model '{self.model_name}' (root: {self.root_dir})...")
            # Providers: prefer CoreMLExecutionProvider on macOS or CPUExecutionProvider
            providers = ["CPUExecutionProvider"]
            try:
                import onnxruntime
                available = onnxruntime.get_available_providers()
                if "CoreMLExecutionProvider" in available:
                    providers.insert(0, "CoreMLExecutionProvider")
            except Exception:
                pass

            self._app = FaceAnalysis(
                name=self.model_name,
                root=self.root_dir,
                providers=providers,
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            self._initialized = True
            logger.info("InsightFace FaceAnalysis pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {str(e)}")
            raise AppException(
                error_code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"Failed to initialize face analysis engine: {str(e)}",
                status_code=500,
            )

    @property
    def app(self):
        if not self._initialized:
            self.initialize()
        return self._app

    def detect_all_faces(self, image: np.ndarray) -> List[Any]:
        """Detect all faces in an image and return raw InsightFace face objects."""
        if image is None or image.size == 0:
            return []
        try:
            faces = self.app.get(image)
            return faces or []
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []

    def process_document_face(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an identity document image:
        - Must contain exactly 1 face.
        - Returns face metadata and normalized embedding vector.
        """
        faces = self.detect_all_faces(image)
        face_count = len(faces)

        if face_count == 0:
            raise DocumentFaceNotFoundException(
                "DOCUMENT_FACE_NOT_FOUND: No face detected in the identity document."
            )
        elif face_count > 1:
            raise MultipleDocumentFacesException(
                f"MULTIPLE_DOCUMENT_FACES: {face_count} faces detected. The document must contain exactly one face."
            )

        face = faces[0]
        embedding = face.embedding
        norm_embedding = self.normalize_embedding(embedding)

        return {
            "face_detected": True,
            "face_count": 1,
            "bbox": [float(x) for x in face.bbox],
            "det_score": float(face.det_score) if hasattr(face, "det_score") else 1.0,
            "embedding": norm_embedding,
            "raw_face": face,
        }

    def process_live_face(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process a live camera frame:
        - Must contain exactly 1 face.
        - Returns face metadata and normalized embedding vector.
        """
        faces = self.detect_all_faces(image)
        face_count = len(faces)

        if face_count == 0:
            raise LiveFaceNotFoundException(
                "LIVE_FACE_NOT_FOUND: No face detected in the live camera frame."
            )
        elif face_count > 1:
            raise MultipleLiveFacesException(
                f"MULTIPLE_LIVE_FACES: {face_count} faces detected in camera frame. Only one person must be visible."
            )

        face = faces[0]
        embedding = face.embedding
        norm_embedding = self.normalize_embedding(embedding)

        return {
            "face_detected": True,
            "face_count": 1,
            "bbox": [float(x) for x in face.bbox],
            "det_score": float(face.det_score) if hasattr(face, "det_score") else 1.0,
            "embedding": norm_embedding,
            "raw_face": face,
        }

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding vector."""
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            return emb / norm
        return emb

    def compute_cosine_similarity(
        self,
        emb1: Union[np.ndarray, list],
        emb2: Union[np.ndarray, list],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute cosine similarity between two face embeddings.
        Returns:
            {
                "similarity": float,
                "threshold": float,
                "face_match": bool
            }
        """
        thresh = threshold if threshold is not None else settings.FACE_SIMILARITY_THRESHOLD
        
        vec1 = self.normalize_embedding(np.array(emb1, dtype=np.float32))
        vec2 = self.normalize_embedding(np.array(emb2, dtype=np.float32))

        # Dot product of L2 normalized vectors is cosine similarity
        sim = float(np.dot(vec1, vec2))
        
        # Clamp to [-1.0, 1.0] for numeric stability
        sim = max(-1.0, min(1.0, sim))
        
        # Round to 4 decimal places
        sim_rounded = round(sim, 4)
        is_match = sim >= thresh

        return {
            "similarity": sim_rounded,
            "threshold": round(thresh, 4),
            "face_match": is_match,
        }


# Singleton instance
face_service = FaceService()
