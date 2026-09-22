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
    FaceQualityTooLowException,
    AppException,
    ErrorCode,
)
from app.core.logging import logger
from app.utils.image import (
    align_face_5landmarks,
    normalize_face_lighting,
    enhance_image_for_detection,
    assess_face_quality,
    crop_face_bbox,
    enhance_and_upscale_face_crop,
    encode_image_to_base64_data_uri,
)


class FaceRecognitionService:
    """
    Isolated Age-Robust Face Recognition Service.
    
    Combines:
    1. SCRFD-10G: High-precision face & 5-point landmark detection.
    2. Face Quality & Usability Assessment: Rejects blurry, dark, tiny, or clipped faces.
    3. Canonical 5-Landmark Alignment: Warps face to standard 112x112 ArcFace template.
    4. ArcFace (w600k_r50): Deep angular margin embedding extracting 512-D cranial/facial bone structure invariant to age, hairstyle, and lighting.
    5. Geodesic Hypersphere Cosine Similarity: Robust 1:1 similarity matching.
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

            logger.info(f"Initializing FaceRecognitionService with model '{self.model_name}' (root: {self.root_dir})...")
            
            providers = ["CPUExecutionProvider"]

            self._app = FaceAnalysis(
                name=self.model_name,
                root=self.root_dir,
                providers=providers,
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            self._initialized = True
            logger.info("FaceRecognitionService pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize FaceRecognitionService: {str(e)}")
            raise AppException(
                error_code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=f"Failed to initialize face recognition engine: {str(e)}",
                status_code=500,
            )

    @property
    def app(self):
        if not self._initialized:
            self.initialize()
        return self._app

    def _extract_faces_from_detections(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        kpss: Optional[np.ndarray] = None,
    ) -> List[Any]:
        """Run landmark & recognition models on detected bboxes and return Face objects."""
        from insightface.app.common import Face

        if bboxes is None or len(bboxes) == 0:
            return []

        faces = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = float(bboxes[i, 4])
            kps = kpss[i] if (kpss is not None and len(kpss) > i) else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.app.models.items():
                if taskname == "detection":
                    continue
                model.get(image, face)
            faces.append(face)
        return faces

    def detect_all_faces(self, image: np.ndarray) -> List[Any]:
        """
        Multi-Scale & Robust Face Detection for Real Documents & Live Feeds:
        1. Pass 1 (Standard 640x640): Fast global detection for standard photos & live frames.
        2. Pass 2 (High-Res 1280x1280): Detects small ID photo crops on large scanned documents (e.g. 1500-3000px).
        3. Pass 3 (Overlapping Tiled Quadrants): Searches document quadrants at high effective resolution.
        4. Pass 4 (Adaptive Contrast & Unsharp Mask): Recovers faces in blurry, noisy, or low-contrast scans.
        """
        if image is None or image.size == 0:
            return []

        try:
            # Pass 1: Standard global detection
            faces = self.app.get(image)
            if faces:
                return faces

            h, w = image.shape[:2]
            max_dim = max(h, w)

            # Pass 2: Higher resolution SCRFD pass for large scans
            if max_dim > 640:
                det_size = (1280, 1280)
                bboxes, kpss = self.app.det_model.detect(image, input_size=det_size)
                if bboxes is not None and bboxes.shape[0] > 0:
                    faces = self._extract_faces_from_detections(image, bboxes, kpss)
                    if faces:
                        return faces

            # Pass 3: Tiled Quadrants for large identity document scans
            if max_dim >= 800:
                quadrants = [
                    (0, 0, int(w * 0.65), int(h * 0.65)),               # Top-Left
                    (int(w * 0.35), 0, w, int(h * 0.65)),               # Top-Right
                    (0, int(h * 0.35), int(w * 0.65), h),               # Bottom-Left
                    (int(w * 0.35), int(h * 0.35), w, h),               # Bottom-Right
                ]
                all_bboxes = []
                all_kpss = []
                for (qx1, qy1, qx2, qy2) in quadrants:
                    quad_crop = image[qy1:qy2, qx1:qx2]
                    if quad_crop.size == 0:
                        continue
                    q_faces = self.app.get(quad_crop)
                    for f in q_faces:
                        bx1, by1, bx2, by2 = f.bbox
                        global_bbox = np.array([bx1 + qx1, by1 + qy1, bx2 + qx1, by2 + qy1, f.det_score], dtype=np.float32)
                        all_bboxes.append(global_bbox)
                        if getattr(f, "kps", None) is not None:
                            global_kps = f.kps.copy()
                            global_kps[:, 0] += qx1
                            global_kps[:, 1] += qy1
                            all_kpss.append(global_kps)
                        else:
                            all_kpss.append(np.zeros((5, 2), dtype=np.float32))

                if all_bboxes:
                    bboxes_arr = np.array(all_bboxes, dtype=np.float32)
                    kpss_arr = np.array(all_kpss, dtype=np.float32) if all_kpss else None
                    keep = self.app.det_model.nms(bboxes_arr)
                    bboxes_nms = bboxes_arr[keep]
                    kpss_nms = kpss_arr[keep] if kpss_arr is not None else None
                    faces = self._extract_faces_from_detections(image, bboxes_nms, kpss_nms)
                    if faces:
                        return faces

            # Pass 4: Adaptive contrast boost and unsharp filtering with lowered detection threshold
            enhanced = enhance_image_for_detection(image)
            det_size = (1280, 1280) if max_dim > 640 else (640, 640)
            orig_thresh = getattr(self.app.det_model, "det_thresh", 0.5)
            try:
                self.app.det_model.det_thresh = 0.30
                bboxes, kpss = self.app.det_model.detect(enhanced, input_size=det_size)
            finally:
                self.app.det_model.det_thresh = orig_thresh

            if bboxes is not None and bboxes.shape[0] > 0:
                faces = self._extract_faces_from_detections(image, bboxes, kpss)
                if faces:
                    return faces

            return []
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return []

    def process_document_face(
        self,
        image: np.ndarray,
        enforce_quality: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Process an identity document image:
        - Must contain exactly 1 face.
        - Evaluates face quality & usability.
        - Generates 512-D normalized age-robust facial embedding.
        """
        if enforce_quality is None:
            enforce_quality = settings.ENFORCE_STRICT_FACE_QUALITY

        faces = self.detect_all_faces(image)
        face_count = len(faces)

        if face_count == 0:
            raise DocumentFaceNotFoundException(
                "DOCUMENT_FACE_NOT_FOUND: No face detected in the identity document. Ensure the document photograph is clearly visible and not obstructed."
            )
        elif face_count > 1:
            raise MultipleDocumentFacesException(
                f"MULTIPLE_DOCUMENT_FACES: {face_count} faces detected. The document must contain exactly one face."
            )

        face = faces[0]
        bbox = [float(x) for x in face.bbox]
        kps = getattr(face, "kps", None)

        # Assess face quality
        quality_res = assess_face_quality(image, bbox, kps)
        
        if enforce_quality and not quality_res["is_usable"]:
            issues_str = ", ".join(quality_res["issues"])
            raise FaceQualityTooLowException(
                f"FACE_QUALITY_TOO_LOW: Document photo quality is insufficient for verification ({issues_str}).",
                details=quality_res,
            )

        norm_embedding = self.normalize_embedding(face.embedding)

        # Extract raw and enhanced/upscaled face crops
        raw_crop = crop_face_bbox(image, bbox, padding_ratio=0.18)
        enhanced_crop = enhance_and_upscale_face_crop(raw_crop, target_size=(256, 256))
        raw_b64 = encode_image_to_base64_data_uri(raw_crop)
        enhanced_b64 = encode_image_to_base64_data_uri(enhanced_crop)

        fw = max(0, int(bbox[2] - bbox[0]))
        fh = max(0, int(bbox[3] - bbox[1]))

        return {
            "face_detected": True,
            "face_count": 1,
            "bbox": bbox,
            "det_score": float(getattr(face, "det_score", 1.0)),
            "quality": quality_res["quality"],
            "quality_details": quality_res,
            "embedding": norm_embedding,
            "raw_face": face,
            "face_image_base64": enhanced_b64,
            "raw_face_image_base64": raw_b64,
            "face_crop_width": fw,
            "face_crop_height": fh,
            "enhanced_resolution": [256, 256],
        }

    def process_live_face(
        self,
        image: np.ndarray,
        enforce_quality: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Process a live camera frame:
        - Must contain exactly 1 face.
        - Evaluates face quality & usability.
        - Generates 512-D normalized age-robust facial embedding.
        """
        if enforce_quality is None:
            enforce_quality = settings.ENFORCE_STRICT_FACE_QUALITY

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
        bbox = [float(x) for x in face.bbox]
        kps = getattr(face, "kps", None)

        # Assess face quality
        quality_res = assess_face_quality(image, bbox, kps)

        if enforce_quality and not quality_res["is_usable"]:
            issues_str = ", ".join(quality_res["issues"])
            raise FaceQualityTooLowException(
                f"FACE_QUALITY_TOO_LOW: Live camera face quality is insufficient ({issues_str}). Please adjust lighting or distance.",
                details=quality_res,
            )

        norm_embedding = self.normalize_embedding(face.embedding)

        # Extract raw and enhanced/upscaled face crops
        raw_crop = crop_face_bbox(image, bbox, padding_ratio=0.18)
        enhanced_crop = enhance_and_upscale_face_crop(raw_crop, target_size=(256, 256))
        raw_b64 = encode_image_to_base64_data_uri(raw_crop)
        enhanced_b64 = encode_image_to_base64_data_uri(enhanced_crop)

        fw = max(0, int(bbox[2] - bbox[0]))
        fh = max(0, int(bbox[3] - bbox[1]))

        return {
            "face_detected": True,
            "face_count": 1,
            "bbox": bbox,
            "det_score": float(getattr(face, "det_score", 1.0)),
            "quality": quality_res["quality"],
            "quality_details": quality_res,
            "embedding": norm_embedding,
            "raw_face": face,
            "face_image_base64": enhanced_b64,
            "raw_face_image_base64": raw_b64,
            "face_crop_width": fw,
            "face_crop_height": fh,
            "enhanced_resolution": [256, 256],
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
        Compute cosine similarity between two 512-D facial embeddings.
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

        # Dot product of L2 normalized unit vectors is exact cosine similarity
        sim = float(np.dot(vec1, vec2))
        
        # Clamp to [-1.0, 1.0] for numeric stability
        sim = max(-1.0, min(1.0, sim))
        
        sim_rounded = round(sim, 4)
        is_match = sim >= thresh

        return {
            "similarity": sim_rounded,
            "threshold": round(thresh, 4),
            "face_match": is_match,
        }


# Singleton instance
face_recognition_service = FaceRecognitionService()
