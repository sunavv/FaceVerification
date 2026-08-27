import cv2
import numpy as np
import threading
import time
from typing import List, Dict, Any, Optional, Generator
from app.core.config import settings
from app.core.logging import logger
from app.services.face_service import face_service


class CameraService:
    """
    OpenCV Camera Service managing webcam access, camera enumeration,
    frame capture, and real-time annotated video streaming.
    """

    def __init__(self):
        self.active_camera_index: int = settings.CAMERA_INDEX
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._is_streaming: bool = False

    def list_available_cameras(self, max_to_check: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Probe connected cameras and return available indices.
        Uses non-blocking or fast check.
        """
        limit = max_to_check or settings.MAX_CAMERAS_TO_CHECK
        available = []
        
        # Suppress OpenCV terminal warnings during device index scanning
        try:
            cv2.setLogLevel(0)
        except Exception:
            pass

        for idx in range(limit):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION) if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.VideoCapture(idx)
                if cap is not None and cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append({
                            "index": idx,
                            "name": f"Camera {idx}",
                            "is_default": idx == self.active_camera_index,
                        })
                    cap.release()
            except Exception:
                pass

        # If none found via read, but system might still have default
        if not available:
            available.append({
                "index": 0,
                "name": "Default Camera (Index 0)",
                "is_default": True,
            })

        return available

    def set_camera_index(self, index: int) -> bool:
        """Change the active camera index and reset the capture device."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self.active_camera_index = index
            return True

    def get_capture(self) -> cv2.VideoCapture:
        """Get or initialize the OpenCV VideoCapture device."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.active_camera_index)
            # Configure standard resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self._cap

    def capture_single_frame(self) -> np.ndarray:
        """Capture a single frame from the currently active camera."""
        with self._lock:
            cap = self.get_capture()
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera index {self.active_camera_index}.")
            
            # Read a fresh frame
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to grab camera frame.")
            return frame.copy()

    def release(self) -> None:
        """Release the camera device."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def generate_mjpeg_stream(
        self,
        reference_embedding: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Generator[bytes, None, None]:
        """
        Generate an MJPEG stream with real-time face bounding box overlays,
        face count indicators, and similarity score overlay when reference_embedding is provided.
        """
        thresh = threshold or settings.FACE_SIMILARITY_THRESHOLD
        
        while True:
            frame = None
            try:
                with self._lock:
                    cap = self.get_capture()
                    if not cap.isOpened():
                        time.sleep(0.1)
                        continue
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.05)
                        continue
                    frame = frame.copy()

                # Process face detections for display
                faces = face_service.detect_all_faces(frame)
                face_count = len(faces)

                h, w = frame.shape[:2]

                # Draw status banner at top
                cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)

                if face_count == 0:
                    status_text = "FACE STATUS: NO FACE DETECTED"
                    status_color = (0, 165, 255)  # Orange
                elif face_count == 1:
                    status_text = "FACE STATUS: 1 FACE (OK)"
                    status_color = (0, 255, 0)  # Green
                else:
                    status_text = f"FACE STATUS: {face_count} FACES (MULTIPLE - REJECT)"
                    status_color = (0, 0, 255)  # Red

                cv2.putText(frame, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

                # Annotate each face
                for i, face in enumerate(faces):
                    bbox = [int(v) for v in face.bbox[:4]]
                    x1, y1, x2, y2 = bbox

                    box_color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # If exactly 1 face and reference embedding exists, compute similarity
                    if face_count == 1 and reference_embedding is not None and face.embedding is not None:
                        sim_res = face_service.compute_cosine_similarity(
                            reference_embedding,
                            face.embedding,
                            threshold=thresh,
                        )
                        sim = sim_res["similarity"]
                        matched = sim_res["face_match"]

                        sim_label = f"Sim: {sim:.4f} | {'MATCH' if matched else 'NO MATCH'}"
                        label_bg_color = (0, 180, 0) if matched else (0, 0, 200)

                        cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + 220, y1), label_bg_color, -1)
                        cv2.putText(
                            frame,
                            sim_label,
                            (x1 + 5, max(15, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                # Encode frame to JPEG
                success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not success:
                    continue

                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                logger.error(f"Error in camera streaming loop: {str(e)}")
                time.sleep(0.1)


# Singleton instance
camera_service = CameraService()
