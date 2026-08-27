from fastapi import APIRouter, Query, Response
from fastapi.responses import StreamingResponse
from typing import Optional, List
import numpy as np

from app.models.schemas import CameraListResponse, CameraInfo, BaseAPIResponse
from app.services.camera_service import camera_service
from app.services.verification_service import verification_service
from app.core.config import settings
from app.core.logging import logger

router = APIRouter(prefix="/cameras", tags=["Camera Management"])


@router.get("", response_model=CameraListResponse)
async def list_cameras():
    """Enumerate available system camera devices."""
    try:
        cams = camera_service.list_available_cameras()
        return CameraListResponse(
            success=True,
            cameras=[CameraInfo(**c) for c in cams],
            selected_camera=camera_service.active_camera_index,
        )
    except Exception as e:
        logger.error(f"Error listing cameras: {str(e)}")
        return CameraListResponse(
            success=False,
            cameras=[CameraInfo(index=0, name="Camera 0", is_default=True)],
            selected_camera=0,
        )


@router.post("/select")
async def select_camera(index: int = Query(..., description="Camera device index to activate")):
    """Switch active webcam device to the specified index."""
    success = camera_service.set_camera_index(index)
    return BaseAPIResponse(success=success)


@router.get("/stream")
async def video_stream(
    session_id: Optional[str] = Query(None, description="Optional active session ID for live comparison overlay"),
    threshold: Optional[float] = Query(None, description="Similarity threshold for overlay"),
):
    """
    Stream live video via MJPEG with real-time face detection bounding box
    and live similarity meter if session_id is provided.
    """
    ref_emb: Optional[np.ndarray] = None
    if session_id:
        try:
            session = verification_service.get_session(session_id)
            ref_emb = session.document_face_embedding
        except Exception:
            pass

    return StreamingResponse(
        camera_service.generate_mjpeg_stream(reference_embedding=ref_emb, threshold=threshold),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
