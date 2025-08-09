from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from services.camera_service import gen_frames

router = APIRouter(
    prefix="",  # Change from "/video" to "" to match original URL structure
    tags=["Video"]
)

@router.get("/video_feed")
def video_feed():
    """Video streaming endpoint"""
    try:
        return StreamingResponse(
            gen_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        print(f"Error in video_feed: {e}")
        return Response(content=f"Error: {str(e)}", media_type="text/plain")
