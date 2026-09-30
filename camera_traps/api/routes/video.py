import base64
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from camera_traps.services.video import annotate, process, smooth
from camera_traps.settings import S

video_router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent.parent / "templates"))


@video_router.post(path="/predict-video", response_class=HTMLResponse)
def predict_video(request: Request, file: UploadFile = File(...)):

    if not file.filename or not file.content_type or not file.content_type.startswith("video/"):
        return HTMLResponse(content="Uploaded file is not a valid video.", status_code=400)

    # Read video file.
    try:
        video_bytes = file.file.read()
    finally:
        file.file.close()

    # Get the predictions.
    response = process(
        filename=file.filename,
        video_bytes=video_bytes,
        detector=request.app.state.detector,
        classifier=request.app.state.classifier,
        tracker=request.app.state.tracker,
        detector_threshold=S.DETECTOR_THRESHOLD,
    )

    if not response.predictions:
        templates.TemplateResponse(
            name="partials/video_result.html",
            request=request,
            context={
                "filename": file.filename,
                "has_detections": False,
                "video_b64": "data:video/mp4;base64," + base64.b64encode(video_bytes).decode("utf-8"),
                "summary": [],
            },
        )

    # Consolidate predictions.
    response = smooth(response=response, threshold=S.CLASSIFIER_THRESHOLD)

    # Render annotated video.
    video_bytes_output = annotate(video_bytes=video_bytes, response=response)

    # Generate distinct track summary list da response.predictions.
    seen_trackers = {}
    for pred in response.predictions:
        if pred.tracking_id not in seen_trackers:
            seen_trackers[pred.tracking_id] = pred.class_name

    return templates.TemplateResponse(
        name="partials/video_result.html",
        request=request,
        context={
            "filename": file.filename,
            "has_detections": True,
            "video_b64": "data:video/mp4;base64," + base64.b64encode(video_bytes_output).decode("utf-8"),
            "summary": [{"tracker_id": tid, "label": label} for tid, label in sorted(seen_trackers.items())],
        },
    )
