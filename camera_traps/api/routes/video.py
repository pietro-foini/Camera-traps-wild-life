import base64
import io
from pathlib import Path

import av
import numpy as np
import pandas as pd
import supervision as sv
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tqdm import tqdm
from trackers import SORTTracker

from camera_traps.db.models import VideoInputModel, VideoOutputModel
from camera_traps.schemas.base import BoundingBox, Detection, VideoDetectionResponse
from camera_traps.services.tracking import smooth_labels
from camera_traps.settings import S

router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent.parent / "templates"))


@router.post(path="/predict-video", response_class=HTMLResponse)
async def predict_video(request: Request, file: UploadFile = File(...)):

    if not file.filename or not file.content_type or not file.content_type.startswith("video/"):
        return HTMLResponse(content="Uploaded file is not a valid video.", status_code=400)

    video_record = VideoInputModel(filename=file.filename)
    saved_video = request.app.state.db.add_record(video_record)

    # Initialize tracker.
    tracker = SORTTracker()

    # Open video file.
    video_bytes = await file.read()
    container = av.open(io.BytesIO(video_bytes))

    tracking_data = []
    frame_id = 0
    try:
        with tqdm(total=container.streams.video[0].frames, unit="frame") as pbar:
            for frame in container.decode(video=0):
                frame = frame.to_ndarray(format="rgb24")

                # Run detector.
                predictions = request.app.state.detector.predict(frame)

                if len(predictions) > 0:
                    detections = sv.Detections(
                        xyxy=np.array([d.box.to_xyxy() for d in predictions]),
                        confidence=np.array([d.confidence for d in predictions]),
                        class_id=np.array([d.class_id for d in predictions]),
                    )
                else:
                    detections = sv.Detections.empty()

                # Run tracker.
                detections = tracker.update(detections)

                # Store detection and tracking results.
                if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                    for xyxy, confidence, class_id, tracker_id in zip(
                        detections.xyxy,
                        detections.confidence,
                        detections.class_id,
                        detections.tracker_id,
                    ):
                        if tracker_id == -1 or confidence < S.DETECTOR_THRESHOLD:
                            continue

                        # Run classifier.
                        xmin, ymin, xmax, ymax = xyxy
                        crop = frame[int(ymin) : int(ymax), int(xmin) : int(xmax)]

                        if crop.size == 0 or xmax <= xmin or ymax <= ymin:
                            continue

                        predictions = request.app.state.classifier.predict(crop, top=1)

                        # Store the results.
                        tracking_data.append(
                            {
                                "frame_id": frame_id,
                                "tracker_id": tracker_id,
                                "xmin": xmin,
                                "ymin": ymin,
                                "xmax": xmax,
                                "ymax": ymax,
                                "confidence": predictions[0].confidence,
                                "label": predictions[0].class_name,
                            }
                        )

                frame_id += 1
                pbar.update(1)
    finally:
        container.close()
        await file.close()

    if not tracking_data:
        return VideoDetectionResponse(filename=file.filename, predictions=[])

    # Process data.
    df = pd.DataFrame(tracking_data)
    df_smooth = smooth_labels(df, threshold=S.CLASSIFIER_THRESHOLD)

    # Build the final video predictions.
    response = VideoDetectionResponse(
        filename=file.filename,
        predictions=[
            Detection(
                class_name=row["label"],
                box=BoundingBox(**row[["xmin", "ymin", "xmax", "ymax"]]),
                confidence=float(row["confidence"]),
                tracking_id=int(row["tracker_id"]),
                frame_id=int(row["frame_id"]),
            )
            for _, row in df_smooth.iterrows()
        ],
    )

    # Store record on database.
    for pred in response.predictions:
        detection_record = VideoOutputModel(
            video_input_id=saved_video.id,
            frame_id=pred.frame_id,
            tracker_id=pred.tracking_id,
            label=pred.class_name,
            confidence=pred.confidence,
            x_min=pred.box.xmin,
            y_min=pred.box.ymin,
            x_max=pred.box.xmax,
            y_max=pred.box.ymax,
        )
        request.app.state.db.add_record(detection_record)

    # Render annotated video entirely in memory (PyAV -> BytesIO).
    in_container = av.open(io.BytesIO(video_bytes))
    in_stream = in_container.streams.video[0]

    out_buffer = io.BytesIO()
    out_container = av.open(out_buffer, mode="w", format="mp4")

    # Use H.264 codec and faststart flags for seamless web playback.
    out_stream = out_container.add_stream("libx264", rate=in_stream.average_rate or 30)
    out_stream.width = in_stream.codec_context.width
    out_stream.height = in_stream.codec_context.height
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"preset": "ultrafast", "movflags": "+faststart"}

    box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.6,
        text_thickness=1,
        color_lookup=sv.ColorLookup.TRACK,
    )

    for current_frame_id, frame in enumerate(in_container.decode(video=0)):
        frame_bgr = frame.to_ndarray(format="bgr24")
        df_frame = df_smooth[df_smooth["frame_id"] == current_frame_id]

        if not df_frame.empty:
            labels = ["{lbl} #{tid}" for lbl, tid in zip(df_frame["label"], df_frame["tracker_id"])]
            detections_sv = sv.Detections(
                xyxy=df_frame[["xmin", "ymin", "xmax", "ymax"]].to_numpy(),
                tracker_id=df_frame["tracker_id"].to_numpy(),
            )
            frame_bgr = box_annotator.annotate(scene=frame_bgr, detections=detections_sv)
            frame_bgr = label_annotator.annotate(scene=frame_bgr, detections=detections_sv, labels=labels)

        out_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)

    # Flush remaining frames.
    for packet in out_stream.encode():
        out_container.mux(packet)

    in_container.close()
    out_container.close()

    # Encode binary MP4 to Base64 data URL.
    video_b64 = "data:video/mp4;base64," + base64.b64encode(out_buffer.getvalue()).decode("utf-8")

    # Generate distinct track summary list.
    df_summary = df_smooth[["tracker_id", "label"]].drop_duplicates(subset=["tracker_id"]).sort_values(by="tracker_id")
    summary = df_summary.to_dict(orient="records")

    return templates.TemplateResponse(
        name="partials/video_result.html",
        request=request,
        context={
            "filename": file.filename,
            "has_detections": True,
            "video_b64": video_b64,
            "summary": summary,
            "total_frames": frame_id,
        },
    )
