import io

import av
import numpy as np
import pandas as pd
import supervision as sv
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from tqdm import tqdm
from trackers import SORTTracker

from camera_traps.backend.api.dependencies import get_classifier, get_db, get_detector
from camera_traps.backend.db.domain import DBInterface
from camera_traps.backend.db.models import VideoInputModel, VideoOutputModel
from camera_traps.backend.models.domain import Predictor
from camera_traps.backend.schemas.base import BoundingBox, Detection, VideoDetectionResponse
from camera_traps.backend.services.tracking import smooth_labels
from camera_traps.settings import S

video_router = APIRouter(prefix="/video", tags=["Video"])


@video_router.post("/predict-video", response_model=VideoDetectionResponse)
async def predict_video(
    file: UploadFile = File(...),
    classifier: Predictor = Depends(get_classifier),
    detector: Predictor = Depends(get_detector),
    db: DBInterface = Depends(get_db),
):

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid video.")

    video_record = VideoInputModel(filename=file.filename)
    saved_video = db.add_record(video_record)

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
                predictions = detector.predict(frame)

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

                        predictions = classifier.predict(crop, top=1)

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

    # Store record on database.
    for _, row in df_smooth.iterrows():
        detection_record = VideoOutputModel(
            video_input_id=saved_video.id,
            frame_id=int(row["frame_id"]),
            tracker_id=int(row["tracker_id"]),
            label=row["label"],
            confidence=float(row["confidence"]),
            x_min=float(row["xmin"]),
            y_min=float(row["ymin"]),
            x_max=float(row["xmax"]),
            y_max=float(row["ymax"]),
        )
        db.add_record(detection_record)

    return VideoDetectionResponse(
        filename=file.filename,
        predictions=[
            Detection(
                class_name=row["label"],
                box=BoundingBox(**row[["xmin", "ymin", "xmax", "ymax"]]),
                confidence=row["confidence"],
                tracking_id=row["tracker_id"],
                frame_id=row["frame_id"],
            )
            for _, row in df_smooth.iterrows()
        ],
    )
