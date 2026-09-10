import io
from unittest.mock import MagicMock
import av
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.config import *
from camera_traps.backend.schemas.base import Classification, Detection, BoundingBox


def create_dummy_video_bytes():
    """Genera 2 frames MP4 into memory."""

    output = io.BytesIO()

    container = av.open(output, mode="w", format="mp4")
    stream = container.add_stream("mpeg4", rate=30)
    stream.width = 100
    stream.height = 100
    stream.pix_fmt = "yuv420p"

    for i in range(4):
        array = np.zeros((10, 10, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()

    return output.getvalue()


def test_predict_video_success():
    """Successful video prediction."""

    from camera_traps.backend.api.dependencies import get_classifier, get_detector, get_db
    from camera_traps.backend.api.routes import video_router

    # Setup Detector Mock.
    mock_detector = MagicMock()
    mock_detector.predict.side_effect = [
        [
            Detection(
                class_id=0,
                class_name="animal",
                confidence=0.90,
                box=BoundingBox(xmin=10.0, ymin=10.0, xmax=50.0, ymax=50.0),
            )
        ],  # Frame 0
        [
            Detection(
                class_id=1,
                class_name="animal",
                confidence=0.95,
                box=BoundingBox(xmin=11.0, ymin=11.0, xmax=50.0, ymax=50.0),
            )
        ],  # Frame 1
        [
            Detection(
                class_id=0,
                class_name="animal",
                confidence=0.90,
                box=BoundingBox(xmin=12.0, ymin=12.0, xmax=50.0, ymax=50.0),
            )
        ],  # Frame 0
        [
            Detection(
                class_id=1,
                class_name="animal",
                confidence=0.95,
                box=BoundingBox(xmin=13.0, ymin=13.0, xmax=50.0, ymax=50.0),
            )
        ],  # Frame 1
    ]

    # Setup Classifier Mock.
    mock_classifier = MagicMock()
    mock_classifier.predict.side_effect = [
        [Classification(class_id=1, class_name="fox", confidence=0.95)],  # Crop 0
        [Classification(class_id=1, class_name="fox", confidence=0.98)],  # Crop 1
    ]

    # Application.
    app = FastAPI()
    app.include_router(video_router)
    client = TestClient(app)

    app.dependency_overrides[get_detector] = lambda: mock_detector
    app.dependency_overrides[get_classifier] = lambda: mock_classifier
    app.dependency_overrides[get_db] = lambda: MagicMock()

    # Create dummy video.
    video_bytes = create_dummy_video_bytes()

    # Send request.
    response = client.post(
        "/video/predict-video",
        files={"file": ("test.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["filename"] == "test.mp4"
    predictions = json_data["predictions"]

    print(predictions)

    assert len(predictions) > 0
    assert predictions[0] == {
        "class_id": None,
        "class_name": "fox",
        "confidence": 0.95,
        "box": {"xmin": 12.0, "ymin": 12.0, "xmax": 50.0, "ymax": 50.0},
        "tracking_id": 0,
        "frame_id": 2,
    }
    assert predictions[1] == {
        "class_id": None,
        "class_name": "fox",
        "confidence": 0.98,
        "box": {"xmin": 13.0, "ymin": 13.0, "xmax": 50.0, "ymax": 50.0},
        "tracking_id": 0,
        "frame_id": 3,
    }
    assert mock_detector.predict.call_count == 4
    assert mock_classifier.predict.call_count == 2

    app.dependency_overrides.clear()


def test_predict_video_invalid_content_type():
    """Invalid content type."""

    from camera_traps.backend.api.dependencies import get_classifier, get_detector, get_db
    from camera_traps.backend.api.routes import video_router

    app = FastAPI()
    app.include_router(video_router)
    client = TestClient(app)

    app.dependency_overrides[get_detector] = lambda: MagicMock()
    app.dependency_overrides[get_classifier] = lambda: MagicMock()
    app.dependency_overrides[get_db] = lambda: MagicMock()

    response = client.post(
        "/video/predict-video",
        files={"file": ("test.txt", io.BytesIO(b"not a video"), "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid video."


def test_predict_video_no_detections():
    """No detections in video."""

    from camera_traps.backend.api.dependencies import get_classifier, get_detector, get_db
    from camera_traps.backend.api.routes import video_router

    mock_detector = MagicMock()
    mock_detector.predict.return_value = []

    mock_classifier = MagicMock()

    app = FastAPI()
    app.include_router(video_router)
    client = TestClient(app)

    app.dependency_overrides[get_detector] = lambda: mock_detector
    app.dependency_overrides[get_classifier] = lambda: mock_classifier
    app.dependency_overrides[get_db] = lambda: MagicMock()

    video_bytes = create_dummy_video_bytes()

    response = client.post(
        "/video/predict-video",
        files={"file": ("empty.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["filename"] == "empty.mp4"
    assert json_data["predictions"] == []
    mock_classifier.predict.assert_not_called()

    app.dependency_overrides.clear()
