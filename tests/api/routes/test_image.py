import io
from unittest.mock import MagicMock
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import ANY

from tests.config import *
from camera_traps.schemas.base import Classification


def create_dummy_image_bytes():
    """Generate image bytes."""

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, encoded_img = cv2.imencode(".jpg", img)

    return encoded_img


def test_predict_image_success():
    """Successful image prediction."""

    from camera_traps.api.routes import image_router

    # Setup Classifier Mock.
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = [
        Classification(class_id=1, class_name="fox", confidence=0.85),
        Classification(class_id=2, class_name="deer", confidence=0.10),
        Classification(class_id=3, class_name="bear", confidence=0.03),
        Classification(class_id=4, class_name="rabbit", confidence=0.01),
        Classification(class_id=5, class_name="wolf", confidence=0.01),
    ]

    # Setup DB Mock.
    mock_db = MagicMock()
    mock_saved_image = MagicMock()
    mock_saved_image.id = 123
    mock_db.add_record.return_value = mock_saved_image

    # Application.
    app = FastAPI()
    app.include_router(image_router)
    client = TestClient(app)
    app.state.classifier = mock_predictor
    app.state.db = mock_db

    # Create dummy image.
    image_bytes = create_dummy_image_bytes()

    # Send request.
    response = client.post(
        "/predict-image",
        files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )

    assert response.status_code == 200

    html_content = response.text
    assert "fox" in html_content
    mock_predictor.predict.assert_called_once_with(ANY, top=5)
    assert mock_db.add_record.call_count == 6

    app.dependency_overrides.clear()


def test_predict_image_invalid_content_type():
    """Invalid content type."""

    from camera_traps.api.routes import image_router

    app = FastAPI()
    app.include_router(image_router)
    client = TestClient(app)

    app.state.classifier = MagicMock()
    app.state.db = MagicMock()

    response = client.post(
        "/predict-image",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )

    assert response.status_code == 400
    assert response.text == "Uploaded file is not a valid image."

    app.dependency_overrides.clear()


def test_predict_image_corrupt_bytes():
    """Corrupt bytes."""

    from camera_traps.api.routes import image_router

    app = FastAPI()
    app.include_router(image_router)
    client = TestClient(app)

    app.state.classifier = MagicMock()
    app.state.db = MagicMock()

    response = client.post(
        "/predict-image",
        files={"file": ("corrupt.jpg", io.BytesIO(b"corrupted data"), "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.text == "Uploaded file is not a valid image."

    app.dependency_overrides.clear()
