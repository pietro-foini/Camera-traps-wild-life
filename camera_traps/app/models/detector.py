import logging
import os
from typing import Any

import numpy as np
from ultralytics import YOLO

from camera_traps.app.models.domain import Predictor
from camera_traps.app.schemas.base import BoundingBox, Detection


class ImageDetector(Predictor):
    """Image detector."""

    def __init__(self) -> None:
        super().__init__()

    def load(self, artifact_uri: str) -> None:
        """Load model."""

        if not os.path.exists(artifact_uri):
            raise FileNotFoundError(f"Model not found: {artifact_uri}")

        self._model = YOLO(model=artifact_uri)

        logging.info("Loaded model: %s", artifact_uri)

    def preprocess(self, prediction_input: np.ndarray) -> np.ndarray:
        """Preprocess image."""

        return prediction_input

    def predict(self, instance: np.ndarray, **kwargs: Any) -> list[Detection]:
        """Run image detection."""

        if self._model is None:
            raise RuntimeError("Model has not been loaded.")

        # Predict.
        prediction_input = self.preprocess(instance)

        results = self._model(prediction_input, verbose=False)
        result = results[0]

        return [
            Detection(
                class_id=int(box.cls[0]),
                class_name=self._model.names[int(box.cls[0])],
                box=BoundingBox(
                    xmin=float(box.xyxy[0][0]),
                    ymin=float(box.xyxy[0][1]),
                    xmax=float(box.xyxy[0][2]),
                    ymax=float(box.xyxy[0][3]),
                ),
                confidence=float(box.conf[0]),
            )
            for box in result.boxes
        ]
