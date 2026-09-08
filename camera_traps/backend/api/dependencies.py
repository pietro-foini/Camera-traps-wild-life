from fastapi import Request

from camera_traps.backend.models.domain import Predictor


def get_classifier(request: Request) -> Predictor:
    """Extracts the classifier instance from app state."""
    return request.app.state.classifier


def get_detector(request: Request) -> Predictor:
    """Extracts the detector instance from app state."""
    return request.app.state.detector
