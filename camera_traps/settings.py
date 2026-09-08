from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global Settings"""

    API_URL: str = "http://127.0.0.1:8000"
    """base URL for the backend API"""

    # Models.
    CLASSIFIER_MODEL_PATH: str = "/home/pietr/projects/camera-traps-wild-life/notebooks/classification/model.keras"
    """path to the model used for image classification"""
    CLASSIFIER_IMAGE_SIZE: tuple[int, int] = (224, 224)
    """image size used for image classification"""
    CLASSIFIER_CLASSES_PATH: str = (
        "/home/pietr/projects/camera-traps-wild-life/notebooks/classification/class_names.json"
    )
    """path to json file containing class names"""
    DETECTOR_MODEL_PATH: str = "/home/pietr/projects/camera-traps-wild-life/notebooks/detection/md_v1000.0.0-sorrel.pt"
    """path to the model used for object detection"""
    DETECTOR_IMAGE_SIZE: tuple[int, int] = (640, 640)
    """image size used for object detection"""

    # Thresholds.
    CLASSIFIER_THRESHOLD: float = 0.95
    """threshold for keeping classified images"""
    DETECTOR_THRESHOLD: float = 0.9
    """threshold for keeping detected objects"""


S = Settings()  # type: ignore[call-arg]
