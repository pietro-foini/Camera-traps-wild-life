from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global Settings"""

    # Models.
    CLASSIFIER_MODEL_PATH: str
    """path to the model used for image classification"""
    CLASSIFIER_CLASSES_PATH: str
    """path to json file containing class names"""
    DETECTOR_MODEL_PATH: str
    """path to the model used for object detection"""
    CLASSIFIER_IMAGE_SIZE: tuple[int, int] = (224, 224)
    """image size used for image classification"""
    DETECTOR_IMAGE_SIZE: tuple[int, int] = (640, 640)
    """image size used for object detection"""

    # Thresholds.
    CLASSIFIER_THRESHOLD: float = 0.95
    """threshold for keeping classified images"""
    DETECTOR_THRESHOLD: float = 0.8
    """threshold for keeping detected objects"""


S = Settings()  # type: ignore[call-arg]
