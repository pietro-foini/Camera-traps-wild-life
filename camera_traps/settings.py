from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global Settings"""

    # Models for detection and classification.
    CLASSIFIER_MODEL_PATH: str
    """path to the model used for image classification"""

    CLASSIFIER_IMAGE_SIZE: tuple[int, int] = (224, 224)
    """image size used for image classification"""

    # Thresholds filtering predictions.
    CLASSIFIER_THRESHOLD: float = 0.95
    """threshold for keeping classified images"""


S = Settings()  # type: ignore
