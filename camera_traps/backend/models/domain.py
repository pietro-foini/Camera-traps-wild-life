from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Predictor(ABC):
    """Abstract Base Class for model predictors."""

    def __init__(self) -> None:
        """Initialize predictor class."""
        self._model: Any = None

    @abstractmethod
    def load(self, artifact_uri: str) -> None:
        """Load model artifacts and auxiliary resources."""
        pass

    @abstractmethod
    def preprocess(self, prediction_input: np.ndarray) -> np.ndarray:
        """Preprocess raw input before model prediction."""
        pass

    @abstractmethod
    def predict(self, instance: np.ndarray) -> Any:
        """Perform model inference."""
        pass
