import json

from camera_traps.app.models import ImageClassifier, ImageDetector
from camera_traps.settings import S

# Load class names of the classifier.
with open(S.CLASSIFIER_CLASSES_PATH, "r") as f:
    labels = json.load(f)


# Initialize classifier.
classifier = ImageClassifier(
    img_size=S.CLASSIFIER_IMAGE_SIZE,
    class_names=[v for k, v in labels.items()],
)
classifier.load(S.CLASSIFIER_MODEL_PATH)
# Initialize detector.
detector = ImageDetector()
detector.load(S.DETECTOR_MODEL_PATH)
