import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from trackers import SORTTracker

from camera_traps.api.routes import home_router, image_router, video_router
from camera_traps.db.postgres import PostgresHandler
from camera_traps.settings import S

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up server")

    # Database initialization.
    db = PostgresHandler(
        db_user=S.DB_USER,
        db_password=S.DB_PASSWORD,
        db_host=S.DB_HOST,
        db_port=S.DB_PORT,
        db_name=S.DB_NAME,
    )
    db.init_db()
    app.state.db = db

    # Tracker initialization.
    tracker = SORTTracker()

    # Dynamic Classifier Initialization.
    if Path(S.CLASSIFIER_MODEL_PATH).suffix == ".keras":
        from camera_traps.models.tf_classifier import TFImageClassifier

        classifier = TFImageClassifier(img_size=S.CLASSIFIER_IMAGE_SIZE)
        classifier.load(S.CLASSIFIER_MODEL_PATH)

    elif Path(S.CLASSIFIER_MODEL_PATH).suffix == ".tflite":
        from camera_traps.models.tflite_classifier import TFLiteImageClassifier

        classifier = TFLiteImageClassifier(img_size=S.CLASSIFIER_IMAGE_SIZE)
        classifier.load(S.CLASSIFIER_MODEL_PATH)

    else:
        raise ValueError(f"Unsupported classifier")

    # Dynamic Detector Initialization.
    if Path(S.DETECTOR_MODEL_PATH).suffix == ".pt":
        from camera_traps.models.pt_detector import PTImageDetector

        detector = PTImageDetector()
        detector.load(S.DETECTOR_MODEL_PATH)

    elif Path(S.DETECTOR_MODEL_PATH).suffix == ".tflite":
        from camera_traps.models.tflite_detector import TFLiteImageDetector

        detector = TFLiteImageDetector(img_size=S.DETECTOR_IMAGE_SIZE)
        detector.load(S.DETECTOR_MODEL_PATH)

    else:
        raise ValueError(f"Unsupported detector")

    # Save initialized instances to FastAPI state.
    app.state.classifier = classifier
    app.state.detector = detector
    app.state.tracker = tracker

    yield

    logging.info("Shutting down server and releasing resources...")


app = FastAPI(title="Camera Traps Wildlife", lifespan=lifespan)

# Include API routes.
app.include_router(home_router, tags=["Home"])
app.include_router(image_router, tags=["Image"])
app.include_router(video_router, tags=["Video"])
