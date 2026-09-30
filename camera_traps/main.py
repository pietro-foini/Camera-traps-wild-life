import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from camera_traps.api.routes import home_router, image_router
from camera_traps.models.classification.tflite import TFLiteImageClassifier
from camera_traps.settings import S

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up server")

    # Classifier Initialization.
    classifier = TFLiteImageClassifier(img_size=S.CLASSIFIER_IMAGE_SIZE)
    classifier.load(str(BASE_DIR / S.CLASSIFIER_MODEL_PATH))

    # Save initialized instances to FastAPI state.
    app.state.classifier = classifier

    yield

    logging.info("Shutting down server and releasing resources...")


app = FastAPI(title="Camera Traps Wildlife", lifespan=lifespan)

# Include API routes.
app.include_router(home_router, tags=["Home"])
app.include_router(image_router, tags=["Image"])
