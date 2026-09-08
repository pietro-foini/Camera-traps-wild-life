import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from camera_traps.app.api.dependencies import get_classifier
from camera_traps.app.models.domain import Predictor
from camera_traps.app.schemas.base import ImageClassificationResponse

image_router = APIRouter(prefix="/image", tags=["Image"])


@image_router.post("/predict-image", response_model=ImageClassificationResponse)
async def predict_image(
    file: UploadFile = File(...),
    classifier: Predictor = Depends(get_classifier),
):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # Open image.
    try:
        img_bytes = await file.read()
    finally:
        await file.close()

    img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img_array is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    predictions = classifier.predict(img_array, top=5)

    return ImageClassificationResponse(filename=file.filename, predictions=predictions)
