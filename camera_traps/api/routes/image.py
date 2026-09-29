import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from camera_traps.db.models import ImageInputModel, ImageOutputModel
from camera_traps.models.models import ImageClassificationResponse

image_router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent.parent / "templates"))


@image_router.post(path="/predict-image", response_class=HTMLResponse)
def predict_image(request: Request, file: UploadFile = File(...)):

    if not file.filename or not file.content_type or not file.content_type.startswith("image/"):
        return HTMLResponse(content="Uploaded file is not a valid image.", status_code=400)

    # Store input record on database.
    image_record = ImageInputModel(filename=file.filename)
    saved_image = request.app.state.db.add_record(image_record)

    # Read image.
    try:
        img_bytes = file.file.read()
    finally:
        file.file.close()

    img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img_array is None:
        return HTMLResponse(content="Uploaded file is not a valid image.", status_code=400)

    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Predict.
    predictions = request.app.state.classifier.predict(img_array, top=5)
    response = ImageClassificationResponse(filename=file.filename, predictions=predictions)

    # Store output record on database.
    for pred in response.predictions:
        pred_record = ImageOutputModel(
            image_input_id=saved_image.id,
            label=pred.class_name,
            confidence=float(pred.confidence),
        )
        request.app.state.db.add_record(pred_record)

    return templates.TemplateResponse(
        name="partials/image_result.html",
        request=request,
        context={
            "response": response,
            "image_b64": base64.b64encode(img_bytes).decode("utf-8"),  # type: ignore
            "mime_type": file.content_type,
        },
    )
