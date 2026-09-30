from pydantic import BaseModel, Field


class Classification(BaseModel):
    """Image classification result."""

    class_id: int
    class_name: str
    confidence: float | int = Field(ge=0.0, le=1.0)


class ImageClassificationResponse(BaseModel):
    """Response payload containing image classification predictions."""

    filename: str
    predictions: list[Classification] = Field(default_factory=list)
