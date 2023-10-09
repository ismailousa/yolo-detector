from pydantic import BaseModel


class ObjectDetectionResponse(BaseModel):
    bounding_boxes: list
    confidence_scores: list
    classes: list
