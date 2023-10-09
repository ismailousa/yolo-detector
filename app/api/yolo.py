import json
from fastapi import APIRouter, UploadFile, HTTPException
import numpy as np
from PIL import Image
from io import BytesIO
from app.schemas.models import ObjectDetectionResponse
from app.models.yolo import Yolov5

router = APIRouter()
yolov5_model = Yolov5()


@router.post(
    "/detect_objects/", response_model=ObjectDetectionResponse, status_code=200
)
async def detect_objects(file: UploadFile):
    try:
        # Read the image file from the request as bytes
        image_data = await file.read()

        # Convert image bytes to a NumPy array
        image = np.array(Image.open(BytesIO(image_data)))

        # Perform object detection using the YOLOv5 model
        results = yolov5_model.predict(image)

        # Retrieve the data as a NumPy array
        result_data = results.xyxy[0].cpu().numpy()

        # Extract relevant information from the YOLOv5 results
        bounding_boxes, confidence_scores, class_ids = (
            result_data[:, 0:4].tolist(),
            result_data[:, 4].tolist(),
            result_data[:, 5].astype(int).tolist(),
        )

        # Get class labels
        classes = [yolov5_model.class_names[id] for id in class_ids]

        # Prepare a structured response
        response_data = {
            "bounding_boxes": bounding_boxes,
            "confidence_scores": confidence_scores,
            "classes": classes,
        }

        # Serialize the response data to JSON
        print(response_data)
        return json.loads(json.dumps(response_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
