# Object Detection with YOLOv5 and FastAPI

This project demonstrates an object detection microservice using the YOLOv5 model and FastAPI. Given a byte array as input, the service outputs a response in the following JSON format:

```json
response_data = {
    "bounding_boxes": [...],
    "confidence_scores": [...],
    "classes": [...]
}
```

## Installation

To install and run this project, follow these steps:

1. Clone this repository to your local machine:
```
git clone https://github.com/ismailousa/yolo-detector.git
```

2. Navigate to the project directory:
```
cd yolo-detector
```

3. Use Poetry to install the project dependencies:
```
poetry install 
```

## Usage

To run the Yolo detector service, execute the following command:
```
poetry run ./run.sh
```
This will start the server, and you can use the following endpoint for object detection:

**POST /detect_objects/**: Upload an image as a byte array to this endpoint to receive object detection results in the specified response format.

A quick demo can also be started by:
```
poetry run python demo.py
```

## License

This project is licensed under the MIT License

