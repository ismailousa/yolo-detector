import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from app.models.yolo import Yolov5

yolov5_model = Yolov5()

# Define the URL of the FastAPI endpoint
endpoint_url = "http://localhost:8002/yolo/detect_objects/"

# Open a connection to the webcam (usually, 0 represents the default camera)
cap = cv2.VideoCapture(0)
while count < 2:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Ensure the frame was read successfully
    if not ret:
        break

    # Convert the NumPy array to bytes
    image_bytes = Image.fromarray(frame).tobytes()

    # Encode the frame as JPEG
    _, img_encoded = cv2.imencode(".jpg", frame)

    # Convert the encoded frame to bytes
    img_bytes = img_encoded.tobytes()

    image = Image.open(BytesIO(img_bytes))

    # Display the frame in an OpenCV window
    cv2.imshow("Camera Stream", frame)

    try:
        # Send a POST request to the FastAPI endpoint with the frame data
        response = requests.post(
            endpoint_url, files={"file": ("frame.jpg", img_bytes, "image/jpeg")}
        )

        # Process the response if needed
        data = response.json()
        print(data)
    except Exception as e:
        print(f"Error sending request: {e}")

    resp = yolov5_model.predict(np.array(image))
    print(resp)

    # Display the frame in an OpenCV window
    cv2.imshow("Camera Stream", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    count = count + 1

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
