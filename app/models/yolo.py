import torch


class Yolov5:
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    @property
    def class_names(self):
        return self.model.names

    def predict(self, image):
        return self.model(image)
