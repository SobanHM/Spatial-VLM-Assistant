from ultralytics import YOLO
import cv2

class SpatialDetector:
    def __init__(self, model="yolov8n.pt", device_id=0):
        self.model = YOLO(model)
        self.device_id = device_id

    def detect(self, image):
        results = self.model(image, device=self.device_id)[0]

        detections = []
        for box, cls, conf in zip(
            results.boxes.xyxy,
            results.boxes.cls,
            results.boxes.conf
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = self.model.names[int(cls)]

            detections.append({
                "label": label,
                "confidence": float(conf),
                "box": [x1, y1, x2, y2]
            })

        return detections
