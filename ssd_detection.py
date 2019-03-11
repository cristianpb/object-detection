import cv2
import numpy as np
import time
from models.ssd_mobilenet.ssd_labels import LABEL_MAP

THRESHOLD = 0.5

class SSD():
    """Class ssd"""

    def __init__(self):
        start = time.time()
        self.model = cv2.dnn.readNetFromTensorflow(
                'models/ssd_mobilenet/frozen_inference_graph.pb',
                'models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
        self.colors = np.random.uniform(0, 255, size=(100, 3))
        print("Loading model in: {:.3f} s".format(time.time() - start))

    def prediction(self, image):
        start = time.time()
        self.model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
        output = self.model.forward()
        print("Model inference {:.2f} s".format(time.time() - start))
        return output

    def filter_prediction(self, output):
        start = time.time()
        final_detection = list()
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > THRESHOLD:
                final_detection.append(detection)
        print("Filter predictions {:.4f} s".format(time.time() - start))
        return final_detection

    def draw_boxes(self, image, output):
        image_height, image_width, _ = image.shape
        start = time.time()
        for detection in output:
            class_id = detection[1]
            confidence = detection[2]
            class_name = self.id_class_name(class_id)
            print("--> Detected: ({}:{}) - Score: {:.2f}".format(class_id, class_name,  confidence))
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            color = self.colors[int(class_id)]
            cv2.rectangle(
                    image, (int(box_x), int(box_y)),
                    (int(box_width), int(box_height)),
                    color, thickness=6)
            cv2.putText(
                    image, "{}:{:.2f}".format(class_name, confidence), (int(box_x + 0.001*image_width), int(box_y+.01*image_height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print("Print bouding boxes {:.4f} s".format(time.time() - start))
        return image

    def id_class_name(self, class_id):
        for key, value in LABEL_MAP.items():
            if class_id == key:
                return value

if __name__ == "__main__":
    pass
