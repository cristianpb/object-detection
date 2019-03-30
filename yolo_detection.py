import cv2
import numpy as np
import time
from models.yolo.yolo_labels import LABEL_MAP

THRESHOLD = 0.5
SCALE = 0.00392  # 1/255
NMS_THRESHOLD = 0.4  # Non Maximum Supression threshold


class YOLO():
    """Class yolo"""

    def __init__(self):
        start = time.time()
        self.model = cv2.dnn.readNetFromDarknet(
                #'models/yolo/yolov3.cfg',
                #'models/yolo/yolov3.weights')
                'models/yolo/yolov3-tiny.cfg',
                'models/yolo/yolov3-tiny.weights')
        self.colors = np.random.uniform(0, 255, size=(100, 3))
        print("Loading model in: {:.3f} s".format(time.time() - start))

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def prediction(self, image):
        start = time.time()
        image_height, image_width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, SCALE, (416, 416), (0, 0, 0),
                                     swapRB=True, crop=False)
        self.model.setInput(blob)
        output = self.model.forward(self.get_output_layers(self.model))
        print("Model inference {:.2f} s".format(time.time() - start))
        return output, image_width, image_height

    def filter_prediction(self, output, image_width, image_height):
        start = time.time()
        class_ids = []
        confidences = []
        boxes = []

        print(image_width, image_height)
        for out in output:
            print(out.shape)
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > THRESHOLD:
                    class_name = self.id_class_name(class_id, LABEL_MAP)
                    print("--> Detected: ({}:{}) - Score: {:.2f}".format(class_id, class_name, confidence))
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)
                    w = int(detection[2] * image_width)
                    h = int(detection[3] * image_height)
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                   THRESHOLD, NMS_THRESHOLD)
        print("Filter predictions {:.4f} s".format(time.time() - start))
        return indices, boxes, confidences, class_ids

    def draw_boxes(self, image, indices, boxes, confidences, class_ids):
        start = time.time()
        if len(indices) > 0:
            # loop over the indexes we are keeping
            for i in indices.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = self.colors[class_ids[i]]
                class_name = self.id_class_name(class_ids[i], LABEL_MAP)

                # draw a bounding box rectangle and label on the image
                color = self.colors[int(class_ids[i])]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 6)
                text = "{}: {:.3f}".format(class_name, confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
        print("Print bouding boxes {:.4f} s".format(time.time() - start))
        return image

    def id_class_name(self, class_id, classes):
        for key, value in classes.items():
            if class_id == key:
                return value


if __name__ == "__main__":
    start = time.time()
    image = cv2.imread("./static/imgs/image.jpeg")
    print("Reaing image {}".format(time.time() - start))

    yolo = YOLO()
    output, image_width, image_height = yolo.prediction(image)
    indices, boxes, confidences, class_ids = yolo.filter_prediction(
            output, image_width, image_height)
    image = yolo.draw_boxes(image, indices, boxes, confidences, class_ids)

    start = time.time()
    cv2.imwrite("./static/imgs/outputcv.jpg", image)
    print("Writing img {:.2f}".format(time.time() - start))
