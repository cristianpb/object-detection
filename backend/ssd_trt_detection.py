import os
import cv2
import json
import numpy as np
import pycuda.autoinit  # This is needed for initializing CUDA driver
from backend.utils import timeit

from utils_ssd.ssd import TrtSSD
from utils_ssd.visualization import BBoxVisualization

DETECTION_MODEL = 'yolo'
conf_th = 0.8
INPUT_HW = (300, 300)

with open(os.path.join('models', DETECTION_MODEL, 'labels.json')) as json_data:
    CLASS_NAMES = json.load(json_data)
vis = BBoxVisualization(CLASS_NAMES)


class Detector():
    """Class ssd"""

    @timeit
    def __init__(self):
        #filename = "imgs/image.jpeg"
        #img = cv2.imread(filename)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.model = TrtSSD("ssd_mobilenet_v2_coco", INPUT_HW)

    @timeit
    def prediction(self, image, conf_class=[]):
        boxes, confs, clss = self.model.detect(image, conf_th, conf_class=conf_class)
        #self.model.setInput(
        #        cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=SWAPRB))
        #output = self.model.forward()
        #result = output[0, 0, :, :]
        return boxes, confs, clss

    @timeit
    def filter_prediction(self, clss):
        print([CLASS_NAMES[c] for c in clss])
        #height, width = image.shape[:-1]
        #df = pd.DataFrame(
        #        output,
        #        columns=[
        #            '_', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        #df = df.assign(
        #        x1=lambda x: (x['x1'] * width).astype(int).clip(0),
        #        y1=lambda x: (x['y1'] * height).astype(int).clip(0),
        #        x2=lambda x: (x['x2'] * width).astype(int),
        #        y2=lambda x: (x['y2'] * height).astype(int),
        #        class_name=lambda x: (
        #            x['class_id'].astype(int).astype(str).replace(CLASS_NAMES)
        #            ),
        #        # TODO: don't work in python 3.5
        #        # label=lambda x: (
        #        #     x.class_name + ': ' + (
        #        #         x['confidence'].astype(str).str.slice(stop=4)
        #        #         )
        #        #     )
        #        )
        #df['label'] = (df['class_name'] + ': ' +
        #               df['confidence'].astype(str).str.slice(stop=4))
        #df = df[df['confidence'] > THRESHOLD]
        #return df

    def draw_boxes(self, image, boxes, confs, clss):
        image = vis.draw_bboxes(image, boxes, confs, clss)
        #for idx, box in df.iterrows():
        #    print('--> Detected: ({}:{}) - Score: {:.3f}'
        #          .format(box['class_id'],
        #                  box['class_name'],
        #                  box['confidence'])
        #          )
        #    color = self.colors[int(box['class_id'])]
        #    cv2.rectangle(
        #            image,
        #            (box['x1'], box['y1']),
        #            (box['x2'], box['y2']),
        #            color, 6)
        #    cv2.putText(
        #            image,
        #            box['label'],
        #            (box['x1'], box['y1'] - 5),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image#[..., ::-1]


if __name__ == "__main__":
    image = cv2.imread("./imgs/image.jpeg")

    detector = Detector()
    boxes, confs, clss = detector.prediction(image)
    detector.filter_prediction(clss)
    image = detector.draw_boxes(image, boxes, confs, clss)
    cv2.imwrite("./imgs/outputcv.jpg", image)
