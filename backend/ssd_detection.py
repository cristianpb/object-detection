import os
import json
import cv2
import numpy as np
import pandas as pd
from backend.utils import timeit

THRESHOLD = 0.5
DETECTION_MODEL = 'ssd_mobilenet/'
SWAPRB = True

with open(os.path.join('models', DETECTION_MODEL, 'labels.json')) as json_data:
    CLASS_NAMES = json.load(json_data)


class Detector():
    """Class ssd"""

    @timeit
    def __init__(self):
        self.model = cv2.dnn.readNetFromTensorflow(
                'models/ssd_mobilenet/frozen_inference_graph.pb',
                'models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
        self.colors = np.random.uniform(0, 255, size=(100, 3))

    @timeit
    def prediction(self, image):
        self.model.setInput(
                cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=SWAPRB))
        output = self.model.forward()
        result = output[0, 0, :, :]
        return result

    @timeit
    def filter_prediction(self, output, image):
        height, width = image.shape[:-1]
        df = pd.DataFrame(
                output,
                columns=[
                    '_', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        df = df.assign(
                x1=lambda x: (x['x1'] * width).astype(int).clip(0),
                y1=lambda x: (x['y1'] * height).astype(int).clip(0),
                x2=lambda x: (x['x2'] * width).astype(int),
                y2=lambda x: (x['y2'] * height).astype(int),
                class_name=lambda x: (
                    x['class_id'].astype(int).astype(str).replace(CLASS_NAMES)
                    ),
                # TODO: don't work in python 3.5
                # label=lambda x: (
                #     x.class_name + ': ' + (
                #         x['confidence'].astype(str).str.slice(stop=4)
                #         )
                #     )
                )
        df['label'] = (df['class_name'] + ': ' +
                       df['confidence'].astype(str).str.slice(stop=4))
        df = df[df['confidence'] > THRESHOLD]
        return df

    def draw_boxes(self, image, df):
        for idx, box in df.iterrows():
            print('--> Detected: ({}:{}) - Score: {:.3f}'
                  .format(box['class_id'],
                          box['class_name'],
                          box['confidence'])
                  )
            color = self.colors[int(box['class_id'])]
            cv2.rectangle(
                    image,
                    (box['x1'], box['y1']),
                    (box['x2'], box['y2']),
                    color, 6)
            cv2.putText(
                    image,
                    box['label'],
                    (box['x1'], box['y1'] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image


if __name__ == "__main__":
    image = cv2.imread("./imgs/image.jpeg")
    print(CLASS_NAMES)

    detector = Detector()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    print(df)
    image = detector.draw_boxes(image, df)
    cv2.imwrite("./imgs/outputcv.jpg", image)
