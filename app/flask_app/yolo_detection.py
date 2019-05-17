import os
import cv2
import json
import numpy as np
import pandas as pd
from utils import timeit

DETECTION_MODEL = 'yolo'
THRESHOLD = 0.5
SCALE = 0.00392  # 1/255
NMS_THRESHOLD = 0.4  # Non Maximum Supression threshold
SWAPRB = True

with open(os.path.join('/models', DETECTION_MODEL, 'labels.json')) as json_data:
    CLASS_NAMES = json.load(json_data)


def filter_yolo(chunk):
    pred = np.argmax(chunk[:, 5:], axis=1)
    prob = np.max(chunk[:, 5:], axis=1)
    df = pd.DataFrame(
            np.concatenate(
                [chunk[:, :4], pred.reshape(-1, 1), prob.reshape(-1, 1)],
                axis=1
                ),
            columns=[
                'center_x', 'center_y', 'w', 'h', 'class_id', 'confidence'])
    df = df[df['confidence'] > THRESHOLD]
    return df


class Detector():
    """Class yolo"""

    @timeit
    def __init__(self):
        self.model = cv2.dnn.readNetFromDarknet(
                'models/yolo/yolov3.cfg',
                'models/yolo/yolov3.weights')
                #'models/yolo/yolov3-tiny.cfg',
                #'models/yolo/yolov3-tiny.weights')
        self.colors = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    @timeit
    def prediction(self, image):
        blob = cv2.dnn.blobFromImage(image, SCALE, (416, 416), (0, 0, 0),
                                     swapRB=SWAPRB, crop=False)
        self.model.setInput(blob)
        output = self.model.forward(self.get_output_layers(self.model))
        return output

    @timeit
    def filter_prediction(self, output, image):
        image_height, image_width, _ = image.shape
        df = pd.concat([filter_yolo(i) for i in output])
        df = df.assign(
                center_x=lambda x: (x['center_x'] * image_width),
                center_y=lambda x: (x['center_y'] * image_height),
                w=lambda x: (x['w'] * image_width),
                h=lambda x: (x['h'] * image_height),
                x1=lambda x: (x.center_x - (x.w / 2)).astype(int).clip(0),
                y1=lambda x: (x.center_y - (x.h / 2)).astype(int).clip(0),
                x2=lambda x: (x.x1 + x.w).astype(int),
                y2=lambda x: (x.y1 + x.h).astype(int),
                class_name=lambda x: (
                    x['class_id'].astype(int).astype(str).replace(CLASS_NAMES)),
                label=lambda x: (
                    x.class_name + ': ' + (
                        x['confidence'].astype(str).str.slice(stop=4)
                        )
                    )
                )
        cols = ['x1', 'y1', 'w', 'h']
        indices = cv2.dnn.NMSBoxes(
                df[cols].values.tolist(),
                df['confidence'].tolist(), THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            df = df.iloc[indices.flatten()]
        return df

    def draw_boxes(self, image, df):
        for idx, box in df.iterrows():
            print('--> Detected: ({}:{}) - Score: {:.3f}'.format(box['class_id'], box['class_name'], box['confidence']))
            color = self.colors[int(box['class_id'])]
            cv2.rectangle(image, (box['x1'], box['y1']), (box['x2'], box['y2']), color, 6)
            cv2.putText(image, box['label'], (box['x1'], box['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image


if __name__ == "__main__":
    image = cv2.imread("./imgs/image.jpeg")

    detector = Detector()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    image = detector.draw_boxes(image, df)

    cv2.imwrite("./imgs/outputcv.jpg", image)
