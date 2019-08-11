import cv2
import numpy as np
import pandas as pd
from utils import timeit

class Detector():
    """Class ssd"""

    @timeit
    def __init__(self):
        #self.model = cv2.CascadeClassifier("models/cascade/fullbody_recognition_model.xml") # an opencv classifier
        #self.model = cv2.CascadeClassifier("models/cascade/upperbody_recognition_model.xml") # an opencv classifier
        self.model = cv2.CascadeClassifier("models/cascade/facial_recognition_model.xml") # an opencv classifier
        self.colors = np.random.uniform(0, 255, size=(100, 3))

    @timeit
    def prediction(self, image):
        objects = self.model.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )
        return objects

    @timeit
    def filter_prediction(self, output, image):
        df = pd.DataFrame(
                output,
                columns=[
                    'x1', 'y1', 'w', 'h'])
        df = df.assign(
                x2=lambda x: (x['x1'] + x['w']),
                y2=lambda x: (x['y1'] + x['h']),
                label=lambda x: x.index.astype(str),
                class_name=lambda x: x.index.astype(str),
                )
        return df

    def draw_boxes(self, image, df):
        for idx, box in df.iterrows():
            color = self.colors[int(box['label'])]
            cv2.rectangle(image, (box['x1'], box['y1']), (box['x2'], box['y2']), color, 6)
            cv2.putText(image, box['label'], (box['x1'], box['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image


if __name__ == "__main__":
    image = cv2.imread("./imgs/image.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = Detector()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    image = detector.draw_boxes(image, df)

    cv2.imwrite("./imgs/outputcv.jpg", image)
