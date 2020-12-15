import cv2
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from backend.utils import timeit

DELTA_THRESH = 10
MIN_AREA = 4000


class Detector():
    """Class motion detector"""

    @timeit
    def __init__(self):
        self.avg = None
        self.colors = np.random.uniform(0, 255, size=(100, 3))

    @timeit
    def prediction(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (21, 21), 0)
        if self.avg is None:
            self.avg = image.copy().astype(float)
        cv2.accumulateWeighted(image, self.avg, 0.5)
        frameDelta = cv2.absdiff(image, cv2.convertScaleAbs(self.avg))
        thresh = cv2.threshold(
                frameDelta, DELTA_THRESH, 255,
                cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        # select contours depending on opencv version
        if len(cnts) == 2:
            cnts = cnts[0]
        elif len(cnts) == 3:
            cnts = cnts[1]
        self.avg = image.copy().astype(float)
        return cnts

    @timeit
    def filter_prediction(self, output, image):
        if len(output) < 2:
            return pd.DataFrame()
        else:
            df = pd.DataFrame(output)
            df = df.assign(
                    area=lambda x: df[0].apply(lambda x: cv2.contourArea(x)),
                    bounding=lambda x: df[0].apply(lambda x: cv2.boundingRect(x))
                    )
            df = df[df['area'] > MIN_AREA]
            df_filtered = pd.DataFrame(
                    df['bounding'].values.tolist(), columns=['x1', 'y1', 'w', 'h'])
            df_filtered = df_filtered.assign(
                    x1=lambda x: x['x1'].clip(0),
                    y1=lambda x: x['y1'].clip(0),
                    x2=lambda x: (x['x1'] + x['w']),
                    y2=lambda x: (x['y1'] + x['h']),
                    label=lambda x: x.index.astype(str),
                    class_name=lambda x: x.index.astype(str),
                    )
            return df_filtered

    def draw_boxes(self, image, df):
        for idx, box in df.iterrows():
            color = self.colors[int(box['label'])]
            cv2.rectangle(image, (box['x1'], box['y1']), (box['x2'], box['y2']), color, 6)
            cv2.putText(image, box['label'], (box['x1'], box['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image


if __name__ == "__main__":
    image = cv2.imread("./imgs/image.jpeg")
    print(image.shape)

    detector = Detector()

    image2 = cv2.imread("./imgs/image_box.jpg")
    print(image2.shape)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.GaussianBlur(image2, (21, 21), 0)
    detector.avg = image2.astype(float)

    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    print(df)
    image = detector.draw_boxes(image, df)

    cv2.imwrite("./imgs/outputcv.jpg", image)
