import os
import cv2
import time
from datetime import datetime
from .base_camera import BaseCamera
from .ssd_detection import Detector
# from yolo_detection import Detector
detector = Detector()

IMAGE_FOLDER = "./imgs"

class CameraPred(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            # Prediction
            output = detector.prediction(img)
            df = detector.filter_prediction(output, img)
            img = detector.draw_boxes(img, df)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()


class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()


def CaptureContinous():
    while(1):
        cap = cv2.VideoCapture(0)
        # Capture frame-by-frame
        ret, image = cap.read()
        output = detector.prediction(image)
        df = detector.filter_prediction(output, image)
        if len(df) > 0:
            day = datetime.now().strftime("%Y%m%d")
            directory = os.path.join(IMAGE_FOLDER, 'webcam', day)
            if not os.path.exists(directory):
                os.makedirs(directory)
            image = detector.draw_boxes(image, df)
            classes = df['class_name'].unique().tolist()
            hour = datetime.now().strftime("%H%M%S")
            filename_output = os.path.join(directory, "{}_{}_.jpg".format(hour, "-".join(classes)))
            cv2.imwrite(filename_output, image)
        cap.release()
        time.sleep(2)
