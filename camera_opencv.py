import cv2
import time
from datetime import datetime
from base_camera import BaseCamera
from ssd_detection import SSD
ssd = SSD()

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
            output = ssd.prediction(img)
            output = ssd.filter_prediction(output)
            img = ssd.draw_boxes(img, output)

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
        output = ssd.prediction(image)
        df = ssd.filter_prediction(output, image)
        if len(df) > 0:
            image = ssd.draw_boxes(image, df)
            classes = df['class_name'].unique().tolist()
            today = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_output = "./imgs/webcam/{}_{}_.jpg".format(today, "-".join(classes))
            cv2.imwrite(filename_output, image)
        cap.release()
        time.sleep(2)
