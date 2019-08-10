import os
import cv2
from celery import Celery
from dotenv import load_dotenv
from importlib import import_module
from datetime import datetime, timedelta
from backend.base_camera import BaseCamera

load_dotenv()
Detector = import_module('backend.' + os.environ['DETECTION_MODEL']).Detector
detector = Detector()

celery = Celery("app")
celery.conf.update(
        broker_url='redis://localhost:6379/0',
        result_backend='redis://localhost:6379/0',
        beat_schedule={
            "photos_SO": {
                "task": "backend.camera_opencv.CaptureContinous",
                "schedule": timedelta(seconds=int(str(os.environ['BEAT_INTERVAL']))),
                "args": []
                }
            }
)

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


@celery.task(bind=True)
def CaptureContinous(self):
    cap = cv2.VideoCapture(0)
    # Capture frame-by-frame
    ret, image = cap.read()
    cap.release()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    if len(df) > 0:
        if (df['class_name']
                .str
                .contains('person|bird|cat|wine glass|cup|sandwich')
                .any()):
            day = datetime.now().strftime("%Y%m%d")
            directory = os.path.join(IMAGE_FOLDER, 'webcam', day)
            if not os.path.exists(directory):
                os.makedirs(directory)
            image = detector.draw_boxes(image, df)
            classes = df['class_name'].unique().tolist()
            hour = datetime.now().strftime("%H%M%S")
            filename_output = os.path.join(
                    directory, "{}_{}_.jpg".format(hour, "-".join(classes))
                    )
            cv2.imwrite(filename_output, image)
