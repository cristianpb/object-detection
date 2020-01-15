import os
import cv2
import base64
from celery import Celery
from dotenv import load_dotenv
from importlib import import_module
from datetime import datetime, timedelta
from backend.base_camera import BaseCamera

load_dotenv('.env')
Detector = import_module('backend.' + os.environ['DETECTION_MODEL']).Detector
detector = Detector()

celery = Celery("app")
celery.conf.update(
        broker_url='redis://localhost:6379/0',
        result_backend='redis://localhost:6379/0',
        beat_schedule={
            "photos_SO": {
                "task": "backend.camera_opencv.CaptureContinous",
                "schedule": timedelta(
                    seconds=int(str(os.environ['BEAT_INTERVAL']))
                    ),
                "args": []
                }
            }
)

IMAGE_FOLDER = "./imgs"

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=120,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True "
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            yield img

    @staticmethod
    def prediction(img):
        boxes, confs, clss = detector.prediction(img)
        img = detector.draw_boxes(img, boxes, confs, clss)
        #output = detector.prediction(img)
        #df = detector.filter_prediction(output, img)
        #img = detector.draw_boxes(img, df)
        return img

    @staticmethod
    def img_to_base64(img):
        """encode as a jpeg image and return it"""
        buffer = cv2.imencode('.jpg', img)[1].tobytes()
        jpg_as_text = base64.b64encode(buffer)
        base64_string = jpg_as_text.decode('utf-8')
        return base64_string


@celery.task(bind=True)
def CaptureContinous(self):
    cap = cv2.VideoCapture(0)
    # Capture frame-by-frame
    ret, image = cap.read()
    cap.release()
    boxes, confs, clss = detector.prediction(image)
    image = detector.draw_boxes(image, boxes, confs, clss)
    #output = detector.prediction(image)
    #df = detector.filter_prediction(output, image)
    #if len(df) > 0:
    #    if (df['class_name']
    #            .str
    #            .contains('person|bird|cat|wine glass|cup|sandwich')
    #            .any()):
    #        day = datetime.now().strftime("%Y%m%d")
    #        directory = os.path.join(IMAGE_FOLDER, 'webcam', day)
    #        if not os.path.exists(directory):
    #            os.makedirs(directory)
    #        image = detector.draw_boxes(image, df)
    #        classes = df['class_name'].unique().tolist()
    #        hour = datetime.now().strftime("%H%M%S")
    #        filename_output = os.path.join(
    #                directory, "{}_{}_.jpg".format(hour, "-".join(classes))
    #                )
    #        cv2.imwrite(filename_output, image)


if __name__ == '__main__':
    CaptureContinous()
