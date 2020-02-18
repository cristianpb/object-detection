import os
import cv2
import glob
import base64
import time
import numpy as np
from celery import Celery
from functools import reduce
from dotenv import load_dotenv
from importlib import import_module
from datetime import datetime, timedelta
from backend.centroidtracker import CentroidTracker
from backend.base_camera import BaseCamera
from backend.utils import reduce_tracking

load_dotenv('.env')
Detector = import_module('backend.' + os.environ['DETECTION_MODEL']).Detector

celery = Celery("app")
celery.conf.update(
        broker_url='redis://localhost:6379/0',
        result_backend='redis://localhost:6379/0',
        beat_schedule={
            "photos_SO": {
                "task": "backend.camera_pi.CaptureContinous",
                "schedule": timedelta(
                    seconds=int(str(os.environ['BEAT_INTERVAL']))
                    ),
                "args": []
                }
            }
)

IMAGE_FOLDER = "imgs"

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


class Predictor(object):
    """Docstring for Predictor. """

    def __init__(self):
        self.detector = Detector()
        self.ct = CentroidTracker(maxDisappeared=20)

    def prediction(self, img, conf_th=0.3, conf_class=[]):
        output = self.detector.prediction(img)
        boxes, confs, clss = self.detector.filter_prediction(output, img, conf_th=conf_th, conf_class=conf_class)
        img = self.detector.draw_boxes(img, boxes, confs, clss)
        return img

    def object_track(self, img, conf_th=0.3, conf_class=[]):
        output = self.detector.prediction(img)
        boxes, confs, clss = self.detector.filter_prediction(output, img, conf_th=conf_th, conf_class=conf_class)
        img = self.detector.draw_boxes(img, boxes, confs, clss)
        objects = self.ct.update(boxes)
        if len(boxes) > 0 and 1 in clss:
            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        return img

    def img_to_base64(self, img):
        """encode as a jpeg image and return it"""
        buffer = cv2.imencode('.jpg', img)[1].tobytes()
        jpg_as_text = base64.b64encode(buffer)
        base64_string = jpg_as_text.decode('utf-8')
        return base64_string


@celery.task(bind=True)
def CaptureContinous(self, detector):
    pass


@celery.task(bind=True)
def ObjectTracking(self):
    detector = Detector()
    myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', '*.jpg'),
                        recursive=True)
    newdict = reduce(lambda a, b: reduce_tracking(a,b), myiter, dict())
    startID = max(map(int, newdict.keys()), default=0) + 1
    ct = CentroidTracker(startID=startID)
    camera = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        raise RuntimeError('Could not start camera.')

    try:
        while True:
            _, img = camera.read()
            boxes, confs, clss = detector.prediction(img, conf_th=0.8, conf_class=[1])
            img = detector.draw_boxes(img, boxes, confs, clss)
            previous_object_ID = ct.nextObjectID
            objects = ct.update(boxes)
            if len(boxes) > 0 and 1 in clss and previous_object_ID in list(objects.keys()):
                print("detected {} {} {} {}".format(ct.nextObjectID, confs, objects, boxes))

                # loop over the tracked objects
                for (objectID, centroid) in objects.items():
                    text = "ID {}".format(objectID)
                    cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                day = datetime.now().strftime("%Y%m%d")
                directory = os.path.join(IMAGE_FOLDER, 'pi', day)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                ids = "-".join(list([str(i) for i in objects.keys()]))
                hour = datetime.now().strftime("%H%M%S")
                filename_output = os.path.join(
                        directory, "{}_person_{}_.jpg".format(hour, ids)
                        )
                cv2.imwrite(filename_output, img)
            time.sleep(0.100)
    except KeyboardInterrupt:
        print('interrupted!')
        camera.release()
        print(type(objects))
        print(objects)
    except Exception as e:
        print('interrupted! by:')
        print(e)
        camera.release()
        print(type(objects))
        print(objects)


if __name__ == '__main__':
    detector = Detector()
    #CaptureContinous(detector)
    ObjectTracking(detector)
