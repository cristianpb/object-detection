import os
import io
import cv2
import time
import glob
import yaml
import numpy as np
from functools import reduce
from datetime import datetime, timedelta
from importlib import import_module
from picamera.array import PiRGBArray
from picamera import PiCamera
from backend.centroidtracker import CentroidTracker
from backend.base_camera import BaseCamera
from backend.utils import reduce_tracking

with open("config.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

Detector = import_module('backend.' + os.environ['DETECTION_MODEL']).Detector
detector = None
ct = None

WIDTH = 640
HEIGHT = 480
IMAGE_FOLDER = "./imgs"


class Camera(BaseCamera):

    def __init__(self, config):
        if config['source']:
            self.set_video_source(config['source'])
        if config['rotation']:
            self.rotation = config['rotation']

    @staticmethod
    def frames():
        with PiCamera() as camera:
            camera.rotation = config['rotation']
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, 'jpeg',
                                               use_video_port=True):
                # return current frame
                stream.seek(0)
                _stream = stream.getvalue()
                data = np.fromstring(_stream, dtype=np.uint8)
                img = cv2.imdecode(data, 1)
                yield img

                # reset stream for next frame
                stream.seek(0)
                stream.truncate()

def load_detector():
    global detector, ct
    detector = Detector()
    ct = CentroidTracker(maxDisappeared=20)

def CaptureContinous():
    detector = Detector()
    with PiCamera() as camera:
        camera.resolution = (1280, 960)  # twice height and widht
        camera.rotation = int(str(os.environ['CAMERA_ROTATION']))
        camera.framerate = 10
        with PiRGBArray(camera, size=(WIDTH, HEIGHT)) as output:
            camera.capture(output, 'bgr', resize=(WIDTH, HEIGHT))
            image = output.array
            result = detector.prediction(image)
            df = detector.filter_prediction(result, image)
            if len(df) > 0:
                if (df['class_name']
                        .str
                        .contains('person|bird|cat|wine glass|cup|sandwich')
                        .any()):
                    day = datetime.now().strftime("%Y%m%d")
                    directory = os.path.join(IMAGE_FOLDER, 'pi', day)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    image = detector.draw_boxes(image, df)
                    classes = df['class_name'].unique().tolist()
                    hour = datetime.now().strftime("%H%M%S")
                    filename_output = os.path.join(
                            directory,
                            "{}_{}_.jpg".format(hour, "-".join(classes))
                            )
                    cv2.imwrite(filename_output, image)


class Predictor(object):
    """Docstring for Predictor. """

    def prediction(self, img, conf_th=0.3, conf_class=[]):
        global detector
        if detector is None:
            load_detector()
        output = detector.prediction(img)
        df = detector.filter_prediction(output, img, conf_th=conf_th, conf_class=conf_class)
        img = detector.draw_boxes(img, df)
        return img

    def object_track(self, img, conf_th=0.3, conf_class=[]):
        global detector, ct
        if detector is None:
            load_detector()
        output = detector.prediction(img)
        df = detector.filter_prediction(output, img, conf_th=conf_th, conf_class=conf_class)
        img = detector.draw_boxes(img, df)
        boxes = df[['x1', 'y1', 'x2', 'y2']].values
        objects = ct.update(boxes)
        if len(boxes) > 0 and (df['class_name'].str.contains('person').any()):
            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        return img

    def PeriodicCaptureContinous(self):
        interval=config['beat_interval']
        while True:
            CaptureContinous()
            time.sleep(interval)

    def ObjectTracking(self):
        interval=config['beat_interval']
        detector = Detector()
        myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', '*.jpg'),
                            recursive=True)
        newdict = reduce(lambda a, b: reduce_tracking(a,b), myiter, dict())
        startID = max(map(int, newdict.keys()), default=0) + 1
        ct = CentroidTracker(startID=startID)
        with PiCamera() as camera:
            camera.resolution = (1280, 960)  # twice height and widht
            camera.rotation = int(str(os.environ['CAMERA_ROTATION']))
            camera.framerate = 10
            with PiRGBArray(camera, size=(WIDTH, HEIGHT)) as output:
                while True:
                    camera.capture(output, 'bgr', resize=(WIDTH, HEIGHT))
                    img = output.array
                    result = detector.prediction(img)
                    df = detector.filter_prediction(result, img)
                    img = detector.draw_boxes(img, df)
                    boxes = df[['x1', 'y1', 'x2', 'y2']].values
                    previous_object_ID = ct.nextObjectID
                    objects = ct.update(boxes)
                    if len(boxes) > 0 and (df['class_name'].str.contains('person').any()) and previous_object_ID in list(objects.keys()):
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
                    time.sleep(interval)


if __name__ == '__main__':
    predictor = Predictor()
    CaptureContinous()
