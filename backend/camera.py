import os
import io
import cv2
import glob
import time
import yaml
import numpy as np
from functools import reduce
from importlib import import_module
from datetime import datetime, timedelta
from .centroidtracker import CentroidTracker
from .base_camera import BaseCamera
from .utils import reduce_tracking

with open("config.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

IMAGE_FOLDER = "imgs"

class Camera(BaseCamera):
    # default value
    video_source = 0
    rotation = None
    detector = None
    camera = None
    ct = None

    def __init__(self, camera_config):
        if 'source' in camera_config:
            self.video_source = camera_config['source']
        if 'rotation' in camera_config:
            self.rotation = camera_config['rotation']
        if camera_config['source'] == 'picamera':
            self.frames = self.frames_pi
        elif camera_config['source'] == 'jetson':
            self.frames = self.frames_jetson
        else:
            self.frames = self.frames_pc

    def frames_pc(self):
        if self.camera is None or not self.camera.isOpened():
            self.load_camera()
        while True:
            # read current frame
            _, img = self.camera.read()

            if self.rotation:
                if self.rotation == 90:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                if self.rotation == 180:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                if self.rotation == 270:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            yield img

    def frames_pi(self):
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        self.PiCamera = PiCamera
        self.PiRGBArray = PiRGBArray
        self.camera = PiCamera()
        self.camera.rotation = self.rotation
        stream = io.BytesIO()
        for _ in self.camera.capture_continuous(stream, 'jpeg',
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

    def release(self):
        if self.video_source == 'picamera':
            self.camera.close()
        else:
            self.camera.release()

    def load_detector(self, startID=0):
        Detector = import_module(f"backend.{config['model']}").Detector
        self.detector = Detector()
        self.ct = CentroidTracker(maxDisappeared=50, startID=startID)

    def load_camera(self):
        self.camera = cv2.VideoCapture(self.video_source)
        if not self.camera.isOpened():
            raise RuntimeError('Could not start camera.')

    def CaptureContinous(self):
        if self.detector is None:
            self.load_detector()
        if self.video_source == 'picamera':
            WIDTH = 640
            HEIGHT = 480
            with self.PiCamera() as camera:
                camera.resolution = (1280, 960)  # twice height and widht
                camera.rotation = self.rotation
                camera.framerate = 10
                with self.PiRGBArray(camera, size=(WIDTH, HEIGHT)) as output:
                    camera.capture(output, 'bgr', resize=(WIDTH, HEIGHT))
                    image = output.array
        else:
            image = self.get_frame()
        output = self.detector.prediction(image)
        df = self.detector.filter_prediction(output, image)
        if len(df) > 0:
            if (df['class_name']
                    .str
                    .contains('person|bird|cat|wine glass|cup|sandwich')
                    .any()):
                day = datetime.now().strftime("%Y%m%d")
                directory = os.path.join(IMAGE_FOLDER, 'webcam', day)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                image = self.detector.draw_boxes(image, df)
                classes = df['class_name'].unique().tolist()
                hour = datetime.now().strftime("%H%M%S")
                filename_output = os.path.join(
                        directory, "{}_{}_.jpg".format(hour, "-".join(classes))
                        )
                cv2.imwrite(filename_output, image)

    def prediction(self, img, conf_th=0.3, conf_class=[]):
        if self.detector is None:
            self.load_detector()
        output = self.detector.prediction(img)
        df = self.detector.filter_prediction(output, img, conf_th=conf_th, conf_class=conf_class)
        img = self.detector.draw_boxes(img, df)
        return img

    def object_track(self, img, conf_th=0.3, conf_class=[]):
        if self.detector is None:
            self.load_detector()
        output = self.detector.prediction(img)
        df = self.detector.filter_prediction(output, img, conf_th=conf_th, conf_class=conf_class)
        img = self.detector.draw_boxes(img, df)
        boxes = df[['x1', 'y1', 'x2', 'y2']].values
        objects = self.ct.update(boxes)
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
            self.CaptureContinous()
            time.sleep(interval)

    def ObjectTracking(self):
        interval=config['beat_interval']
        if False:
            # search startID
            myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', '*.jpg'),
                                recursive=True)
            newdict = reduce(lambda a, b: reduce_tracking(a,b), myiter, dict())
            startID = max(map(int, newdict.keys()), default=0) + 1
        if self.detector is None:
            self.load_detector()

        try:
            while True:
                if self.video_source == 'picamera':
                    WIDTH = 640
                    HEIGHT = 480
                    with self.PiCamera() as camera:
                        camera.resolution = (1280, 960)  # twice height and widht
                        camera.rotation = self.rotation
                        camera.framerate = 10
                        with self.PiRGBArray(camera, size=(WIDTH, HEIGHT)) as output:
                            camera.capture(output, 'bgr', resize=(WIDTH, HEIGHT))
                            img = output.array
                else:
                    img = self.get_frame()
                output = self.detector.prediction(img)
                df = self.detector.filter_prediction(output, img)
                img = self.detector.draw_boxes(img, df)
                boxes = df[['x1', 'y1', 'x2', 'y2']].values
                previous_object_ID = self.ct.nextObjectID
                objects = self.ct.update(boxes)
                if len(boxes) > 0 and (df['class_name'].str.contains('person').any()) and previous_object_ID in list(objects.keys()):
                    for (objectID, centroid) in objects.items():
                        text = "ID {}".format(objectID)
                        cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                    day = datetime.now().strftime("%Y%m%d")
                    directory = os.path.join(IMAGE_FOLDER, 'webcam', day)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    ids = "-".join(list([str(i) for i in objects.keys()]))
                    hour = datetime.now().strftime("%H%M%S")
                    filename_output = os.path.join(
                            directory, "{}_person_{}_.jpg".format(hour, ids)
                            )
                    cv2.imwrite(filename_output, img)
                time.sleep(interval)
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
    camera = Camera(config['cameras'][0])
    camera.CaptureContinous()
    #camera.ObjectTracking()
