import os
import cv2
import base64
import glob
from celery import Celery
from functools import reduce
from dotenv import load_dotenv
from importlib import import_module
from datetime import datetime, timedelta
from backend.centroidtracker import CentroidTracker
from backend.base_camera import BaseCamera
from backend.utils import reduce_tracking

load_dotenv()
Detector = import_module('backend.' + os.environ['DETECTION_MODEL']).Detector

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

IMAGE_FOLDER = "imgs"


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

            yield img

class Predictor(object):
    """Docstring for Predictor. """

    def __init__(self):
        self.detector = Detector()
        self.ct = CentroidTracker(maxDisappeared=50)

    def prediction(self, img, conf_th=0.3, conf_class=[]):
        output = self.detector.prediction(img)
        df = self.detector.filter_prediction(output, img, conf_th=conf_th, conf_class=conf_class)
        img = self.detector.draw_boxes(img, df)
        return img

    def object_track(self, img, conf_th=0.3, conf_class=[]):
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

    def img_to_base64(self, img):
        """encode as a jpeg image and return it"""
        buffer = cv2.imencode('.jpg', img)[1].tobytes()
        jpg_as_text = base64.b64encode(buffer)
        base64_string = jpg_as_text.decode('utf-8')
        return base64_string


@celery.task(bind=True)
def CaptureContinous(self, detector):
    cap = cv2.VideoCapture(0)
    _, image = cap.read()
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


@celery.task(bind=True)
def ObjectTracking(self):
    detector = Detector()
    myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', '*.jpg'),
                        recursive=True)
    newdict = reduce(lambda a, b: reduce_tracking(a,b), myiter, dict())
    startID = max(map(int, newdict.keys()), default=0) + 1
    ct = CentroidTracker(startID=startID)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError('Could not start camera.')

    try:
        while True:
            _, img = camera.read()
            output = detector.prediction(img)
            df = detector.filter_prediction(output, img)
            img = detector.draw_boxes(img, df)
            boxes = df[['x1', 'y1', 'x2', 'y2']].values
            previous_object_ID = ct.nextObjectID
            #self.update_state(state='PROGRESS',
            #        meta={
            #            'object_id': previous_object_ID,
            #            })
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
            #time.sleep(0.100)
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
    ObjectTracking()
