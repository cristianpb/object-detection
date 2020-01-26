import os
import cv2
import base64
from celery import Celery
from dotenv import load_dotenv
from importlib import import_module
from datetime import datetime, timedelta
from centroidtracker import CentroidTracker
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
    def prediction(img, conf_class=[]):
        boxes, confs, clss = detector.prediction(img, conf_class=conf_class)
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


def ObjectTracking():
    ct = CentroidTracker()
    (H, W) = (None, None)
    camera = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        raise RuntimeError('Could not start camera.')

    try:
        while True:
            _, img = camera.read()
            boxes, confs, clss = detector.prediction(img, conf_class=[1])
            img = detector.draw_boxes(img, boxes, confs, clss)
            if len(boxes) > 0 and 1 in clss:
                #print("detected")
                #print("conf", confs)
                #print('clss', clss)

                objects = ct.update(boxes)
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
                hour = datetime.now().strftime("%H%M%S")
                filename_output = os.path.join(
                        directory, "{}_{}_.jpg".format(hour, "person")
                        )
                cv2.imwrite(filename_output, img)

    except KeyboardInterrupt:
        print('interrupted!')
        camera.release()
        print(type(objects))
        print(objects)
        cv2.imwrite("last.jpg", img)


if __name__ == '__main__':
    #CaptureContinous()
    ObjectTracking()
