import io
import time
from datetime import datetime
import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import cv2
from base_camera import BaseCamera

from ssd_detection import SSD
ssd = SSD()

WIDTH = 640
HEIGHT = 480

class CameraPred(BaseCamera):
    @staticmethod
    def frames():
        with PiCamera() as camera:
            camera.rotation = 180

            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, 'jpeg',
                                                 use_video_port=True):
                # return current frame
                #stream.seek(0)
                #yield stream.read()

                #_stream = stream.getvalue()
                #data = np.fromstring(_stream, dtype=np.uint8)
                #img = cv2.imdecode(data, 1)
                #yield _stream

                _stream = stream.getvalue()
                data = np.fromstring(_stream, dtype=np.uint8)
                img = cv2.imdecode(data, 1)
                # Prediction 
                output = ssd.prediction(img)
                output = ssd.filter_prediction(output)
                img = ssd.draw_boxes(img, output)
                yield cv2.imencode('.jpg', img)[1].tobytes()

                # reset stream for next frame
                stream.seek(0)
                stream.truncate()


class Camera(BaseCamera):
    @staticmethod
    def frames():
        with PiCamera() as camera:
            camera.rotation = 180
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, 'jpeg',
                                                 use_video_port=True):
                # return current frame
                stream.seek(0)
                yield stream.read()

                # reset stream for next frame
                stream.seek(0)
                stream.truncate()


def CameraStatic():
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.rotation = 180
    camera.resolution = (WIDTH, HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(WIDTH, HEIGHT))
    rawCapture.truncate(0)
    return camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)


def CaptureContinous():
    camera = PiCamera()
    camera.rotation = 180
    camera.resolution = (WIDTH, HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(WIDTH, HEIGHT))
    rawCapture.truncate(0)
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        print("Capturing")
        image = frame.array
        output = ssd.prediction(image)
        df = ssd.filter_prediction(output, image)
        if len(df) > 0:
            image = ssd.draw_boxes(image, df)
            classes = df['class_name'].unique().tolist()
            today = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_output = "./imgs/pi/{}_{}_.jpg".format(today, "-".join(classes))
            cv2.imwrite(filename_output, image)
        rawCapture.truncate(0)
        time.sleep(20)
