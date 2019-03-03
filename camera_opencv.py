import cv2
from base_camera import BaseCamera
from ssd_detection import SSD
ssd = SSD()

class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
    #def frames(self):
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
