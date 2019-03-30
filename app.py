#!/usr/bin/env python3
import os
import glob
import cv2
import time
import threading
from datetime import datetime
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, Response, render_template_string, send_from_directory, request

from ssd_detection import SSD
ssd = SSD()

WIDTH = 320
HEIGHT = 240

# Raspberry Pi camera module (requires picamera package)
from camera_pi import Camera, CameraPred, CameraStatic, CaptureContinous

# Webcam camera usin opencv module
#from camera_opencv import Camera, CameraPred, CaptureContinous

app = Flask(__name__)

@app.route('/<path:filename>')
def image(filename):
    w = request.args.get('w', None)
    h = request.args.get('h', None)
    date = request.args.get('date', None)

    try:
        im = cv2.imread(filename)
        if w and h:
            w, h = int(w), int(h)
            im = cv2.resize(im, (w, h))
        elif date:
            date = (datetime
                    .strptime(date, "%Y%m%d_%H%M%S")
                    .strftime("%d %b %-H:%M")
                    )
            img_h, img_w = im.shape[:-1]
            cv2.putText(
                    im, "{}".format(date), (0, int(img_h*0.98)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return Response(cv2.imencode('.jpg', im)[1].tobytes(), mimetype='image/jpeg')

    except Exception as e:
        print(e)

    return send_from_directory('.', filename)


@app.route('/images')
def images():
    images = []
    myclasses = request.args.getlist('class', None)
    myhour = request.args.get('hour', None)
    mydate = request.args.get('date', None)
    myday = None
    if mydate:
        myday = mydate[3:5]
    image_folder = './imgs/'
    hours = set()
    days = set()
    _objects = set()
    for filename in glob.iglob(image_folder + '**/*.jpg', recursive=True):
        base = os.path.basename(filename)
        if not base.endswith('.jpg') or base in ['image_box.jpg', 'image_box_text.jpg']:
            continue
        year, hour = None, None
        if len(base.split("_")) == 4:
            year, hour, objects, _ = base.split("_")
            hours.add(hour[:2])
            days.add(year[6:])
            [_objects.add(x) for x in objects.split("-")]
            if len(myclasses) > 0:
                list_intersection  = [value for value in myclasses if value in objects.split("-")]
                if len(list_intersection) == 0:
                    continue
            if myhour:
                if myhour != hour[:2]:
                    continue
            if myday:
                if myday != year[6:]:
                    continue
            year = datetime.strptime("{}_{}".format(year, hour), "%Y%m%d_%H%M%S")

        images.append({
            'width': int(WIDTH),
            'height': int(HEIGHT),
            'date': year,
            'src': filename
        })

    return render_template("preview.html", **{
        'images': images,
        'days': list(days),
        'hours': list(hours),
        'objects': list(_objects)
    })


@app.route('/picam')
def picam():
    for frame in CameraStatic(WIDTH, HEIGHT):
        image = frame.array
        filename_output = "./imgs/webcam.jpg"
        output = ssd.prediction(image)
        output = ssd.filter_prediction(output)
        image = ssd.draw_boxes(image, output)
        cv2.imwrite(filename_output,image)
        height, width, _ = image.shape
        images = [{
                    'width': int(width),
                    'height': int(height),
                    'src': filename_output
                }]
        break
    return render_template("preview.html", **{
        'images': images
    })

@app.route('/webcam')
def webcam():
    video_capture = cv2.VideoCapture(0)
    ret, image = video_capture.read()
    time.sleep(2)
    ret, image = video_capture.read()
    time.sleep(2)
    filename_output = "./imgs/webcam.jpg"
    output = ssd.prediction(image)
    output = ssd.filter_prediction(output)
    image = ssd.draw_boxes(image, output)
    cv2.imwrite(filename_output,image)
    height, width, _ = image.shape
    images = [{
                'width': int(width),
                'height': int(height),
                'src': filename_output
            }]
    video_capture.release()
    return render_template("preview.html", **{
        'images': images
    })

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_pred')
def video_pred():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(CameraPred()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    thread = threading.Thread(target=CaptureContinous)
    thread.start()
    app.run(host='0.0.0.0', threaded=True)
