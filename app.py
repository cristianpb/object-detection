#!/usr/bin/env python3
import os
import cv2
import time
import threading
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, Response, render_template_string, send_from_directory, request

from ssd_detection import SSD
ssd = SSD()

WIDTH = 640
HEIGHT = 480

# Raspberry Pi camera module (requires picamera package)
from camera_pi import Camera, CameraPred, CameraStatic, CaptureContinous

# Webcam camera usin opencv module
#from camera_opencv import Camera, CameraPred, CaptureContinous

app = Flask(__name__)

@app.route('/<path:filename>')
def image(filename):
    try:
        w = int(request.args['w'])
        h = int(request.args['h'])
    except (KeyError, ValueError):
        return send_from_directory('.', filename)

    try:
        im = Image.open(filename)
        im.thumbnail((w, h), Image.ANTIALIAS)
        io = BytesIO()
        im.save(io, format='JPEG')
        return Response(io.getvalue(), mimetype='image/jpeg')

    except IOError:
        abort(404)

    return send_from_directory('.', filename)


@app.route('/images')
def images():
    images = []
    myclass = request.args.get('class', None)
    myhour = request.args.get('hour', None)
    myday = request.args.get('day', None)
    image_folder = './static/imgs/pi'
    hours = set()
    days = set()
    _objects = set()
    for filename in os.listdir(image_folder):
        if not filename.endswith('.jpg'):
            continue
        year, hour, objects, _ = filename.split("_")
        hours.add(hour[:2])
        days.add(year[6:])
        [_objects.add(x) for x in objects.split("-")]
        filename = os.path.join(image_folder, filename)
        if myclass:
            if myclass not in objects.split("-"):
                continue
        if myhour:
            if myhour != hour[:2]:
                continue
        if myday:
            if myday != year[6:]:
                continue
        im = Image.open(filename)
        w, h = im.size
        aspect = 1.0*w/h
        if aspect > 1.0*WIDTH/HEIGHT:
            width = min(w, WIDTH)
            height = width/aspect
        else:
            height = min(h, HEIGHT)
            width = height*aspect
        images.append({
            'width': int(width),
            'height': int(height),
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
        filename_output = "./static/imgs/webcam.jpg"
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
    filename_output = "./static/imgs/webcam.jpg"
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
