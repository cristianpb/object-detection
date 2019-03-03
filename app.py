#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response, render_template_string, send_from_directory, request
from PIL import Image
import cv2
import time
import threading
from ssd_detection import SSD
ssd = SSD()

# Raspberry Pi camera module (requires picamera package)
#from camera_pi import Camera
from camera_opencv import Camera

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
    WIDTH = 640
    HEIGHT = 640
    images = []
    for root, dirs, files in os.walk('.'):
        for filename in [os.path.join(root, name) for name in files]:
            if not filename.endswith('.jpg'):
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
        'images': images
    })


@app.route('/test')
def test():
    filename_input = "./static/imgs/image.jpeg"
    filename_output = "./static/imgs/image_box.jpg"
    image = cv2.imread(filename_input)
    ssd = SSD()
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
    ssd = SSD()
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
    return render_template('output.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#@app.route('/stream')
#def static_output():
#    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    #cam = Camera()
    #return Response(gen(cam),
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def job():
    print("READING")
    ssd = SSD()
    filename_output = "./static/imgs/outputcv.jpg"
    while(1):
        print("REAdiNG")
        cap = cv2.VideoCapture(0)
        # Capture frame-by-frame
        ret, image = cap.read()
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
        cap.release()
        time.sleep(2)

if __name__ == '__main__':
    thread = threading.Thread(target=job)
    thread.start()
    app.run(host='0.0.0.0', threaded=True)
    #app.run(host='127.0.0.1', threaded=True)
