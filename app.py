#!/usr/bin/env python3
import os
import glob
import cv2
import json
from itertools import islice
import threading
from datetime import datetime
from flask import Flask, Response, send_from_directory, request

WIDTH = 320
HEIGHT = 240
IMAGE_FOLDER = 'imgs'

# Raspberry Pi camera module (requires picamera package)
#from camera_pi import Camera, CameraPred, CaptureContinous

# Webcam camera usin opencv module
from camera_opencv import Camera, CameraPred, CaptureContinous

app = Flask(__name__)


@app.route(os.path.join('/', IMAGE_FOLDER, '<path:filename>'))
def image(filename):
    w = request.args.get('w', None)
    h = request.args.get('h', None)
    date = request.args.get('date', None)

    try:
        im = cv2.imread(os.path.join(IMAGE_FOLDER, filename))
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


@app.route('/api/images')
def images():
    page = int(request.args.get('page', 0))
    page_size = int(request.args.get('page_size', 12))
    mydate = request.args.get('date', None)
    if mydate is not None:
        myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', mydate, '*.jpg'),
                            recursive=True)
    else:
        myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', '*.jpg'),
                            recursive=True)
    start = page * page_size
    end = (page + 1) * page_size
    result = [i for i in islice(myiter, start, end)]
    print('->> Start', start, 'end', end, 'len', len(result))
    return json.dumps(result)


@app.route('/')
def status():
    return send_from_directory('./dist', "index.html")


@app.route('/<path:path>')
def build(path):
    return send_from_directory('./dist', path)


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
