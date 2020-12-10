#!/usr/bin/env python3
import os
import glob
import cv2
import json
import yaml
import pandas as pd
from functools import reduce
from importlib import import_module
from itertools import islice
from dotenv import load_dotenv
from datetime import datetime
from flask import Flask, Response, send_from_directory, request, Blueprint, abort
from backend.utils import (reduce_month, reduce_year, reduce_hour,
        reduce_object, reduce_tracking, img_to_base64)

with open("config.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

WIDTH = 320
HEIGHT = 240
IMAGE_FOLDER = 'imgs'
load_dotenv('.env')
if os.getenv('PORT'):
    PORT = int(str(os.getenv('PORT')))
else:
    PORT=5000

cameras = dict()

def load_camera(camera_config):
    Camera = import_module('backend.camera_{}'.format(camera_config['type'])).Camera
    cameras[camera_config['name']] = Camera(camera_config)

for camera in config['cameras']:
    load_camera(camera)

print(cameras)

#if os.getenv('CAMERA_STREAM'):
#    CameraStream = import_module('backend.camera_opencv').Camera
#    camera_stream = CameraStream()
#    camera_stream.video_source = os.getenv('CAMERA_STREAM')
#    camera_stream.rotation = os.getenv('CAMERA_STREAM_ROTATION')
#
#if os.getenv('CAMERA'):
#    Camera = import_module('backend.camera_' + os.environ['CAMERA']).Camera
#    Predictor = import_module('backend.camera_' + os.environ['CAMERA']).Predictor
#    camera = Camera()
#    predictor = Predictor()
#    celery = import_module('backend.camera_' + os.environ['CAMERA']).celery
#else:
#    print('Default USB camera')
#    from backend.camera_opencv import Camera

if os.getenv('BASEURL') and os.getenv('BASEURL') is not None:
    BASEURL=os.getenv('BASEURL').replace('\\', '')
else:
    BASEURL='/'

app = Flask(__name__)

# static html
blueprint_html = Blueprint('html', __name__, url_prefix=BASEURL)

@blueprint_html.route('/', defaults={'filename': 'index.html'})
@blueprint_html.route('/<path:filename>')
def show_pages(filename):
    return send_from_directory('../dist', filename)
app.register_blueprint(blueprint_html)

# API
blueprint_api = Blueprint('api', __name__, url_prefix=BASEURL)

@blueprint_api.route(os.path.join('/', IMAGE_FOLDER, '<path:filename>'))
def image_preview(filename):
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
        return Response(cv2.imencode('.jpg', im)[1].tobytes(),
                        mimetype='image/jpeg')

    except Exception as e:
        print(e)

    return send_from_directory('.', filename)


@blueprint_api.route('/api/delete', methods=['POST'])
def delete_image():
    filename = request.form.get('filename', None)
    try:
        os.remove(filename)
        return json.dumps({'status': filename})
    except Exception as e:
        print(e)
        return abort(404)

def get_data(item):
    if 'pi' in item:
        year = item.split('/')[2][:4]
        month = item.split('/')[2][4:6]
        day = item.split('/')[2][6:8]
        hour = item.split('/')[3][:2]
        minutes = item.split('/')[3][2:4]
        return dict(
                path=item, year=year, month=month, day=day,
                hour=hour, minutes=minutes
                )
    else:
        return dict(path=item)


@blueprint_api.route('/api/images')
def api_images():
    page = int(request.args.get('page', 0))
    page_size = int(request.args.get('page_size', 16))
    mydate = request.args.get('date', None)
    myyear = request.args.get('year', "????")
    mymonth = request.args.get('month', "??")
    myday = request.args.get('day', "??")
    myhour = request.args.get('hour', "??")
    myminutes = request.args.get('minutes', "??")
    mydetection = request.args.get('detected_object', "*")
    if mydate is not None:
        mydate = (datetime
                  .strptime(mydate, "%d/%m/%Y")
                  .strftime("%Y%m%d")
                  )
        myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', mydate, '*.jpg'),
                            recursive=True)
    elif (myyear != "????" or
          mymonth != "??" or
          myday != "??" or
          myhour != "??" or
          myminutes != "??" or
          mydetection != "*"):
        mypath = os.path.join(
                              IMAGE_FOLDER, '**',
                              f'{myyear}{mymonth}{myday}',
                              f'{myhour.zfill(2)}{myminutes}??*{mydetection}*.jpg')
        myiter = glob.iglob(mypath, recursive=True)
    else:
        myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', '*.jpg'),
                            recursive=True)

    start = page * page_size
    end = (page + 1) * page_size
    result = [get_data(i) for i in islice(myiter, start, end)]
    print('->> Start', start, 'end', end, 'len', len(result))
    return json.dumps(result)

@blueprint_api.route('/api/stream_image')
def stream_image():
    url = bool(request.args.get('url', False))
    detection = bool(request.args.get('detection', False))
    tracking = bool(request.args.get('tracking', False))
    if url:
        frame = camera_stream.get_frame()
    if detection:
        frame = predictor.prediction(frame, conf_th=0.3, conf_class=[])
    elif tracking:
        frame = predictor.object_track(frame, conf_th=0.5, conf_class=[1])
    return json.dumps(dict(img=img_to_base64(frame),
                      width=WIDTH,
                      height=HEIGHT))


@blueprint_api.route('/api/single_image')
def single_image():
    camera_name = request.args.get('cameraName', None)
    detection = bool(request.args.get('detection', False))
    tracking = bool(request.args.get('tracking', False))
    frame = None
    if camera_name:
        #frame = camera.get_frame()
        frame = cameras[camera_name].get_frame()
    #if detection:
    #    frame = predictor.prediction(frame, conf_th=0.3, conf_class=[])
    #elif tracking:
    #    frame = predictor.object_track(frame, conf_th=0.5, conf_class=[1])
    if frame is not None:
        return json.dumps(dict(img=img_to_base64(frame),
                          width=WIDTH,
                          height=HEIGHT))
    else:
        return dict(msg='no image')

myconditions = dict(
        month=reduce_month,
        year=reduce_year,
        hour=reduce_hour,
        detected_object=reduce_object,
        tracking_object=reduce_tracking,
        )


@blueprint_api.route('/api/list_files')
def list_folder():
    condition = request.args.get('condition', 'year')
    myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', '*.jpg'),
                        recursive=True)
    newdict = reduce(lambda a, b: myconditions[condition](a,b), myiter, dict())
    # year = item.split('/')[2][:4]
    # month = item.split('/')[2][4:6]
    # day = item.split('/')[2][6:8]
    # hour = item.split('/')[3][:2]
    # minutes = item.split('/')[3][2:4]
    # return json.dumps({k: v for k, v in sorted(newdict.items(), key=lambda item: item[1], reverse=True)})
    return json.dumps(newdict)


@blueprint_api.route('/api/task/status/<task_id>')
def taskstatus(task_id):
    #task = ObjectTracking.AsyncResult(task_id)
    task = predictor.continous_object_tracking.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'object_id': 0,
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'object_id': task.info.get('object_id', 0),
        }
    else:
        response = {
            'state': task.state,
            'object_id': task.info.get('object_id', 0),
        }
    return json.dumps(response)


@blueprint_api.route('/api/task/launch')
def launch_object_tracking():
    task = predictor.ObjectTracking.delay()
    #task = predictor.continous_object_tracking.delay()
    return json.dumps({"task_id": task.id})

@blueprint_api.route('/api/task/kill/<task_id>')
def killtask(task_id):
    response = celery.control.revoke(task_id, terminate=True, wait=True, timeout=10)
    return json.dumps(response)

@blueprint_api.route('/api/beat/launch')
def launch_beat():
    task = predictor.PeriodicCaptureContinous.delay()
    return json.dumps({"task_id": task.id})

#@blueprint_api.route('/tracking/read')
#def read_tracking():
#    df =pd.read_csv('{}/tracking.csv'.format(IMAGE_FOLDER), header=None, names=['date', 'hour', 'idx', 'coord'])
#    print(df.head())
#    print(df.to_dict(orient='records'))
#    return json.dumps(df.to_dict(orient='records'))

@blueprint_api.route('/api/config')
def read_config():
    return config

app.register_blueprint(blueprint_api)

if __name__ == '__main__':
    app.run(
            host='0.0.0.0',
            debug=bool(os.getenv('DEBUG')),
            threaded=False,
            port=PORT
            )
