#!/usr/bin/env python3
import os
import re
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
from multiprocessing import Process
from flask import Flask, Response, send_from_directory, request, Blueprint, abort
from backend.utils import (reduce_year, reduce_year_month, reduce_month,
        reduce_day, reduce_year, reduce_hour, reduce_object, reduce_tracking,
        img_to_base64)

with open("config.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    jobs = dict()

WIDTH = 320
HEIGHT = 240
IMAGE_FOLDER = 'imgs'
load_dotenv('.env')
if os.getenv('PORT'):
    PORT = int(str(os.getenv('PORT')))
else:
    PORT=5000

folder_regex = re.compile('imgs/webcam|imgs/pi')

Camera = import_module('backend.camera_{}'.format(config['device'])).Camera
cameras = dict()

for camera_config in config['cameras']:
    cameras[camera_config['name']] = Camera(camera_config)

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
    if folder_regex.match(item):
        year = item.split('/')[2][:4]
        month = item.split('/')[2][4:6]
        day = item.split('/')[2][6:8]
        hour = item.split('/')[3][:2]
        minutes = item.split('/')[3][2:4]
        seconds = item.split('/')[3][4:6]
        return dict(
                path=item, year=year, month=month, day=day,
                hour=hour, minutes=minutes, seconds=seconds
                )
    else:
        return dict(path=item)

def get_range(units, digit_number):
    dec = []
    for digit in range(digit_number):
        dec.append(set())
    for digit_candidate in str(units).split(","):
        if len(digit_candidate) == digit_number:
            for digit in range(digit_number):
                dec[digit].add(digit_candidate[digit])
        else:
            for digit in range(digit_number):
                if digit == (digit_number - 1):
                    dec[digit].add(digit_candidate)
                else:
                    dec[digit].add("0")
    return "".join(["[" + "".join(x) + "]" for x in dec])

@blueprint_api.route('/api/images')
def api_images():
    page = int(request.args.get('page', 0))
    page_size = int(request.args.get('page_size', 16))
    mydate = request.args.get('date', None)
    myyear = request.args.get('years', "????")
    mymonth = request.args.get('months', "??")
    myday = request.args.get('days', "??")
    myhour = request.args.get('hours', "??")
    myminutes = request.args.get('minutes', "??")
    mydetection = request.args.get('detected_object', "*")
    if mydate is not None:
        date = (datetime
                  .strptime(mydate, "%d/%m/%Y")
                  .strftime("%Y%m%d")
                  )
    else:
        if myyear != "????":
            if myyear:
                myyear = get_range(myyear, 4)
            else:
                myyear = "????"

        if mymonth != "??":
            if mymonth:
                mymonth = get_range(mymonth, 2)
            else:
                mymonth = "??"

        if myday != "??":
            if myday:
                myday = get_range(myday, 2)
            else:
                myday = "??"

        date = f"{myyear}{mymonth}{myday}"

    if myhour != "??":
        if myhour:
            myhour = get_range(myhour, 2)
        else:
            myhour = "??"

    if myminutes != "??":
        if myminutes:
            myminutes = get_range(myminutes, 2)
        else:
            myminutes = "??"

    mypath = os.path.join(
                          IMAGE_FOLDER,
                          '*',
                          f'{date}',
                          f'{myhour.zfill(2)}{myminutes}??*{mydetection}*.jpg').replace('***', '*')
    myiter = glob.iglob(mypath, recursive=True)
    start = page * page_size
    end = (page + 1) * page_size
    result = [get_data(i) for i in islice(myiter, start, end)]
    print('->> Start', start, 'end', end, 'len', len(result))
    return dict(page=page, page_size=page_size, images=result)


@blueprint_api.route('/api/single_image')
def single_image():
    camera_name = request.args.get('cameraName', None)
    detection = request.args.get('detection', 'false')
    tracking = request.args.get('tracking', 'false')
    frame = None
    if camera_name:
        frame = cameras[camera_name].get_frame()
    if detection == 'true':
        frame = cameras[camera_name].prediction(frame, conf_th=0.3, conf_class=[])
    elif tracking == 'true':
        frame = cameras[camera_name].object_track(frame, conf_th=0.5, conf_class=[1])
    if frame is not None:
        return json.dumps(dict(img=img_to_base64(frame),
                          width=WIDTH,
                          height=HEIGHT))
    else:
        return dict(msg='no image')

myconditions = dict(
        years=reduce_year,
        months=reduce_month,
        days=reduce_day,
        year_month=reduce_year_month,
        hours=reduce_hour,
        detected_objects=reduce_object,
        tracking_objects=reduce_tracking,
        )


@blueprint_api.route('/api/list_files')
def list_folder():
    condition = request.args.get('condition', 'years')
    myiter = glob.iglob(os.path.join(IMAGE_FOLDER, '**', '*.jpg'),
                        recursive=True)
    newdict = reduce(lambda a, b: myconditions[condition](a,b), myiter, dict())
    # year = item.split('/')[2][:4]
    # month = item.split('/')[2][4:6]
    # day = item.split('/')[2][6:8]
    # hour = item.split('/')[3][:2]
    # minutes = item.split('/')[3][2:4]
    # return json.dumps({k: v for k, v in sorted(newdict.items(), key=lambda item: item[1], reverse=True)})
    return newdict

@blueprint_api.route('/api/task/start')
def task_launch():
    camera_name = request.args.get('camera', None)
    task_name = request.args.get('task', None)
    if camera_name is None:
        return dict(msg="Camera name is missing")
    if task_name is None:
        return dict(msg="Task name is missing")
    if task_name == 'tracking':
        jobs[task_name] = Process(target=cameras[camera_name].ObjectTracking)
        jobs[task_name].start()
        jobs[task_name].date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        return dict(
            is_alive=jobs[task_name].is_alive(),
            pid=jobs[task_name].pid,
            name=jobs[task_name].name,
            date=jobs[task_name].date
            )
    elif task_name == 'detection':
        print("hello", cameras[camera_name].camera)
        jobs[task_name] = Process(target=cameras[camera_name].PeriodicCaptureContinous)
        jobs[task_name].start()
        jobs[task_name].date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        return dict(
            is_alive=jobs[task_name].is_alive(),
            pid=jobs[task_name].pid,
            name=jobs[task_name].name,
            date=jobs[task_name].date
            )
    else:
        return dict(msg="Don't know the task you want.\
                Try 'detection' or 'tracking'")

@blueprint_api.route('/api/task/stop')
def task_kill():
    task_name = request.args.get('task', None)
    if task_name is None:
        return dict(msg="Task name is missing")
    if task_name not in jobs:
        return dict(msg="Task doesn't exists")
    jobs[task_name].terminate()
    jobs[task_name].join()
    return dict(
        is_alive=jobs[task_name].is_alive(),
        pid=jobs[task_name].pid,
        name=jobs[task_name].name,
        start=jobs[task_name].date,
        end=datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        )


@blueprint_api.route('/api/task/status')
def task_status():
    task_name = request.args.get('task', None)
    if task_name is None:
        return dict(msg="Task name is missing")
    if task_name not in jobs:
        return dict(msg="Task doesn't exists")
    return dict(
        is_alive=jobs[task_name].is_alive(),
        pid=jobs[task_name].pid,
        name=jobs[task_name].name,
        date=jobs[task_name].date
        )

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
