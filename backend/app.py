#!/usr/bin/env python3
import os
import glob
import cv2
import json
import pandas as pd
from functools import reduce
from importlib import import_module
from itertools import islice
from dotenv import load_dotenv
from datetime import datetime
from flask import Flask, Response, send_from_directory, request, abort

WIDTH = 320
HEIGHT = 240
IMAGE_FOLDER = 'imgs'
load_dotenv('.env')
if os.getenv('PORT'):
    PORT = int(str(os.getenv('PORT')))
else:
    PORT=5000

if os.getenv('CAMERA'):
    Camera = import_module('backend.camera_' + os.environ['CAMERA']).Camera
    ObjectTracking = import_module('backend.camera_' + os.environ['CAMERA']).ObjectTracking
    celery = import_module('backend.camera_' + os.environ['CAMERA']).celery
else:
    print('Default USB camera')
    from backend.camera_opencv import Camera

app = Flask(__name__)


@app.route(os.path.join('/', IMAGE_FOLDER, '<path:filename>'))
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


@app.route('/api/delete', methods=['POST'])
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


@app.route('/api/images')
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


@app.route('/api/single_image')
def single_image():
    detection = bool(request.args.get('detection', False))
    tracking = bool(request.args.get('tracking', False))
    frame = Camera().get_frame()
    if detection:
        frame = Camera().prediction(frame)
    elif tracking:
        frame = Camera().object_track(frame)
    return json.dumps(dict(img=Camera().img_to_base64(frame),
                      width=WIDTH,
                      height=HEIGHT))

def reduce_month(accu, item):
    if 'pi' not in item:
        return accu
    year = item.split('/')[2][:4]
    if year not in accu:
        accu[year] = dict()
    month = item.split('/')[2][4:6]
    if month in accu[year]:
        accu[year][month] +=1
    else:
        accu[year][month] = 1
    return accu

def reduce_year(accu, item):
    if 'pi' not in item:
        return accu
    year = item.split('/')[2][:4]
    if year in accu:
        accu[year] +=1
    else:
        accu[year] = 1
    return accu


def reduce_hour(accu, item):
    if 'pi' not in item:
        return accu
    condition = item.split('/')[3][:2]
    if condition in accu:
        accu[condition] +=1
    else:
        accu[condition] = 1
    return accu


def reduce_object(accu, item):
    if 'pi' not in item:
        return accu
    condition = item.split('/')[3].split('_')[1].split('-')
    for val in condition:
        if val in accu:
            accu[val] +=1
        else:
            accu[val] = 1
    return accu

myconditions = dict(
        month=reduce_month,
        year=reduce_year,
        hour=reduce_hour,
        detected_object=reduce_object,
        )


@app.route('/api/list_files')
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


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = ObjectTracking.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', ''),
            'partial_result': task.info.get('partial_result', list())
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return json.dumps(response)


@app.route('/launch')
def launch_object_tracking():
    task = ObjectTracking.delay()
    #return json.dumps({"task_id": task})
    return json.dumps({"task_id": task.id})

@app.route('/killtask/<task_id>')
def killtask(task_id):
    response = celery.control.revoke(task_id, terminate=True)
    return json.dumps(response)

@app.route('/tracking/read')
def read_tracking():
    df =pd.read_csv('{}/tracking.csv'.format(IMAGE_FOLDER), header=None, names=['date', 'hour', 'idx', 'coord'])
    print(df.head())
    print(df.to_dict(orient='records'))
    return json.dumps(df.to_dict(orient='records'))

@app.route('/')
def status():
    return send_from_directory('../dist', "index.html")


@app.route('/<path:path>')
def build(path):
    return send_from_directory('../dist', path)


if __name__ == '__main__':
    app.run(
            host='0.0.0.0',
            debug=bool(os.getenv('DEBUG')),
            threaded=False,
            port=int(str(os.getenv('PORT')))
            )
