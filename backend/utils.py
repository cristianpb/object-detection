"""Utilities for logging."""
import os
import re
import cv2
import numpy as np
import logging
import time
import base64

ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
if os.getenv('LOG_LEVEL') == 'DEBUG':
    level = logging.DEBUG
elif os.getenv('LOG_LEVEL') == 'INFO':
    level = logging.INFO
elif os.getenv('LOG_LEVEL') == 'ERROR':
    level = logging.ERROR
else:
    level = logging.ERROR
logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )
logger = logging.getLogger(__name__)

folder_regex = re.compile('imgs/webcam|imgs/pi')

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger = logging.getLogger(method.__name__)
        logger.debug('{} {:.3f} sec'.format(method.__name__, te-ts))
        return result

    return timed

def img_to_base64(img):
    """encode as a jpeg image and return it"""
    buffer = cv2.imencode('.jpg', img)[1].tobytes()
    jpg_as_text = base64.b64encode(buffer)
    base64_string = jpg_as_text.decode('utf-8')
    return base64_string

def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.

    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.

    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


def reduce_year_month(accu, item):
    if folder_regex.match(item) is None:
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

def reduce_month(accu, item):
    if folder_regex.match(item) is None:
        return accu
    month = item.split('/')[2][4:6]
    if month in accu:
        accu[month] +=1
    else:
        accu[month] = 1
    return accu

def reduce_day(accu, item):
    if folder_regex.match(item) is None:
        return accu
    day = item.split('/')[2][6:8]
    if day in accu:
        accu[day] +=1
    else:
        accu[day] = 1
    return accu


def reduce_year(accu, item):
    if folder_regex.match(item) is None:
        return accu
    year = item.split('/')[2][:4]
    if year in accu:
        accu[year] +=1
    else:
        accu[year] = 1
    return accu


def reduce_hour(accu, item):
    if folder_regex.match(item) is None:
        return accu
    condition = item.split('/')[3][:2]
    if condition in accu:
        accu[condition] +=1
    else:
        accu[condition] = 1
    return accu


def reduce_object(accu, item):
    if folder_regex.match(item) is None:
        return accu
    condition = item.split('/')[3].split('_')[1].split('-')
    for val in condition:
        if val in accu:
            accu[val] +=1
        else:
            accu[val] = 1
    return accu

def reduce_tracking(accu, item):
    if folder_regex.match(item) is None:
        return accu
    condition = item.split('/')[3].split('_')[2].split('-')
    for val in condition:
        if val in accu:
            accu[val] +=1
        else:
            accu[val] = 1
    return accu

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
