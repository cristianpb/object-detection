import cv2
import io
import socket
import struct
import time
import pickle
#import zlib

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 8485))
connection = client_socket.makefile('wb')

cam = cv2.VideoCapture(0)

cam.set(3, 320);
cam.set(4, 240);

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
#    data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(frame, 0)
    size = len(data)


    print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1

cam.release()



#import socket
#import time
#
#import cv2
#
#capture = cv2.VideoCapture(0)
#sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#sock.connect(('localhost', 50505))
#
#while True:
#    ret, frame = capture.read()
#    data = cv2.imencode('.jpg', frame)[1].tostring()
#    sock.sendall(data)
#    time.sleep(2)
