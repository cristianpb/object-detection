import cv2
from backend.ssd_detection import Detector as Detector_SSD
from backend.yolo_detection import Detector as Detector_Yolo
from backend.motion import Detector as Detector_Motion
from backend.cascade import Detector as Detector_Cascade


def test_ssd():
    image = cv2.imread("./imgs/image.jpeg")

    detector = Detector_SSD()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    print(df)
    image = detector.draw_boxes(image, df)
    cv2.imwrite("./imgs/outputcv.jpg", image)


def test_yolo():
    image = cv2.imread("./imgs/image.jpeg")

    detector = Detector_Yolo()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    print(df)
    image = detector.draw_boxes(image, df)
    cv2.imwrite("./imgs/outputcv.jpg", image)


def test_motion():
    image = cv2.imread("./imgs/image.jpeg")
    print(image.shape)

    detector = Detector_Motion()

    image2 = cv2.imread("./imgs/image_box.jpg")
    print(image2.shape)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.GaussianBlur(image2, (21, 21), 0)
    detector.avg = image2.astype(float)

    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    print(df)
    image = detector.draw_boxes(image, df)

    cv2.imwrite("./imgs/outputcv.jpg", image)


def test_cascade():
    image = cv2.imread("./imgs/image.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = Detector_Cascade()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    image = detector.draw_boxes(image, df)

    cv2.imwrite("./imgs/outputcv.jpg", image)
