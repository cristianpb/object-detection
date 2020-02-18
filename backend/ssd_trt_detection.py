import os
import cv2
import json
import ctypes
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This is needed for initializing CUDA driver
from backend.utils import timeit, draw_boxed_text

conf_th = 0.3
INPUT_HW = (300, 300)
OUTPUT_LAYOUT=7

with open(os.path.join('models/ssd_mobilenet/labels.json')) as json_data:
    CLASS_NAMES = json.load(json_data)


def _preprocess_trt(img, shape=(300, 300)):
    """Preprocess an image before TRT SSD inferencing."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = (2.0/255.0) * img - 1.0
    return img


class Detector():
    """Class ssd"""

    def _load_plugins(self):
        ctypes.CDLL("models/ssd_mobilenet/libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        TRTbin = 'models/ssd_mobilenet/TRT_ssd_mobilenet_v2_coco.bin'
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()

    @timeit
    def __init__(self):
        self.colors = np.random.uniform(0, 255, size=(100, 3))
        self.input_shape = INPUT_HW
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()

    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs


    @timeit
    def prediction(self, img):
        img_resized = _preprocess_trt(img, self.input_shape)
        np.copyto(self.host_inputs[0], img_resized.ravel())

        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        output = self.host_outputs[0]
        return output


    @timeit
    def filter_prediction(self, output, img, conf_th=0.3, conf_class=[]):
        img_h, img_w, _ = img.shape
        boxes, confs, clss = [], [], []
        for prefix in range(0, len(output), OUTPUT_LAYOUT):
            conf = float(output[prefix+2])
            if conf < conf_th:
                continue
            x1 = int(output[prefix+3] * img_w)
            y1 = int(output[prefix+4] * img_h)
            x2 = int(output[prefix+5] * img_w)
            y2 = int(output[prefix+6] * img_h)
            cls = int(output[prefix+1])
            if len(conf_class) > 0 and cls not in conf_class:
                continue
            boxes.append((x1, y1, x2, y2))
            confs.append(conf)
            clss.append(cls)
        return boxes, confs, clss

    def draw_boxes(self, image, boxes, confs, clss):
        for (box, cf, cls) in zip(boxes, confs, clss):
            x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
            color = self.colors[cls]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            txt = '{} {:.2f}'.format(CLASS_NAMES[str(cls)], cf)
            image = draw_boxed_text(image, txt, txt_loc, color)
        return image#[..., ::-1]


if __name__ == "__main__":
    image = cv2.imread("./imgs/image.jpeg")

    detector = Detector()
    output = detector.prediction(image)
    boxes, confs, clss = detector.filter_prediction(output, image, conf_th=0.3)
    print([(CLASS_NAMES[str(c)], prob) for (c, prob) in zip(clss, confs)])
    image = detector.draw_boxes(image, boxes, confs, clss)
    cv2.imwrite("./imgs/outputcv.jpg", image)
