import os
import cv2
import json
import ctypes
import numpy as np
import pandas as pd
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
    """Class ssd version gpu"""

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
        return np.reshape(output, (-1, OUTPUT_LAYOUT))


    @timeit
    def filter_prediction(self, output, image, conf_th=0.3, conf_class=[]):
        height, width = image.shape[:-1]
        df = pd.DataFrame(
                output,
                columns=[
                    '_', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        df = df.assign(
                x1=lambda x: (x['x1'] * width).astype(int).clip(0),
                y1=lambda x: (x['y1'] * height).astype(int).clip(0),
                x2=lambda x: (x['x2'] * width).astype(int),
                y2=lambda x: (x['y2'] * height).astype(int),
                class_name=lambda x: (
                    x['class_id'].astype(int).astype(str).replace(CLASS_NAMES)
                    ),
                )
        df['label'] = (df['class_name'] + ': ' +
                df['confidence'].astype(str).str.slice(stop=4))
        df = df[df['confidence'] > conf_th]
        if len(conf_class) > 0:
            df = df[df['class_id'].isin(conf_class)]
        return df

    def draw_boxes(self, image, df):
        for idx, box in df.iterrows():
            x_min, y_min, x_max, y_max = box['x1'], box['y1'], box['x2'], box['y2']
            color = self.colors[int(box['class_id'])]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            txt = box['label']
            image = draw_boxed_text(image, txt, txt_loc, color)
        return image

if __name__ == "__main__":
    image = cv2.imread("./imgs/image.jpeg")

    detector = Detector()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image, conf_th=0.3)
    image = detector.draw_boxes(image, df)
    print(df)
    cv2.imwrite("./imgs/outputcv.jpg", image)
