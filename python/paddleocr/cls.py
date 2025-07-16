import cv2
import numpy as np

import onnxruntime as ort

from .utils import CustomLogger, BaseConfig


class AngleClassifier:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.image_shape = [3, 48, 192]
        self.label_list = ['0', '180']

        self.session = ort.InferenceSession(str(config.cls_path), providers=config.providers)

    def __call__(self, tensor):
        ort_inputs:dict[str, np.ndarray] = {i.name: tensor for i in self.session.get_inputs()}
        cls_results: list[np.ndarray] = self.session.run(None, ort_inputs)
        angle_results = []
        for i, cls_result in enumerate(cls_results[0]):
            angle = self.label_list[np.argmax(cls_result)]
            angle_results.append(angle)
        return angle_results