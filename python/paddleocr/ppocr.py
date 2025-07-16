import cv2
import numpy as np

from .utils import BaseConfig, CustomLogger

from .det import DetDecoder
from .cls import AngleClassifier

class PPOCR:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = CustomLogger(logger_level=config.logger_level, file_logging=config.file_logging, logger_name="PPOCR") if config.enable_logging else None

        self.det_model = DetDecoder(config)
        self.cls_model = 
        self.dict_list = self.load_dict(config.dict_path)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))


    def infer(self, images:np.ndarray):
        tensor = self.preprocess(images)
        if self.logger:
            cv2.imwrite("preprocessed_image.jpg", (tensor[0].transpose(1, 2, 0) * self.std + self.mean) * 255.0)

    def preprocess(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Padding to the nearest multiple of 32
        original_h, original_w = image.shape[:2]
        new_h = (original_h // 32) * 32
        new_w = (original_w // 32) * 32
        pad_h = new_h - original_h
        pad_w = new_w - original_w
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # Normalize the image to input_data
        input_data = image.astype(np.float32) / 255.0
        input_data = (input_data - self.mean) / self.std
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        return input_data, pad_h, pad_w
    
    def load_dict(self, dict_path) -> list[str]:
        with open(dict_path, 'r', encoding='utf-8') as f:
            dict_list = [line.strip() for line in f.readlines()]
        if self.logger:
            self.logger.info(f"Loaded dictionary with {len(dict_list)} entries from {dict_path.resolve()}")
        return dict_list