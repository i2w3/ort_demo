import cv2
import numpy as np

import onnxruntime as ort

from .utils import CustomLogger, BaseConfig


class TextRecognizer:
    """

    """
    def __init__(self, config: BaseConfig):
        self.config = config
        self.image_shape = [3, 48, 320]
        self.label_list = ['0', '180']
        self.dict_list = self.load_dict(config.dict_path)

        self.blank_idx:int = 0
        self.offset:int = 1
        self.text_thresh: float = 0.5
        self.allow_repeat:bool = False

        self.session = ort.InferenceSession(str(config.rec_path), providers=config.providers)

    def __call__(self, tensor):
        ort_inputs:dict[str, np.ndarray] = {i.name: tensor for i in self.session.get_inputs()}
        rec_results: list[np.ndarray] = self.session.run(None, ort_inputs)
        text_results = []
        for i, rec_result in enumerate(rec_results[0]):
            # 去掉batch维度 -> (seq_len, num_classes)
            text = self.ctc_decode(rec_result)
            text_results.append(text)
        return text_results
    
    def load_dict(self, dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            char_list = [line.strip() for line in f]
        return char_list
    
    def ctc_decode(self, preds):
        assert preds.ndim == 2, f"Expected preds to be 2D, got {preds.ndim}D"
        indices = np.argmax(preds, axis=1)   # 每个时间步选择概率最大的类 -> (seq_len,)
        scores = np.max(preds, axis=1)

        processed_indices = []
        last_idx = self.blank_idx
        for i in range(len(indices)):
            idx = indices[i]
            score = scores[i]
            
            if score < self.text_thresh:
                continue
            
            if idx == last_idx:
                continue

            processed_indices.append(idx)
            last_idx = idx

        # 3. 移除 "blank" 字符并映射到真实字符
        #    这是 CTC 解码的第二步：[h, blank, e, l, o] -> [h, e, l, o]
        decoded_chars = []
        for idx in processed_indices:
            if idx == self.blank_idx:
                continue
            decoded_chars.append(self.dict_list[idx - self.offset])

        return ''.join(decoded_chars)