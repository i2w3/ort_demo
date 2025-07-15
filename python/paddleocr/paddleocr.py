import cv2
import numpy as np
from typing import Tuple

import onnxruntime as ort

from .utils import CustomLogger, PPOCRv4, PPOCRv5


class DetDecoder:
    def __init__(self):
        self.mask_thresh = 0.3
        self.box_thresh  = 0.5

    def __call__(self,):
        pass


class PPOCR:
    def __init__(self, ppocrconfig:str):
        self.config = PPOCRv5() if ppocrconfig == "PPOCRv5" else PPOCRv4()

        self.logger = CustomLogger(logger_level=self.config.logger_level, file_logging=self.config.file_logging, logger_name="PPOCR") if self.config.enable_logging else None

        self.session_det = ort.InferenceSession(str(self.config.det_path), providers=self.config.providers)
        self.session_rec = ort.InferenceSession(str(self.config.rec_path), providers=self.config.providers)
        self.session_cls = ort.InferenceSession(str(self.config.cls_path), providers=self.config.providers)
        self.dict_list = self.load_dict(self.config.dict_path)
        
        self.det_input_name = self.session_det.get_inputs()[0].name
        self.rec_input_name = self.session_rec.get_inputs()[0].name
        self.cls_input_name = self.session_cls.get_inputs()[0].name

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))

        self.det_mask_thresh = 0.3
        self.det_box_thresh  = 0.5
        self.det_unclip_ratio = 1.6

    def unclip_opencv(self, box: np.ndarray) -> np.ndarray:
        """
        使用纯 OpenCV 和 Numpy 实现多边形的扩展。
        """
        # 1. 计算扩展距离
        # cv2.contourArea 和 cv2.arcLength 需要 int32 或 float32 类型的轮廓
        contour = box.astype(np.float32)
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)

        # 避免除以零
        if length == 0:
            return box

        distance = area * self.det_unclip_ratio / length

        # 2. 找到多边形的几何中心
        M = cv2.moments(contour)
        if M["m00"] == 0: # 避免除以零
            return box
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        center = np.array([cx, cy])

        # 3. 将每个顶点沿“中心到顶点”的方向向外移动
        expanded_box = []
        for point in box:
            # 计算从中心到顶点的向量
            vec = point - center
            # 计算向量长度 (范数)
            vec_norm = np.linalg.norm(vec)

            # 避免除以零
            if vec_norm == 0:
                expanded_box.append(point)
                continue

            # 单位化向量
            unit_vec = vec / vec_norm
            # 计算新的顶点位置
            new_point = point + unit_vec * distance
            expanded_box.append(new_point)
            
        return np.array(expanded_box)

    def load_dict(self, dict_path) -> list[str]:
        with open(dict_path, 'r', encoding='utf-8') as f:
            dict_list = [line.strip() for line in f.readlines()]
        if self.logger:
            self.logger.info(f"Loaded dictionary with {len(dict_list)} entries from {dict_path.resolve()}")
        return dict_list
    
    def recognize(self, image):
        # Implement recognition logic using self.session_rec
        pass

    def classify(self, image):
        # Implement classification logic using self.session_cls
        pass


    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()

        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def infer(self, image: np.ndarray):
        # det -> cls -> rec
        if image is None:
            self.logger.warning("Image not found or unable to read.")
            return None

        src_img = image.copy()
        
        # 调用新的 preprocess，并接收返回的 ratio
        input_data, ratio, _ = self.preprocess(image)

        if self.config.enable_det:
            # 将 ratio 传递给 detect 函数
            boxes, scores = self.detect(input_data, ratio)
            for box in boxes:
                points = box.astype(np.int32)
                cv2.polylines(src_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            if self.config.logger_level == 'debug':
                cv2.imshow('img', src_img)
                cv2.waitKey(0)

# 修改 detect 函数
    def detect(self, image: np.ndarray, ratio: float): # 接收 ratio 作为参数
        boxes = []
        scores = []

        det_result = self.session_det.run(None, {self.det_input_name: image})
        pred = det_result[0][0, 0, ...]
        bitmap = (pred > self.det_mask_thresh).astype(np.uint8)
        
        # 注意：这里的 height 和 width 是填充后图像的尺寸
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            box, short_size = self.get_mini_boxes(contour)
            if short_size < 5:
                continue
            
            box = np.array(box)

            score = self.box_score_fast(pred, box)
            if score < self.det_box_thresh:
                continue

            expanded_box = self.unclip_opencv(box)

            _, short_size = self.get_mini_boxes(expanded_box.astype(np.float32))
            if short_size < 5:
                continue
            
            box = expanded_box

            # ✅ 关键修改：使用统一的比例进行坐标还原
            # 因为预处理是等比例缩放的，所以还原时也用同一个比例
            # 直接除以 ratio 等价于乘以 (1.0 / ratio)
            box /= ratio
            
            boxes.append(box)
            scores.append(score)

        return boxes, scores


    # def preprocess(self, image):
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     # Resize to the nearest multiple of 32
    #     original_h, original_w = image.shape[:2]
    #     new_h = (original_h // 32) * 32
    #     new_w = (original_w // 32) * 32
    #     image = cv2.resize(image, (new_w, new_h))

    #     # Normalize the image to input_data
    #     input_data = image.astype(np.float32) / 255.0
    #     input_data = (input_data - self.mean) / self.std
    #     input_data = np.transpose(input_data, (2, 0, 1))
    #     input_data = np.expand_dims(input_data, axis=0)

    #     return input_data

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple]:
        """
        一个更健壮的预处理函数，通过填充保持图像的原始宽高比。
        返回:
            - input_data: 处理后用于模型的输入图像
            - ratio: 原始图像到缩放后图像的统一缩放比例
            - padded_size: 填充后的最终图像尺寸 (w, h)
        """
        # 1. 获取原始尺寸
        original_h, original_w = image.shape[:2]

        # 2. 计算一个统一的缩放比例，并确定缩放后的尺寸
        #    这里以长边不超过960为例，这是一个常用尺寸，可以防止图像过大
        max_side = 960
        ratio = min(max_side / original_w, max_side / original_h)
        resize_w, resize_h = int(original_w * ratio), int(original_h * ratio)

        # 3. 对缩放后的图像进行 resize
        resized_image = cv2.resize(image, (resize_w, resize_h))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # 4. 计算填充后的目标尺寸（向上取整到32的倍数）
        padded_w = ((resize_w + 31) // 32) * 32
        padded_h = ((resize_h + 31) // 32) * 32

        # 5. 创建一个灰色的背景（画布），并将缩放后的图像粘贴上去
        #    使用 128 作为填充值是常见的做法
        padded_image = np.full((padded_h, padded_w, 3), 128, dtype=np.uint8)
        padded_image[0:resize_h, 0:resize_w, :] = resized_image

        # 6. 归一化和维度变换
        input_data = padded_image.astype(np.float32) / 255.0
        input_data = (input_data - self.mean) / self.std
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        return input_data, ratio, (padded_w, padded_h)
