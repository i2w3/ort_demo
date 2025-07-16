import cv2
import numpy as np

import onnxruntime as ort

from .utils import CustomLogger, BaseConfig


class DetDecoder:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.mask_thresh: float = 0.3 # Threshold for mask score
        self.box_thresh: float = 0.5 # Threshold for box score
        self.unclip_ratio: float = 1.6 # Unclip ratio for box expansion
        self.short_side_thresh: int = 3 # Minimum side length of the box

        self.session = ort.InferenceSession(str(config.det_path), providers=config.providers)

    def __call__(self, tensor):
        ort_inputs:dict[str, np.ndarray] = {i.name: tensor for i in self.session.get_inputs()}
        det_results: list[np.ndarray] = self.session.run(None, ort_inputs)
        box_results = []
        for i, det_result in enumerate(det_results[0]):
            boxes = []
            # 1 x H x W -> H x W
            preds = det_result[0, ...] 
            # Apply thresholding
            bitmap = (preds > self.mask_thresh).astype(np.uint8) * 255

            height, width = bitmap.shape
            contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for j, contour in enumerate(contours):
                box, min_side_length = self.get_mini_box(contour)
                if min_side_length < self.short_side_thresh:
                    continue
                # unclip_ratio
                unclip_box = self.unclip_box(box)
                
                score = self.box_score_fast(preds, box) # 注意比较的是未扩展的box

                if score < self.box_thresh:
                    continue
                boxes.append((np.int32(unclip_box), score))
            box_results.append(boxes)
        return box_results

    def get_mini_box(self, contour: np.ndarray) -> np.ndarray:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        side_lengths = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
        min_side_length = min(side_lengths)
        return np.int32(box), min_side_length
                
    def unclip_box(self, box: np.ndarray) -> np.ndarray:
        assert box.shape == (4, 2), "Box should have shape (4, 2)"

        # calculate the offset distance after unclip
        area = cv2.contourArea(box)
        length = cv2.arcLength(box, closed=True)
        if length == 0:
            return box
        distance = area * self.unclip_ratio / length

        # move each point outward by the distance
        num_points = len(box)
        offset_lines = []
        for i in range(num_points):
            p1 = box[i]
            p2 = box[(i - 1) % num_points]
            
            vec = p1 - p2
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue

            # 计算指向外部的单位法向量
            # (vec[1], -vec[0]) 是 vec 顺时针旋转90度的向量
            normal_vec = np.array([vec[1], -vec[0]])
            unit_normal = normal_vec / np.linalg.norm(normal_vec)
            
            # 计算总的偏移向量
            offset_vec = unit_normal * distance
            
            # 将边的两个端点都加上偏移向量
            offset_lines.append([p1 + offset_vec, p2 + offset_vec])

        # 3. 计算交点形成新顶点
        out_poly = []
        num_lines = len(offset_lines)
        for i in range(num_lines):
            # 取两条相邻的偏移线段
            # line1: a -> b
            # line2: c -> d
            a, b = offset_lines[i]
            c, d = offset_lines[(i + 1) % num_lines]

            # 计算交点 pt
            pt = self._get_line_intersection(a, b, c, d)
            out_poly.append(pt)
            
        return np.array(out_poly, dtype=np.float32)

    def _get_line_intersection(self, a, b, c, d):
        """计算线段 a-b 和 c-d 的交点"""
        # 每条线表示为 P = P1 + t * (P2-P1)
        v1 = b - a # 线段1的方向向量
        v2 = d - c # 线段2的方向向量

        # 计算向量叉积，用于判断是否平行
        denom = np.cross(v1, v2)
        
        # 如果叉积接近0，则认为两线平行
        if abs(denom) < 1e-6:
            # C++代码中的启发式策略: 取两端点的中点
            # 这里的 c 和 d 对应 C++ 代码中的 b 和 c
            return (b + c) * 0.5

        # 使用 Cramer 法则求解交点
        # t = (c - a) x v2 / (v1 x v2)
        t = np.cross(c - a, v2) / denom
        intersection_pt = a + t * v1
        
        return intersection_pt
    
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
    
    def clip_and_rotate_image(self, image, box):
        assert box.shape == (4, 2), "Box should have shape (4, 2)"
        # 1. 对4个点进行排序，确保顺序为：左上, 右上, 右下, 左下 (顺时针)
        #    - x+y 最小的是左上角(tl)
        #    - x+y 最大的是右下角(br)
        s = box.sum(axis=1)
        tl = box[np.argmin(s)]
        br = box[np.argmax(s)]

        #    - y-x 最小的是右上角(tr)
        #    - y-x 最大的是左下角(bl)
        diff = np.diff(box, axis=1)
        tr = box[np.argmin(diff)]
        bl = box[np.argmax(diff)]
        
        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)

        # 2. 计算目标矩形的宽度和高度
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        dst_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        dst_height = max(int(height_a), int(height_b))
        
        if dst_width == 0 or dst_height == 0:
            # 避免除以零或创建空图像的错误
            return None

        # 3. 定义目标矩形的4个角点
        dst_pts = np.array([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1]
        ], dtype=np.float32)

        # 4. 计算透视变换矩阵并应用
        transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, transform_matrix, (dst_width, dst_height))
        
        # 5. (可选) 处理竖排文本：如果校正后高度大于宽度，则顺时针旋转90度
        h, w = warped.shape[:2]
        if h > w:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped
