import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils import BaseConfig, CustomLogger, dataclass

from .det import DetDecoder
from .cls import AngleClassifier
from .rec import TextRecognizer

@dataclass
class OCRResult:
    box:np.ndarray
    box_score:float
    angle:str
    text:str


class PPOCR:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.logger = CustomLogger(logger_level=config.logger_level, file_logging=config.file_logging, logger_name="PPOCR") if config.enable_logging else None

        self.det_model = DetDecoder(config)
        self.cls_model = AngleClassifier(config)
        self.rec_model = TextRecognizer(config)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))
        self.const_height = 48
        self.cls_max_width = 192
        self.rec_max_width = 320
        self.font_path = "/home/tuf/code/ort_demo/assets/YAHEI CONSOLAS HYBRID.ttf"


    def __call__(self, image:np.ndarray):
        ocr_results = []
        h, w = image.shape[:2]
        canvas = image.copy()
        input_data = self.preprocess(image)
        det_results = self.det_model(input_data)

        # Clip and rotate images based on detection results
        cliped_images = []
        for box, score in det_results[0]:
            ocr_results.append(OCRResult(box=box, box_score=score, angle="", text=""))
            cv2.polylines(canvas, [box], True, (0, 0, 255), thickness=2)
            clip_image = self.det_model.clip_and_rotate_image(image, box)
            cliped_images.append(clip_image)

        tensor_data = np.zeros((len(cliped_images), 3, self.const_height, self.cls_max_width), dtype=np.float32)
        for i, image in enumerate(cliped_images):
            tensor = self.clip_preprocess(image, self.cls_max_width)
            tensor_data[i] = tensor
        cls_results = self.cls_model(tensor_data)

        for i, cls_result in enumerate(cls_results):
            ocr_results[i].angle = cls_result
            if cls_result=='180':
                cliped_images[i] = cv2.rotate(cliped_images[i], 1)

        tensor_data = np.zeros((len(cliped_images), 3, self.const_height, self.rec_max_width), dtype=np.float32)
        for i, image in enumerate(cliped_images):
            tensor = self.clip_preprocess(image, self.rec_max_width)
            tensor_data[i] = tensor
        rec_results = self.rec_model(tensor_data)

        for i, rec_result in enumerate(rec_results):
            ocr_results[i].text = rec_result
            canvas = self.draw_chinese_text(canvas, f"{score:.2f} {rec_result}", tuple(ocr_results[i].box[0]))
        cv2.imwrite("demo.png", canvas)
        self.logger.info(ocr_results)
        return ocr_results
    

    def preprocess(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Padding to the nearest multiple of 32
        original_h, original_w = image.shape[:2]
        new_h = ((original_h + 31) // 32) * 32  # 向上取整到32的倍数
        new_w = ((original_w + 31) // 32) * 32  # 向上取整到32的倍数
        pad_h = new_h - original_h
        pad_w = new_w - original_w
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # Normalize the image to input_data
        input_data = image.astype(np.float32) / 255.0
        input_data = (input_data - self.mean) / self.std
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        return input_data
    
    def clip_preprocess(self, image: np.ndarray, max_width: int, const_height: int = 48):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 固定高度 48，动态调整宽度到 <= 192
        ratio = w / float(h)
        resized_w = int(const_height * ratio)
        if resized_w > max_width:
            resized_w = max_width
        resized_image = cv2.resize(image, (resized_w, const_height))
        pad_w = max_width - resized_w
        pad_h = 0
        padded_image = cv2.copyMakeBorder(resized_image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # Normalize the image to input_data
        input_data = padded_image.astype(np.float32) / 255.0
        input_data = (input_data - self.mean) / self.std
        input_data = np.transpose(input_data, (2, 0, 1))
        return input_data
    
    def draw_chinese_text(self, image, text, position, font_size=16, color=(255, 0, 0)):
        """在图像上绘制中文文本"""
        # 转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 加载字体
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)