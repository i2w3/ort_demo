from paddleocr import AngleClassifier, PPOCRv4, PPOCRv5
import numpy as np
import cv2


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))
CONST_HEIGHT = 48
MAX_WIDTH = 192

def preprocess(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # 固定高度 48，动态调整宽度到 <= 192
    ratio = w / float(h)
    resized_w = int(CONST_HEIGHT * ratio)
    if resized_w > MAX_WIDTH:
        resized_w = MAX_WIDTH
    resized_image = cv2.resize(image, (resized_w, CONST_HEIGHT))
    pad_w = MAX_WIDTH - resized_w
    pad_h = 0
    padded_image = cv2.copyMakeBorder(resized_image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # Normalize the image to input_data
    input_data = padded_image.astype(np.float32) / 255.0
    input_data = (input_data - MEAN) / STD
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    return input_data, pad_h, pad_w

if __name__ == "__main__":
    # Example usage of DetDecoder
    config = PPOCRv5()
    angle_classifier = AngleClassifier(config)

    image = cv2.imread("/home/tuf/code/ort_demo/images/2.png")
    h, w = image.shape[:2]
    
    tensor, pad_h, pad_w = preprocess(image)
    results = angle_classifier(tensor)
    if results[0]=='180':
        image = cv2.rotate(image, 1)
    cv2.imwrite("preprocessed_image.jpg", image.astype(np.uint8))