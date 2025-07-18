from paddleocr import TextRecognizer, PPOCRv4, PPOCRv5
import numpy as np
import cv2


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))
CONST_HEIGHT = 48
MAX_WIDTH = 320

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

    return input_data


if __name__ == "__main__":
    # Example usage of DetDecoder
    config = PPOCRv4()
    text_recognizer = TextRecognizer(config)

    image_list = ["/home/tuf/code/ort_demo/python/cliped_image_1.jpg",
                  "/home/tuf/code/ort_demo/python/cliped_image_2.jpg",
                  "/home/tuf/code/ort_demo/python/cliped_image_3.jpg",
                  "/home/tuf/code/ort_demo/python/cliped_image_4.jpg"]

    tensor_data = np.zeros((len(image_list), 3, CONST_HEIGHT, MAX_WIDTH), dtype=np.float32)

    for i, image_path in enumerate(image_list):
        image = cv2.imread(image_path)
        tensor = preprocess(image)
        tensor_data[i] = tensor

    results = text_recognizer(tensor_data)
    print(results)