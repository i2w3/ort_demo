from paddleocr import DetDecoder, PPOCRv4
import numpy as np
import cv2


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))

def preprocess(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Padding to the nearest multiple of 32
    original_h, original_w = image.shape[:2]
    new_h = (original_h // 31) * 32
    new_w = (original_w // 31) * 32
    pad_h = new_h - original_h
    pad_w = new_w - original_w
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # Normalize the image to input_data
    input_data = image.astype(np.float32) / 255.0
    input_data = (input_data - MEAN) / STD
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    return input_data, pad_h, pad_w

if __name__ == "__main__":
    # Example usage of DetDecoder
    config = PPOCRv4()
    det_decoder = DetDecoder(config)

    image = cv2.imread("/home/tuf/code/ort_demo/images/5.jpg")
    
    tensor, pad_h, pad_w = preprocess(image)
    processed_image = cv2.cvtColor(tensor[0].transpose(1, 2, 0) * STD + MEAN, cv2.COLOR_RGB2BGR) * 255.0

    cv2.imwrite("preprocessed_image.jpg", processed_image.astype(np.uint8))
    results = det_decoder(tensor)
    
    canvas = image.copy()
    cliped_images = []
    for box, angle, score in results[0]:
        cv2.polylines(canvas, [box], True, (0, 0, 255), thickness=2)
        cv2.putText(canvas, f"{score:.2f}", tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        clip_image = det_decoder.clip_and_rotate_image(image, box)
        cliped_images.append(clip_image)
        cv2.imwrite(f"cliped_image_{len(cliped_images)}.jpg", clip_image)
    cv2.imwrite("det_results.jpg", canvas)

