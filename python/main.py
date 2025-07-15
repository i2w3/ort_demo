from paddleocr import PPOCR

import cv2

image = cv2.imread("/home/tuf/code/ort_demo/images/4.jpg")
ppocr = PPOCR("PPOCRv5")
ppocr.infer(image)