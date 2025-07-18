import cv2
from paddleocr import PPOCR, PPOCRv4


if __name__ == "__main__":
    ppocr = PPOCR(PPOCRv4())
    
    # input an image path
    while(True):
        image_path = input("Please input the image path: ")
        image = cv2.imread(image_path)
        if image is None:
            print("Image not found or could not be read. Please try again.")
            continue
        ppocr(image)