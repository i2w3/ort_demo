#include "ppocr.h"

int main(){
    Config config = PPOCRv4Config();
    PPOCR ppocr(config);
    std::string image_path;
    while(1){
        std::cout << "Please input the image file path: ";
        std::cin >> image_path;
        auto image = cv::imread(image_path);
        if (image.empty()){
            std::cout << "Failed to read image from " << image_path << ". Please try again." << std::endl;
            continue;
        }
        bool enable_det = true; // Enable detection
        bool enable_cls = true; // Enable classification

        PPOCRResults results = ppocr.infer(image, enable_det, enable_cls);
    }
}