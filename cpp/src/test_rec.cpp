#include <string>
#include <iostream>

#include "rec.h"
#include "config.h"

const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> STD = {0.229f, 0.224f, 0.225f};

const int CONST_HEIGHT = 48;
const int MAX_WIDTH = 320;

std::vector<float> preprocess(cv::Mat &src) {
    // int c = src.channels();
    int h = src.rows;
    int w = src.cols;

    cv::Mat image;
    cv::cvtColor(src, image, cv::COLOR_BGR2RGB);


    float ratio = w / h;
    int resized_w = int(CONST_HEIGHT * ratio);

    if (resized_w > MAX_WIDTH) {
        resized_w = MAX_WIDTH;
    }


    cv::resize(image, image, cv::Size(resized_w, CONST_HEIGHT));

    int pad_w = MAX_WIDTH - resized_w;
    int pad_h = 0;

    cv::copyMakeBorder(image, image, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat float_image;
    image.convertTo(float_image, CV_32F, 1.0 / 255.0);
    std::vector<float> input_data(1 * 3 * CONST_HEIGHT * MAX_WIDTH);
        for (int c = 0; c < 3; c++) {
        for (int h = 0; h < CONST_HEIGHT; h++) {
            for (int w = 0; w < MAX_WIDTH; w++) {
                cv::Vec3f pixel = float_image.at<cv::Vec3f>(h, w);
                float value = pixel[c]; // RGB 通道
                
                // 应用均值和标准差归一化
                value = (value - MEAN[c]) / STD[c];
                
                // 存储为 CHW 格式 (batch_size=1)
                int index = c * CONST_HEIGHT * MAX_WIDTH + h * MAX_WIDTH + w;
                input_data[index] = value;
            }
        }
    }
    return input_data;
}


int main(){
    Config config = PPOCRv4Config();
    TextRecognizer model(config);
    std::string image_path;
    while(1){
        std::cout << "Please input the image file path: ";
        std::cin >> image_path;
        auto image = cv::imread(image_path);
        if (image.empty()){
            std::cout << "Failed to read image from " << image_path << ". Please try again." << std::endl;
            continue;
        }
        auto tensor = preprocess(image);
        std::cout << "Image preprocessed successfully." << std::endl;
        TextResults results = model.getTexts(tensor);
        for (auto &content : results) {
            std::cout << "Recognized text: " ;
            for (auto &text : content) {
                std::cout << text.content << " (" << text.score << ") ";
            }
            std::cout << std::endl;
        }
    }
}