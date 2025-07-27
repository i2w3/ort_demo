#include <string>
#include <iostream>

#include "det.h"
#include "config.h"

const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> STD = {0.229f, 0.224f, 0.225f};

const int IMAGE_RATE = 32;


struct PreProcessedImage {
    std::vector<float> data;
    std::array<int64_t, 4> image_shape;

    PreProcessedImage(const std::vector<float> &data, const std::array<int64_t, 4> &image_shape)
        : data(data), image_shape(image_shape) {}
};


PreProcessedImage preprocess(cv::Mat &src) {
    int c = src.channels();
    int h = src.rows;
    int w = src.cols;

    cv::Mat image;
    cv::cvtColor(src, image, cv::COLOR_BGR2RGB);


    int new_h = ((h + 31) / 32) * 32;
    int new_w = ((w + 31) / 32) * 32;

    int pad_h = new_h - h;
    int pad_w = new_w - w;

    cv::copyMakeBorder(image, image, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat float_image;
    image.convertTo(float_image, CV_32F, 1.0 / 255.0);
    std::vector<float> input_data(1 * 3 * new_h * new_w);
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < new_h; h++) {
            for (int w = 0; w < new_w; w++) {
                cv::Vec3f pixel = float_image.at<cv::Vec3f>(h, w);
                float value = pixel[c]; // RGB 通道
                
                // 应用均值和标准差归一化
                value = (value - MEAN[c]) / STD[c];
                
                // 存储为 CHW 格式 (batch_size=1)
                int index = c * new_h * new_w + h * new_w + w;
                input_data[index] = value;
            }
        }
    }
    return PreProcessedImage(input_data, {1, 3, new_h, new_w});
}

int main(){
    Config config = PPOCRv4Config();
    BoundingBoxDetector model(config);
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
        auto outputdata = model.detect(tensor.data, tensor.image_shape);
        auto results = model.decode(outputdata, tensor.image_shape);

        std::vector<cv::Mat> clip_images;
        for (const auto& result : results) {
            // 提取四个顶点
            cv::Point2f box_points[4];
            for (int i = 0; i < 4; i++) {
                box_points[i] = cv::Point2f(
                    result.boxPoints.at<float>(i, 0),
                    result.boxPoints.at<float>(i, 1)
                );
            }
            
            // 裁剪并旋转图像
            cv::Mat clip_image = model.clip_and_rotate_image(image, std::vector<cv::Point>(box_points, box_points + 4));
            if (!clip_image.empty()) {
                clip_images.push_back(clip_image);
                cv::imwrite("clip_image_" + std::to_string(&result - &results[0]) + ".jpg", clip_image);
            } else {
                std::cout << "Warning: Failed to clip image for box " << (&result - &results[0]) << std::endl;
            }
        }
        
        // 绘制检测结果
        cv::Mat display_image = image.clone();
        for (const auto& box : results) {
            // 提取四个顶点
            std::vector<cv::Point> points;
            for (int i = 0; i < 4; i++) {
                points.emplace_back(
                    static_cast<int>(box.boxPoints.at<float>(i, 0)),
                    static_cast<int>(box.boxPoints.at<float>(i, 1))
                );
            }
            
            // 绘制四边形
            for (int i = 0; i < 4; i++) {
                cv::line(display_image, points[i], points[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
            
            // 绘制置信度分数
            std::string score_text = cv::format("%.2f", box.score);
            cv::putText(display_image, score_text, points[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }
        
        // 显示结果
        cv::imshow("Detection Results", display_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
        
        // 保存结果
        std::string save_path = "detection_result.jpg";
        cv::imwrite(save_path, display_image);
        std::cout << "Result saved to: " << save_path << std::endl;
        std::cout << "Detected " << results.size() << " text regions." << std::endl;
    }
}