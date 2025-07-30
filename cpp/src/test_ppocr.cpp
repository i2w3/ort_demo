#include "ppocr.h"

void ConsoleDrawResults(cv::Mat& image, const PPOCRResults& results) {
    std::cout << "\n=== OCR Results ===" << std::endl;
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        
        // 1. 绘制检测框和序号
        if (!result.boundingBox.boxPoints.empty()) {
            std::vector<cv::Point> points;
            for (int j = 0; j < 4; ++j) {
                points.push_back(cv::Point(
                    static_cast<int>(result.boundingBox.boxPoints.at<float>(j, 0)),
                    static_cast<int>(result.boundingBox.boxPoints.at<float>(j, 1))
                ));
            }
            
            // 绘制检测框
            cv::polylines(image, points, true, cv::Scalar(0, 255, 0), 2);
            
            // 绘制序号
            cv::circle(image, points[0], 15, cv::Scalar(0, 0, 255), -1);
            cv::putText(image, std::to_string(i + 1), 
                       cv::Point(points[0].x - 5, points[0].y + 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }
        
        // 2. 在控制台输出中文文本
        if (!result.contents.empty()) {
            std::cout << "Text " << (i + 1) << ": ";
            for (const auto& content : result.contents) {
                std::cout << content.content;
            }
            std::cout << " (confidence: ";
            for (const auto& content : result.contents) {
                std::cout << content.score << " ";
            }
            std::cout << ")" << std::endl;
        }
    }
}

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
        bool enable_det = true; // Enable detection or not
        bool enable_cls = true; // Enable classification or not
        bool enable_rec = true; // Enable recognition or not

        PPOCRResults results = ppocr.infer(image, enable_det, enable_cls, enable_rec);
        std::cout << "Detection results: " << results.size() << " text boxes detected." << std::endl;

        // Display results
        ConsoleDrawResults(image, results);
        // cv::imwrite("output.png", image);
    }
}