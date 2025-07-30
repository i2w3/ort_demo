#include "ppocr.h"


PPOCR::PPOCR(const Config &config){
    auto start = std::chrono::high_resolution_clock::now();

    #ifdef USE_PARALLEL
    auto detFuture = std::async(std::launch::async, [&config]() {
        return std::make_unique<BoundingBoxDetector>(config);
    });
    
    auto clsFuture = std::async(std::launch::async, [&config]() {
        return std::make_unique<AngleClassifier>(config);
    });
    auto recFuture = std::async(std::launch::async, [&config]() {
        return std::make_unique<TextRecognizer>(config);
    });
    this->det_model = detFuture.get();
    this->cls_model = clsFuture.get();
    this->rec_model = recFuture.get();
    #else
    this->det_model = std::make_unique<BoundingBoxDetector>(config);
    this->cls_model = std::make_unique<AngleClassifier>(config);
    this->rec_model = std::make_unique<TextRecognizer>(config);
    #endif

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "--- Model initialization completed!(Elapsed time: " << elapsed.count() << "s)"<< std::endl;
}

PPOCR::~PPOCR() {
    // Unique pointers will automatically clean up resources
    std::cout << "--- PPOCR destructor called, resources cleaned up." << std::endl;
}

PPOCRResults PPOCR::infer(const cv::Mat &image, bool &enable_det, bool &enable_cls, bool &enable_rec) {
    PPOCRResults results;
    if (image.empty()) {
        std::cerr << "--- Error: Input image is empty." << std::endl;
        return results;
    }
    if (!enable_rec) {
        std::cerr << "--- Error: Recognition is disabled." << std::endl;
        return results;
    }
    std::vector<cv::Mat> clip_images;


    DetResults det_results;
    if(enable_det) {
        // Step 1: Preprocess the image
        PreProcessedImage preprocessedImage = this->preprocess_image(image);

        // Step 2: Run the detection model
        auto det_data = this->det_model->detect(preprocessedImage.data, preprocessedImage.image_shape);
        det_results = this->det_model->decode(det_data, preprocessedImage.image_shape);
        std::cout << "--- Detection " << det_results.size() << " word blocks!" <<std::endl;

        // std::size_t idx = 0;
        for (const auto& result : det_results) {
            // 提取四个顶点
            cv::Point2f box_points[4];
            for (int i = 0; i < 4; i++) {
                box_points[i] = cv::Point2f(
                    result.boxPoints.at<float>(i, 0),
                    result.boxPoints.at<float>(i, 1)
                );
            }
            
            // 裁剪并旋转图像
            cv::Mat clip_image = this->det_model->clip_and_rotate_image(image, std::vector<cv::Point>(box_points, box_points + 4));
            clip_images.push_back(clip_image);
            // cv::imwrite("clipdet_" + std::to_string(idx++) + ".png", clip_image);

        }
    }
    else{
        clip_images.push_back(image);
    }

    AngleResults cls_results;
    if(enable_cls) {
        // Step 3: Preprocess each clip image for classification
        // TODO: Parallelize
        std::vector<std::vector<float>> clip_datas;
        std::vector<float> clip_datas_flatten;
        size_t total_size = 0;

        for (auto &clip : clip_images) {
            std::vector<float> clip_data = this->preprocess_clip(clip, this->cls_model->cls_image_shape[3]);
            clip_datas.push_back(clip_data);
            total_size += clip_data.size();
        }
        clip_datas_flatten.reserve(total_size);

        for (const auto& clip_data : clip_datas) {
            clip_datas_flatten.insert(clip_datas_flatten.end(), clip_data.begin(), clip_data.end());
        }

        // Step 4: Run the classification model
        cls_results = this->cls_model->getAngles(clip_datas_flatten);
        std::cout << "--- Classification " << cls_results.size() << " angles!" << std::endl;

        for (size_t i = 0; i < cls_results.size(); ++i) {
            if(cls_results[i].index == AngleType::ANGLE_180) {
                cv::rotate(clip_images[i], clip_images[i], cv::ROTATE_180);
            }
        }
    }
    else{
        
    }

    // Step 5: Preprocess each clip image for recognition
    std::vector<std::vector<float>> clip_datas;
    std::vector<float> clip_datas_flatten;
    size_t total_size = 0;

    for (auto &clip : clip_images) {
        std::vector<float> clip_data = this->preprocess_clip(clip, this->rec_model->rec_image_shape[3]);
        clip_datas.push_back(clip_data);
        total_size += clip_data.size();
    }
    clip_datas_flatten.reserve(total_size);

    for (const auto& clip_data : clip_datas) {
        clip_datas_flatten.insert(clip_datas_flatten.end(), clip_data.begin(), clip_data.end());
    }

    // Step 6: Run the recognition model
    auto rec_results = this->rec_model->getTexts(clip_datas_flatten);
    std::cout << "--- Recognition " << rec_results.size() << " texts!" << std::endl;

    // Step 7: Prepare the final results
    results.reserve(rec_results.size());
    
    for(size_t i = 0; i < rec_results.size(); ++i) {
        // Create appropriate bounding box based on detection enablement
        cv::Mat boxPoints = cv::Mat::zeros(4, 2, CV_32F);
        float bbox_score = 1.0f;
        if (enable_det && i < det_results.size()) {
            boxPoints = det_results[i].boxPoints.clone();  // Use actual detection results if available
            bbox_score = det_results[i].score;
        }
        BoundingBox bbox(boxPoints, bbox_score);
        
        // Create appropriate angle based on classification enablement
        Angle angle(AngleType::ANGLE_0, 1.0f);
        if (enable_cls && i < cls_results.size()) {
            angle = cls_results[i];
        }
        
        // Use recognition results directly
        results.push_back(PPOCRResult(bbox, angle, rec_results[i]));
    }
    return results;
}

PreProcessedImage PPOCR::preprocess_image(const cv::Mat &image) {
    // int c = image.channels();
    int h = image.rows;
    int w = image.cols;

    cv::Mat input;
    cv::cvtColor(image, input, cv::COLOR_BGR2RGB);


    int new_h = ((h + 31) / 32) * 32;
    int new_w = ((w + 31) / 32) * 32;

    int pad_h = new_h - h;
    int pad_w = new_w - w;

    cv::copyMakeBorder(input, input, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat float_image;
    input.convertTo(float_image, CV_32F, 1.0 / 255.0);
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


std::vector<float> PPOCR::preprocess_clip(cv::Mat &clip, const int &clip_width) {
    // int c = clip.channels();
    int h = clip.rows;
    int w = clip.cols;

    cv::Mat image;
    cv::cvtColor(clip, image, cv::COLOR_BGR2RGB);


    float ratio = w / h;
    int resized_w = int(this->clip_height * ratio);

    if (resized_w > clip_width) {
        resized_w = clip_width;
    }


    cv::resize(image, image, cv::Size(resized_w, this->clip_height));

    int pad_w = clip_width - resized_w;
    int pad_h = 0;

    cv::copyMakeBorder(image, image, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat float_image;
    image.convertTo(float_image, CV_32F, 1.0 / 255.0);
    std::vector<float> input_data(1 * 3 * this->clip_height * clip_width);

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < this->clip_height; h++) {
            for (int w = 0; w < clip_width; w++) {
                cv::Vec3f pixel = float_image.at<cv::Vec3f>(h, w);
                float value = pixel[c]; // RGB 通道
                
                // 应用均值和标准差归一化
                value = (value - MEAN[c]) / STD[c];
                
                // 存储为 CHW 格式 (batch_size=1)
                int index = c * this->clip_height * clip_width + h * clip_width + w;
                input_data[index] = value;
            }
        }
    }
    return input_data;
}