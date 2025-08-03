#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <array>
#include <algorithm>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> STD = {0.229f, 0.224f, 0.225f};

class MultiLabelClassification {
    public:
        MultiLabelClassification(const std::string &model_path, const std::string &label_file);
        ~MultiLabelClassification();

        std::vector<float> classify(const cv::Mat &image) {
            std::vector<float> resultss;
            std::vector<float> input_data = this->preprocessImage(image);
            auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            // auto memoryInfo = Ort::MemoryInfo::CreateCuda(OrtArenaAllocator, 0);
            auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, input_data.data(), input_data.size(), 
                                                               this->input_shape.data(), this->input_shape.size());
            auto outputTensor = session->Run(Ort::RunOptions{nullptr}, 
                                             this->inputNames.data(), &inputTensor, this->inputNames.size(), 
                                             this->outputNames.data(), this->outputNames.size());
            auto outputShape  = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
            float* outputData = outputTensor[0].GetTensorMutableData<float>();
            int num_classes = static_cast<int>(outputShape[1]);
            std::vector<float> results(outputData, outputData + num_classes);

            // 找到最大的值和对应的索引
            // int max_index = std::max_element(results.begin(), results.end()) - results.begin();
            // float max_prob = results[max_index];
            // std::cout << "Classification results:" << std::endl;
            // std::cout << "Top prediction: ";
            // if (this->label_map.find(max_index) != this->label_map.end()) {
            //     std::cout << this->label_map[max_index] << " (" << max_prob << ")" << std::endl;
            // } else {
            //     std::cout << "Class " << max_index << " (" << max_prob << ")" << std::endl;
            // }

            // 找到所有大于阈值的预测
            std::vector<std::pair<int, float>> predicted_classes;
            for (int i = 0; i < num_classes; i++) {
                if (results[i] >= threshold) {
                    predicted_classes.push_back({i, results[i]});
                }
            }
            std::sort(predicted_classes.begin(), predicted_classes.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            if (predicted_classes.empty()) {
                std::cout << "No classes above threshold " << threshold << std::endl;
                // 显示最高概率的类别
                int max_index = std::max_element(results.begin(), results.end()) - results.begin();
                float max_prob = results[max_index];
                std::cout << "Highest probability: ";
                if (this->label_map.find(max_index) != this->label_map.end()) {
                    std::cout << this->label_map[max_index] << " (" << max_prob << ")" << std::endl;
                } else {
                    std::cout << "Class " << max_index << " (" << max_prob << ")" << std::endl;
                }
            } else {
                std::cout << "Predicted classes:" << std::endl;
                for (const auto& pred : predicted_classes) {
                    int class_idx = pred.first;
                    float prob = pred.second;
                    std::cout << "  ";
                    if (this->label_map.find(class_idx) != this->label_map.end()) {
                        std::cout << this->label_map[class_idx] << ": " << prob << std::endl;
                    } else {
                        std::cout << "Class " << class_idx << ": " << prob << std::endl;
                    }
                }
            }
            return resultss;
        };

    private:
        Ort::Session *session;
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();
        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "TextRecognizer");

        std::vector<Ort::AllocatedStringPtr> inputNamesOwned;
        std::vector<Ort::AllocatedStringPtr> outputNamesOwned;
        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;

        const std::array<int64_t, 4> input_shape = {1, 3, 224, 224};
        const float threshold = 0.25f;
        std::unordered_map<int, std::string> label_map;
        std::unordered_map<int, std::string> get_label_map(const std::string &label_file) {
            std::unordered_map<int, std::string> label_map;
            std::ifstream file(label_file);
            if (!file.is_open()) {
                std::cerr << "Error opening label file: " << label_file << std::endl;
                return label_map;
            }
            
            std::string line;
            int index = 0;
            while (std::getline(file, line)) {
                // 移除行尾可能的空白字符
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (!line.empty()) {
                    label_map[index] = line;
                    index++;
                }
            }
            
            std::cout << "Loaded " << label_map.size() << " labels from " << label_file << std::endl;
            return label_map;
        };

        std::vector<float> preprocessImage(const cv::Mat &image) {
            const int target_size = 224;
            int original_h = image.rows;
            int original_w = image.cols;

            cv::Mat image_copy;
            cv::cvtColor(image, image_copy, cv::COLOR_BGR2RGB);

            float scale = std::min(static_cast<float>(target_size) / original_h, 
                                static_cast<float>(target_size) / original_w);

            int new_h = static_cast<int>(original_h * scale);
            int new_w = static_cast<int>(original_w * scale);
            cv::resize(image_copy, image_copy, cv::Size(new_w, new_h));

            // 创建目标大小的图像并填充为0（黑色背景）
            cv::Mat padded_image = cv::Mat::zeros(target_size, target_size, CV_8UC3);
            
            // 将调整大小的图像放置在左上角（起始位置为0,0）
            cv::Rect roi(0, 0, new_w, new_h);
            image_copy.copyTo(padded_image(roi));

            cv::Mat float_image;
            padded_image.convertTo(float_image, CV_32F, 1.0 / 255.0);
            
            std::vector<float> input_data(1 * 3 * target_size * target_size);
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < target_size; h++) {
                    for (int w = 0; w < target_size; w++) {
                        cv::Vec3f pixel = float_image.at<cv::Vec3f>(h, w);
                        float value = pixel[c]; // RGB 通道

                        // 应用均值和标准差归一化
                        value = (value - MEAN[c]) / STD[c];

                        // 存储为 CHW 格式 (batch_size=1)
                        int index = c * target_size * target_size + h * target_size + w;
                        input_data[index] = value;
                    }
                }
            }
            return input_data;
        };
};

MultiLabelClassification::MultiLabelClassification(const std::string &model_path, const std::string &label_file) {
    // 检查可用的执行提供程序
    std::vector<std::string> providers = Ort::GetAvailableProviders();
    std::cout << "Available providers: ";
    for (const auto& provider : providers) {
        std::cout << provider << " ";
    }
    std::cout << std::endl;


    try {
        // 添加CUDA执行提供程序
        this->sessionOptions.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
        // 配置CUDA选项
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 1;  // 扩展GPU内存池
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;

        this->sessionOptions.SetIntraOpNumThreads(1);
        this->sessionOptions.SetInterOpNumThreads(1);
        this->sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        this->sessionOptions.EnableMemPattern();
        this->sessionOptions.EnableCpuMemArena();
        std::cout << "CUDA provider added successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed to add CUDA provider, falling back to CPU: " << e.what() << std::endl;
    }

    this->session = new Ort::Session(this->env, model_path.c_str(), this->sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputs = this->session->GetInputCount();
    this->inputNamesOwned.reserve(numInputs);
    this->inputNames.reserve(numInputs);
    for (size_t i = 0; i < numInputs; i++) {
        auto inputName = this->session->GetInputNameAllocated(i, allocator);
        this->inputNamesOwned.push_back(std::move(inputName));
        this->inputNames.push_back(this->inputNamesOwned.back().get());
    }

    size_t numOutputs = this->session->GetOutputCount();
    this->outputNamesOwned.reserve(numOutputs);
    this->outputNames.reserve(numOutputs);
    for (size_t i = 0; i < numOutputs; i++) {
        auto outputName = this->session->GetOutputNameAllocated(i, allocator);
        this->outputNamesOwned.push_back(std::move(outputName));
        this->outputNames.push_back(this->outputNamesOwned.back().get());
    }

    this->label_map = this->get_label_map(label_file);

    std::cout << "MultiLabelClassification initialized with model: " << model_path << std::endl;
}

MultiLabelClassification::~MultiLabelClassification() {
    delete this->session;
    std::cout << "MultiLabelClassification session destroyed." << std::endl;
}

int main(){
    std::string model_path = "/home/tuf/code/ort_demo/models/resnet18.onnx";
    std::string label_file = "/home/tuf/code/ort_demo/models/resnet18.txt";
    MultiLabelClassification classifier(model_path, label_file);

    // Load an image using OpenCV
    cv::Mat image = cv::imread("/home/tuf/code/ort_demo/images/10.jpg");
    if (image.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return EXIT_FAILURE;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        std::vector<float> results = classifier.classify(image);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Average classification time: " << duration.count() / 100 << " ms" << std::endl;
    // std::vector<float> results = classifier.classify(image);

    return EXIT_SUCCESS;
}