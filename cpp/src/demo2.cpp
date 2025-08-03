#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <array>
#include <algorithm>
#include <chrono>
#include <memory> // For std::unique_ptr

// ONNX Runtime
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

// OpenCV
#include <opencv2/opencv.hpp>

// CUDA Runtime - 只需要头文件和链接库，无需nvcc编译
#include <cuda_runtime.h>

// --- Helper for CUDA Error Checking ---
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                    \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> STD = {0.229f, 0.224f, 0.225f};

class MultiLabelClassification {
public:
    MultiLabelClassification(const std::string &model_path, const std::string &label_file);
    ~MultiLabelClassification();

    void classify(const cv::Mat &image);

private:
    // 这个函数和你原来的一模一样，在CPU上执行
    std::vector<float> preprocessImage(const cv::Mat &image);
    std::unordered_map<int, std::string> get_label_map(const std::string &label_file);

    // ONNX Runtime
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<Ort::AllocatedStringPtr> inputNamesOwned;
    std::vector<Ort::AllocatedStringPtr> outputNamesOwned;
    
    // I/O Binding 相关: 我们在对象创建时就分配好GPU内存
    Ort::IoBinding binding;
    float* d_input = nullptr;  // 指向GPU输入显存的指针
    float* d_output = nullptr; // 指向GPU输出显存的指针
    int num_classes_;

    const std::array<int64_t, 4> input_shape = {1, 3, 224, 224};
    const float threshold = 0.25f;
    std::unordered_map<int, std::string> label_map;
};

MultiLabelClassification::MultiLabelClassification(const std::string &model_path, const std::string &label_file)
    : env(ORT_LOGGING_LEVEL_WARNING, "SimpleClassifier"),
      binding(nullptr) // 先初始化为空指针
{
    // 1. 设置CUDA执行提供程序
    OrtCUDAProviderOptions cuda_options{};
    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 2. 创建Session
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), sessionOptions);

    // 3. 获取输入输出信息
    Ort::AllocatorWithDefaultOptions allocator;
    inputNamesOwned.push_back(session->GetInputNameAllocated(0, allocator));
    inputNames.push_back(inputNamesOwned.back().get());
    outputNamesOwned.push_back(session->GetOutputNameAllocated(0, allocator));
    outputNames.push_back(outputNamesOwned.back().get());

    auto output_shape_info = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    num_classes_ = static_cast<int>(output_shape_info.back());

    // 4. 【核心改动】预先分配GPU内存
    size_t input_tensor_size = 1 * 3 * 224 * 224;
    size_t output_tensor_size = 1 * num_classes_;
    CUDA_CHECK(cudaMalloc(&d_input, input_tensor_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_tensor_size * sizeof(float)));

    // 5. 【核心改动】设置I/O Binding
    
    // 错误行修改: 从 Ort::MemoryInfo::Create(...) 改为直接调用构造函数
    Ort::MemoryInfo memory_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    
    // 错误行修改: 从 Ort::IoBinding::Create(...) 改为直接调用构造函数
    binding = Ort::IoBinding{*session};
    
    std::vector<int64_t> input_dims = {1, 3, 224, 224};
    std::vector<int64_t> output_dims = {1, (int64_t)num_classes_};

    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, d_input, input_tensor_size, input_dims.data(), input_dims.size());
    auto output_tensor = Ort::Value::CreateTensor<float>(memory_info, d_output, output_tensor_size, output_dims.data(), output_dims.size());

    binding.BindInput(inputNames[0], input_tensor);
    binding.BindOutput(outputNames[0], output_tensor);

    // 6. 加载标签
    this->label_map = this->get_label_map(label_file);
    std::cout << "MultiLabelClassification initialized with I/O Binding." << std::endl;
}

MultiLabelClassification::~MultiLabelClassification() {
    // 【核心改动】释放之前分配的GPU内存
    if (d_input) CUDA_CHECK(cudaFree(d_input));
    if (d_output) CUDA_CHECK(cudaFree(d_output));
    std::cout << "MultiLabelClassification destroyed. GPU memory freed." << std::endl;
}

void MultiLabelClassification::classify(const cv::Mat &image) {
    // 步骤1: 图像预处理 (在CPU上，使用你原来的函数)
    std::vector<float> input_data = this->preprocessImage(image);

    // 步骤2: 【核心改动】将预处理好的数据从CPU拷贝到预先分配好的GPU显存中
    size_t input_tensor_size = input_data.size();
    CUDA_CHECK(cudaMemcpy(d_input, input_data.data(), input_tensor_size * sizeof(float), cudaMemcpyHostToDevice));

    // 步骤3: 【核心改动】使用I/O Binding执行推理，这里没有额外的内存拷贝开销
    session->Run(Ort::RunOptions{nullptr}, binding);

    // 步骤4: 【核心改动】将结果从GPU拷贝回CPU
    std::vector<float> results(num_classes_);
    CUDA_CHECK(cudaMemcpy(results.data(), d_output, results.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 步骤5: 后处理 (和原来一样)
    std::vector<std::pair<int, float>> predicted_classes;
    for (int i = 0; i < num_classes_; i++) {
        if (results[i] >= threshold) {
            predicted_classes.push_back({i, results[i]});
        }
    }
    std::sort(predicted_classes.begin(), predicted_classes.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

    // (为了性能测试，可以注释掉下面的打印输出)
    if (predicted_classes.empty()) {
        std::cout << "No classes above threshold " << threshold << std::endl;
    } else {
        std::cout << "Predicted classes:" << std::endl;
        for (const auto& pred : predicted_classes) {
            std::cout << "  " << this->label_map[pred.first] << ": " << pred.second << std::endl;
        }
    }
}


// --- 下面的函数和你原来的代码完全一样 ---

std::vector<float> MultiLabelClassification::preprocessImage(const cv::Mat &image) {
    const int target_size = 224;
    cv::Mat image_copy;
    cv::cvtColor(image, image_copy, cv::COLOR_BGR2RGB);

    float scale = std::min(static_cast<float>(target_size) / image_copy.rows, static_cast<float>(target_size) / image_copy.cols);
    int new_h = static_cast<int>(image_copy.rows * scale);
    int new_w = static_cast<int>(image_copy.cols * scale);
    cv::resize(image_copy, image_copy, cv::Size(new_w, new_h));

    cv::Mat padded_image = cv::Mat::zeros(target_size, target_size, CV_8UC3);
    image_copy.copyTo(padded_image(cv::Rect(0, 0, new_w, new_h)));

    cv::Mat float_image;
    padded_image.convertTo(float_image, CV_32F, 1.0 / 255.0);
    
    std::vector<float> input_data(1 * 3 * target_size * target_size);
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < target_size; h++) {
            for (int w = 0; w < target_size; w++) {
                cv::Vec3f pixel = float_image.at<cv::Vec3f>(h, w);
                float value = (pixel[c] - MEAN[c]) / STD[c];
                input_data[c * target_size * target_size + h * target_size + w] = value;
            }
        }
    }
    return input_data;
}

std::unordered_map<int, std::string> MultiLabelClassification::get_label_map(const std::string &label_file) {
    std::unordered_map<int, std::string> label_map;
    std::ifstream file(label_file);
    if (!file.is_open()) {
        std::cerr << "Error opening label file: " << label_file << std::endl;
        return label_map;
    }
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty()) {
            label_map[index++] = line;
        }
    }
    return label_map;
}


int main() {
    std::string model_path = "/home/tuf/code/ort_demo/models/resnet18.onnx";
    std::string label_file = "/home/tuf/code/ort_demo/models/resnet18.txt";
    std::string image_path = "/home/tuf/code/ort_demo/images/10.jpg";

    try {
        MultiLabelClassification classifier(model_path, label_file);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error loading image!" << std::endl;
            return EXIT_FAILURE;
        }
        
        // 第一次运行，确保所有初始化完成并显示结果
        std::cout << "--- First run ---" << std::endl;
        classifier.classify(image);
        std::cout << "-----------------" << std::endl;

        // 性能测试
        auto start = std::chrono::high_resolution_clock::now();
        const int iterations = 100;
        for (int i = 0; i < iterations; i++) {
            classifier.classify(image); // 在循环中可以注释掉classify内部的打印，避免影响计时
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "\n--- Performance Test ---" << std::endl;
        std::cout << "Average classification time: " << duration.count() / iterations << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}