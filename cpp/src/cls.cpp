#include "cls.h"
#include <numeric>

AngleClassifier::AngleClassifier(const Config &config)
    : cls_image_shape(config.cls_image_shape) {
    session = new Ort::Session(env, config.cls_path.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputs = session->GetInputCount();
    inputNamesOwned.reserve(numInputs);
    inputNames.reserve(numInputs);
    for (size_t i = 0; i < numInputs; i++) {
        auto inputName = session->GetInputNameAllocated(i, allocator);
        inputNamesOwned.push_back(std::move(inputName));
        inputNames.push_back(inputNamesOwned.back().get());
    }

    size_t numOutputs = session->GetOutputCount();
    outputNamesOwned.reserve(numOutputs);
    outputNames.reserve(numOutputs);
    for (size_t i = 0; i < numOutputs; i++) {
        auto outputName = session->GetOutputNameAllocated(i, allocator);
        outputNamesOwned.push_back(std::move(outputName));
        outputNames.push_back(outputNamesOwned.back().get());
    }
    std::cout << "AngleClassifier initialized with model: " << config.cls_path << std::endl;
}

AngleResults AngleClassifier::getAngles(std::vector<float> &inputData) {
    AngleResults results;
    
    // 计算批次大小
    int batch_size = inputData.size() / (cls_image_shape[1] * cls_image_shape[2] * cls_image_shape[3]);
    this->cls_image_shape[0] = batch_size;

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, 
                                                       inputData.data(), inputData.size(), 
                                                       cls_image_shape.data(), cls_image_shape.size());

    auto outputTensor = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                                    inputNames.size(), outputNames.data(), outputNames.size());

    // 获取输出张量的形状和数据
    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
    float *floatArray = outputTensor[0].GetTensorMutableData<float>();
    std::vector<float> outputData(floatArray, floatArray + outputCount);
    
    // 计算每个样本的输出大小（通常是类别数，如2或4）
    int classes_per_sample = outputCount / batch_size;

    // 为每个批次中的样本处理结果
    // TODO: parallelize this if needed
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        int start_idx = batch_idx * classes_per_sample;
        int end_idx = start_idx + classes_per_sample;
        
        // 找到当前样本的最大值和索引
        int maxIndex = 0;
        float maxScore = outputData[start_idx];
        
        for (int j = start_idx; j < end_idx; j++) {
            if (outputData[j] > maxScore) {
                maxScore = outputData[j];
                maxIndex = j - start_idx; // 相对于当前样本的索引
            }
        }
        results.push_back(Angle(parseAngleType(maxIndex), maxScore));
    }
    
    return results;
}

AngleClassifier::~AngleClassifier() {
    // Destructor implementation (if needed)
}