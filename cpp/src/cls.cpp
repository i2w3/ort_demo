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

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, 
                                                       inputData.data(), inputData.size(), 
                                                       cls_image_shape.data(), cls_image_shape.size());

    auto outputTensor = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                                    inputNames.size(), outputNames.data(), outputNames.size());
    
    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
    float *floatArray = outputTensor.front().GetTensorMutableData<float>();
    std::vector<float> outputData(floatArray, floatArray + outputCount);

    int maxIndex = 0;
    float maxScore = 0;
    for (size_t i = 0; i < outputData.size(); i++) {
        if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = i;
        }
    }
    results.push_back(Angle(indexToAngleType(maxIndex), maxScore));
    
    return results;
}

AngleClassifier::~AngleClassifier() {
    // Destructor implementation (if needed)
}