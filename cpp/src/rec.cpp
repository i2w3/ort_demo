#include "rec.h"
#include <numeric>
#include <fstream>

TextRecognizer::TextRecognizer(const Config &config)
    : rec_image_shape(config.rec_image_shape), dict_blank(config.dict_blank), dict_offset(config.dict_offset), text_thresh(config.text_thresh) {
    session = new Ort::Session(env, config.rec_path.c_str(), sessionOptions);

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
    std::cout << "TextRecognizer initialized with model: " << config.rec_path << std::endl;

    this->dict = getDict(config.dict_path);
}

TextRecognizer::~TextRecognizer() {
    // Destructor implementation (if needed)
}

std::vector<std::string> TextRecognizer::getDict(const std::string &dict_path) {
    std::vector<std::string> dictionary;
    std::ifstream dictFile(dict_path);
    if (!dictFile.is_open()) {
        throw std::runtime_error("Failed to open dictionary file: " + dict_path);
    }

    std::string line;
    while (std::getline(dictFile, line)) {
        if (!line.empty()) {
            dictionary.push_back(line);
        }
    }
    dictFile.close();
    return dictionary;
}

TextResults TextRecognizer::getTexts(std::vector<float> &inputData) {
    TextResults results;

    // 计算批次大小
    int batch_size = inputData.size() / (this->rec_image_shape[1] * this->rec_image_shape[2] * this->rec_image_shape[3]);
    this->rec_image_shape[0] = batch_size;

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, 
                                                       inputData.data(), inputData.size(), 
                                                       this->rec_image_shape.data(), this->rec_image_shape.size());

    auto outputTensor = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                                    inputNames.size(), outputNames.data(), outputNames.size());

    // 获取输出张量的形状和数据
    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
    float *floatArray = outputTensor[0].GetTensorMutableData<float>();

    // 输出形状应该是 (batch_size, seq_len, num_classes)
    int64_t batch_size_out = outputShape[0];
    int64_t seq_len = outputShape[1];
    int64_t num_classes = outputShape[2];

    // 为每个批次中的样本处理结果
    for (int batch_idx = 0; batch_idx < batch_size_out; batch_idx++) {
        Contents contents = ctcDecode(floatArray, batch_idx, seq_len, num_classes);
        results.push_back(contents);
    }
    return results;
}

Contents TextRecognizer::ctcDecode(float* outputData, int batch_idx, int seq_len, int num_classes) {
    Contents contents;
    
    // 获取当前batch的指针
    float* batch_data = outputData + batch_idx * seq_len * num_classes;
    
    // 1. 对每个时间步选择概率最大的类和对应的分数
    std::vector<int> indices;
    std::vector<float> scores;
    
    for (int t = 0; t < seq_len; t++) {
        float* timestep_data = batch_data + t * num_classes;
        int max_idx = 0;
        float max_score = timestep_data[0];
        
        for (int c = 1; c < num_classes; c++) {
            if (timestep_data[c] > max_score) {
                max_score = timestep_data[c];
                max_idx = c;
            }
        }
        
        if (max_score < this->text_thresh) {
            continue; // 跳过低置信度的预测
        }
        
        indices.push_back(max_idx);
        scores.push_back(max_score);
    }
    
    // 2. CTC解码：移除重复字符和blank字符
    std::vector<int> processed_indices;
    int last_idx = this->dict_blank;
    
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        
        if (idx == last_idx) {
            continue; // 跳过重复字符
        }
        
        processed_indices.push_back(idx);
        last_idx = idx;
    }
    
    // 3. 将索引映射到实际字符
    for (int idx : processed_indices) {
        if (idx == this->dict_blank) {
            continue; // 跳过blank字符
        }
        
        int char_idx = idx - this->dict_offset;
        if (char_idx >= 0 && char_idx < static_cast<int>(this->dict.size())) {
            contents.push_back(Text{this->dict[char_idx], 1.0f});
        }
    }
    
    return contents;
}