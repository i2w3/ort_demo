#ifndef __REC_H__
#define __REC_H__

#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include "config.h"
#include "utils.h"


class TextRecognizer {
    public:
        TextRecognizer(const Config &config);
        ~TextRecognizer();

        TextResults getTexts(std::vector<float> &inputData);
        std::array<int64_t, 4> rec_image_shape;
        std::vector<std::string> dict;

    private:
        Ort::Session *session;
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();
        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "TextRecognizer");

        std::vector<Ort::AllocatedStringPtr> inputNamesOwned;
        std::vector<Ort::AllocatedStringPtr> outputNamesOwned;
        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;

        int dict_blank;
        int dict_offset;
        float text_thresh;
        std::vector<std::string> getDict(const std::string &dict_path);
        Contents ctcDecode(float* outputData, int batch_idx, int seq_len, int num_classes);
};

#endif // __REC_H__