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

    private:
        Ort::Session *session;
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();
        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "TextRecognizer");

        std::vector<Ort::AllocatedStringPtr> inputNamesOwned;
        std::vector<Ort::AllocatedStringPtr> outputNamesOwned;
        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;

        std::vector<std::string> dict;
};

#endif // __REC_H__