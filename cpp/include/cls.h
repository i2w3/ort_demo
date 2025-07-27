#ifndef __CLS_H__
#define __CLS_H__

#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include "config.h"
#include "utils.h"


class AngleClassifier {
    public:
        AngleClassifier(const Config &config);
        ~AngleClassifier();

        AngleResults getAngles(std::vector<float> &inputData);
        std::array<int64_t, 4> cls_image_shape;

    private:
        Ort::Session *session;
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();
        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "AngleClassifier");

        std::vector<Ort::AllocatedStringPtr> inputNamesOwned;
        std::vector<Ort::AllocatedStringPtr> outputNamesOwned;
        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;
};

#endif // __CLS_H__