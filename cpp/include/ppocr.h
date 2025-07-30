#ifndef __PPOCR_H__
#define __PPOCR_H__

#include <chrono>

#include "det.h"
#include "cls.h"
#include "rec.h"

#include "config.h"

#ifdef USE_PARALLEL
#include <future>
#endif

class PPOCR {
    public:
        PPOCR(const Config &config);
        ~PPOCR();

        PPOCRResults infer(const cv::Mat &image, bool &enable_det, bool &enable_cls, bool &enable_rec);
        PreProcessedImage preprocess_image(const cv::Mat &image);
        std::vector<float> preprocess_clip(cv::Mat &clip, const int &clip_width);

    private:
        const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
        const std::vector<float> STD  = {0.229f, 0.224f, 0.225f};
        const int clip_height = 48;

        std::unique_ptr<BoundingBoxDetector> det_model;
        std::unique_ptr<AngleClassifier>     cls_model;
        std::unique_ptr<TextRecognizer>      rec_model;
};


#endif // __PPOCR_H__