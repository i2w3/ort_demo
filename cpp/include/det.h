#ifndef __DET_H__
#define __DET_H__

#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include "config.h"
#include "utils.h"


class BoundingBoxDetector {
    public:
        BoundingBoxDetector(const Config &config);
        ~BoundingBoxDetector();

        std::vector<float> detect(std::vector<float> &inputData, const std::array<int64_t, 4> &det_image_shape);
        DetResults decode(std::vector<float> &outputData, const std::array<int64_t, 4> &det_image_shape);
        cv::Mat clip_and_rotate_image(const cv::Mat &image, const std::vector<cv::Point> &box);

    private:
        const float mask_thresh;
        const float box_thresh;
        const float unclip_ratio;
        const int short_side_thresh;

        Ort::Session *session;
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();
        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "AngleClassifier");

        std::vector<Ort::AllocatedStringPtr> inputNamesOwned;
        std::vector<Ort::AllocatedStringPtr> outputNamesOwned;
        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;

        float box_score_fast(const cv::Mat &bitmap, const std::vector<cv::Point> &box);
        cv::Mat unclip_box(const cv::Point2f box_points[4]);
        cv::Point2f _get_line_intersection(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c, const cv::Point2f& d);
};

#endif // __DET_H__