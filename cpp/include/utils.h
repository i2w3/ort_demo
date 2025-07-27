#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdexcept>
#include <opencv2/opencv.hpp>

enum class AngleType {
    ANGLE_0 = 0,
    ANGLE_180 = 1,
};

struct Angle {
    AngleType index;
    float score;

    Angle(AngleType idx, float sc) : index(idx), score(sc) {}
};

struct BoundingBox {
    cv::Mat boxPoints; // 4x2 的矩阵，每一行代表一个顶点的 (x, y) 坐标，类型为 CV_32F (浮点型)
    float score;

    BoundingBox(const cv::Mat& points, float sc) : boxPoints(points), score(sc) {}
};

typedef std::vector<Angle> AngleResults;
typedef std::vector<BoundingBox> DetResults;

inline AngleType parseAngleType(const int index) {
    switch (index) {
        case 0: return AngleType::ANGLE_0;
        case 1: return AngleType::ANGLE_180;
        default: throw std::invalid_argument("Invalid index for AngleType");
    }
}

inline int parseIndex(const AngleType type) {
    switch (type) {
        case AngleType::ANGLE_0: return 0;
        case AngleType::ANGLE_180: return 1;
        default: throw std::invalid_argument("Invalid AngleType");
    }
}

#endif // __UTILS_H__