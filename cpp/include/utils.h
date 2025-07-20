#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdexcept>

enum class AngleType {
    ANGLE_0 = 0,
    ANGLE_180 = 1,
};

struct Angle {
    AngleType index;
    float score;

    Angle(AngleType idx, float sc) : index(idx), score(sc) {}
};

typedef std::vector<Angle> AngleResults;

inline AngleType indexToAngleType(int index) {
    switch (index) {
        case 0: return AngleType::ANGLE_0;
        case 1: return AngleType::ANGLE_180;
        default: throw std::invalid_argument("Invalid index for AngleType");
    }
}

inline int angleTypeToIndex(AngleType angleType) {
    return static_cast<int>(angleType);
}

#endif // __UTILS_H__