#include "det.h"
#include <numeric>

BoundingBoxDetector::BoundingBoxDetector(const Config &config)
    : mask_thresh(config.mask_thresh), box_thresh(config.box_thresh), unclip_ratio(config.unclip_ratio), short_side_thresh(config.short_side_thresh) {
    session = new Ort::Session(env, config.det_path.c_str(), sessionOptions);

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
    std::cout << "BoundingBoxDetector initialized with model: " << config.det_path << std::endl;
}


std::vector<float> BoundingBoxDetector::detect(std::vector<float> &inputData, const std::array<int64_t, 4> &det_image_shape) {
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, 
                                                       inputData.data(), inputData.size(), 
                                                       det_image_shape.data(), det_image_shape.size());

    auto outputTensor = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                                    inputNames.size(), outputNames.data(), outputNames.size());
    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
    float *floatArray = outputTensor.front().GetTensorMutableData<float>();
    std::vector<float> outputData(floatArray, floatArray + outputCount);
    return outputData;
}

DetResults BoundingBoxDetector::decode(std::vector<float> &outputData, const std::array<int64_t, 4> &det_image_shape) {
    DetResults results;

    // 安全类型转换
    int height = static_cast<int>(det_image_shape[2]);
    int width = static_cast<int>(det_image_shape[3]);
    
    // outputData(h*w) -> bitmap(h, w) - 零拷贝转换
    cv::Mat bitmap(height, width, CV_32FC1, outputData.data());

    // 阈值过滤
    cv::Mat maskMat;
    cv::threshold(bitmap, maskMat, mask_thresh, 255, cv::THRESH_BINARY);
    maskMat.convertTo(maskMat, CV_8UC1);
    
    // 保存 mask 图片
    // cv::imwrite("mask.png", maskMat);
        
    // 轮廓检测
    std::vector<cv::Mat> contours;
    cv::findContours(maskMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for(size_t i = 0; i < contours.size(); i++) {
        cv::Point2f boxPoints[4];
        // 获得 contour 的最小边界框
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        rect.points(boxPoints);

        float min_side_length = std::min({cv::norm(boxPoints[0] - boxPoints[1]),
                                          cv::norm(boxPoints[1] - boxPoints[2]),
                                          cv::norm(boxPoints[2] - boxPoints[3]),
                                          cv::norm(boxPoints[3] - boxPoints[0])});
        if (min_side_length < short_side_thresh) {
            continue; // 跳过短边框
        }
        
        // 将 Point2f 转换为 Point 向量用于 box_score_fast
        std::vector<cv::Point> box;
        for(int j = 0; j < 4; j++) {
            box.push_back(cv::Point(static_cast<int>(boxPoints[j].x), static_cast<int>(boxPoints[j].y)));
        }

        // 计算置信度分数
        float score = box_score_fast(bitmap, box);
        if (score < box_thresh) {
            continue; // 跳过低分框
        }

        // 使用unclip_box函数进行基于轮廓的膨胀
        cv::Mat boxMat = unclip_box(boxPoints);
        
        // 添加到结果
        results.emplace_back(boxMat, score);
    }

    return results;
}

BoundingBoxDetector::~BoundingBoxDetector() {
    // Destructor implementation (if needed)
}

cv::Mat BoundingBoxDetector::unclip_box(const cv::Point2f box_points[4]) {
    // 将输入的C风格数组转换为std::vector，方便使用OpenCV函数
    std::vector<cv::Point2f> contour(box_points, box_points + 4);

    // --- 1. 计算扩展距离 delta ---
    float area = cv::contourArea(contour);
    float perimeter = cv::arcLength(contour, true);
    float delta = area * this->unclip_ratio / perimeter;

    // --- 2. 将所有边向外平移 delta 距离 ---
    std::vector<std::pair<cv::Point2f, cv::Point2f>> offset_lines;
    for (size_t i = 0; i < 4; ++i) {
        // 定义一条边 (p2 -> p1)
        cv::Point2f p1 = contour[i];
        cv::Point2f p2 = contour[(i + 3) % 4]; // (i-1+4)%4 的等价写法

        // 计算边的向量
        cv::Point2f vec = p1 - p2;
        
        // 计算指向外部的单位法向量
        // (vec.y, -vec.x) 是将向量 (vec.x, vec.y) 顺时针旋转90度得到的法线
        cv::Point2f normal(vec.y, -vec.x);
        double norm_val = cv::norm(normal);
        if (norm_val < 1e-6) continue; // 避免除以零
        cv::Point2f unit_normal = normal / norm_val;
        
        // 计算总的偏移向量
        cv::Point2f offset_vec = unit_normal * delta;
        
        // 将边的两个端点都加上偏移向量，得到平移后的线段
        offset_lines.push_back({p1 + offset_vec, p2 + offset_vec});
    }

    // --- 3. 计算相邻平移线段的交点，形成新顶点 ---
    std::vector<cv::Point2f> expanded_contour;
    for (size_t i = 0; i < offset_lines.size(); ++i) {
        // 取两条相邻的偏移线段
        const auto& line1 = offset_lines[i];
        const auto& line2 = offset_lines[(i + 1) % offset_lines.size()];

        // 计算交点
        cv::Point2f intersection_pt = _get_line_intersection(
            line1.first, line1.second, line2.first, line2.second
        );
        expanded_contour.push_back(intersection_pt);
    }

    // 将结果转换为 cv::Mat 并返回
    return cv::Mat(expanded_contour, true).clone();
}

cv::Point2f BoundingBoxDetector::_get_line_intersection(const cv::Point2f& a, const cv::Point2f& b, 
                                                        const cv::Point2f& c, const cv::Point2f& d) {
    // 每条线表示为 P = P1 + t * (P2-P1)
    cv::Point2f v1 = b - a; // 线段1的方向向量
    cv::Point2f v2 = d - c; // 线段2的方向向量

    // 使用2D向量叉积计算分母
    float denom = v1.x * v2.y - v1.y * v2.x;

    // 如果分母接近0，则认为两线平行或共线
    if (std::abs(denom) < 1e-6) {
        // 采用启发式策略: 取两个“近”端点的中点作为近似交点
        // 这里我们取线1的起点a和线2的终点d的中点，这对应于两条原始边相交的那个顶点
        return (a + d) * 0.5f;
    }

    // 使用Cramer法则求解参数 t
    // t = ((c - a) x v2) / (v1 x v2)
    cv::Point2f ca = c - a;
    float t = (ca.x * v2.y - ca.y * v2.x) / denom;
    
    // 计算交点
    return a + t * v1;
}

float BoundingBoxDetector::box_score_fast(const cv::Mat &bitmap, const std::vector<cv::Point> &box) {
    int h = bitmap.rows;
    int w = bitmap.cols;
    
    // 复制 box 以避免修改原始数据
    std::vector<cv::Point> boxCopy = box;
    
    // 找到边界框的最小和最大坐标
    int xmin = w - 1, xmax = 0, ymin = h - 1, ymax = 0;
    
    for(const auto& point : boxCopy) {
        xmin = std::min(xmin, point.x);
        xmax = std::max(xmax, point.x);
        ymin = std::min(ymin, point.y);
        ymax = std::max(ymax, point.y);
    }
    
    // 裁剪到图像边界
    xmin = std::max(0, std::min(xmin, w - 1));
    xmax = std::max(0, std::min(xmax, w - 1));
    ymin = std::max(0, std::min(ymin, h - 1));
    ymax = std::max(0, std::min(ymax, h - 1));
    
    // 检查边界是否有效
    if (xmax <= xmin || ymax <= ymin) {
        return 0.0f;
    }
    
    // 创建 mask
    int maskHeight = ymax - ymin + 1;
    int maskWidth = xmax - xmin + 1;
    cv::Mat mask = cv::Mat::zeros(maskHeight, maskWidth, CV_8UC1);
    
    // 调整 box 坐标到相对于裁剪区域的坐标
    for(auto& point : boxCopy) {
        point.x -= xmin;
        point.y -= ymin;
    }
    
    // 填充多边形
    std::vector<std::vector<cv::Point>> contours = {boxCopy};
    cv::fillPoly(mask, contours, cv::Scalar(1));
    
    // 提取对应的 bitmap 区域
    cv::Rect roi(xmin, ymin, maskWidth, maskHeight);
    cv::Mat bitmapRoi = bitmap(roi);
    
    // 计算平均值
    cv::Scalar meanVal = cv::mean(bitmapRoi, mask);
    
    return static_cast<float>(meanVal[0]);
}