#ifndef __CONFIG_H__
#define __CONFIG_H__
#include <string>

class Config {
    public:
        bool enable_det = true;
        bool enable_cls = true;
        bool enable_rec = true;

        std::string cls_path = "";
        std::string det_path = "";
        std::string rec_path = "";
        std::string dict_path = "";

        std::array<int64_t, 4> cls_image_shape;
        std::array<int64_t, 4> rec_image_shape;

};

class PPOCRv4Config: public Config {
    public:
        PPOCRv4Config() {
            cls_path = "/home/tuf/code/ort_demo/models/ch_PP_mobile_v2.0_cls.onnx";
            det_path = "/home/tuf/code/ort_demo/models/ch_PP-OCRv4_mobile_det.onnx";
            rec_path = "/home/tuf/code/ort_demo/models/ch_PP-OCRv4_mobile_rec.onnx";
            dict_path = "/home/tuf/code/ort_demo/models/ch_PP-OCRv4_dict.txt";

            cls_image_shape = {1, 3, 48, 192};
            rec_image_shape = {1, 3, 48, 320};
        }

        // det model settings
        float mask_thresh = 0.3; // Threshold for mask score
        float box_thresh = 0.5; // Threshold for box score
        float unclip_ratio = 1.6; // Unclip ratio for box expansion
        int short_side_thresh = 3; // Minimum side length of the box

        // dictionary settings
        int dict_blank = 0;
        int dict_offset = 1;
};

#endif // __CONFIG_H__