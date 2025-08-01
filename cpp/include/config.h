#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <string>

class Config {
    public:
        std::string cls_path = "";
        std::string det_path = "";
        std::string rec_path = "";
        std::string dict_path = "";

        // cls model settings
        std::array<int64_t, 4> cls_image_shape;

        // det model settings
        float mask_thresh;       // Threshold for mask score
        float box_thresh;        // Threshold for box score
        float unclip_ratio;      // Unclip ratio for box expansion
        int short_side_thresh;   // Minimum side length of the box

        // rec model settings
        std::array<int64_t, 4> rec_image_shape;
        float text_thresh;

        // dictionary settings
        int dict_blank;
        int dict_offset;
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

            mask_thresh = 0.3f;
            box_thresh = 0.5f;
            unclip_ratio = 1.6f;
            short_side_thresh = 3;

            dict_blank = 0;
            dict_offset = 1;
            text_thresh = 0.5f;
        }
};

class PPOCRv5Config: public Config {
    public:
        PPOCRv5Config() {
            cls_path = "/home/tuf/code/ort_demo/models/ch_PP_mobile_v2.0_cls.onnx";
            det_path = "/home/tuf/code/ort_demo/models/ch_PP-OCRv5_mobile_det.onnx";
            rec_path = "/home/tuf/code/ort_demo/models/ch_PP-OCRv5_mobile_rec.onnx";
            dict_path = "/home/tuf/code/ort_demo/models/ch_PP-OCRv5_dict.txt";

            cls_image_shape = {1, 3, 48, 192};
            rec_image_shape = {1, 3, 48, 320};

            mask_thresh = 0.3f;
            box_thresh = 0.5f;
            unclip_ratio = 1.6f;
            short_side_thresh = 3;

            dict_blank = 0;
            dict_offset = 1;
            text_thresh = 0.5f;
        }
};

#endif // __CONFIG_H__