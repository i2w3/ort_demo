from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BaseConfig:
    providers: list[str] = field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    enable_logging: bool = True
    logger_level: str = "debug"  # Options: 'debug', 'info', 'warning', 'error', 'critical'
    file_logging: bool = True

    enable_det: bool = True
    enable_cls: bool = True
    enable_rec: bool = True


@dataclass
class PPOCRv4(BaseConfig):
    """Configuration for PPOCRv4 model."""
    cls_path:  Path = Path("/home/tuf/code/ort_demo/models/ch_PP_mobile_v2.0_cls.onnx")
    det_path:  Path = Path("/home/tuf/code/ort_demo/models/ch_PP-OCRv4_mobile_det.onnx")
    rec_path:  Path = Path("/home/tuf/code/ort_demo/models/ch_PP-OCRv4_mobile_rec.onnx")
    dict_path: Path = Path("/home/tuf/code/ort_demo/models/ch_PP-OCRv4_dict.txt")

    dict_blank: int = 0
    dict_offset: int = 1


@dataclass
class PPOCRv5(BaseConfig):
    """Configuration for PPOCRv5 model."""
    cls_path:  Path = Path("/home/tuf/code/ort_demo/models/ch_PP_mobile_v2.0_cls.onnx")
    det_path:  Path = Path("/home/tuf/code/ort_demo/models/ch_PP-OCRv5_mobile_det.onnx")
    rec_path:  Path = Path("/home/tuf/code/ort_demo/models/ch_PP-OCRv5_mobile_rec.onnx")
    dict_path: Path = Path("/home/tuf/code/ort_demo/models/ch_PP-OCRv5_dict.txt")

    dict_blank: int = 0
    dict_offset: int = 1