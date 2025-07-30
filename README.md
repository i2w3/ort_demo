# What is this
基于 onnxruntime 实现的 cpp 和 python 语言模型推理 demo

# ENVS
[torch 与 torchvision 版本匹配](https://pytorch.org/get-started/previous-versions/)

```bash
mamba create -n cu126 -c conda-forge python==3.11 cuda-toolkit==12.6 cuda-nvcc-tools==12.6.20 cudnn==9.3.0.75 cuda-nvtx==12.6.77
mamba activate cu126
mamba install pytorch==2.7.1 torchvision==0.22.0 onnxruntime==1.22.0
mamba install onnxruntime-cpp libopencv
```

# python
```
PYTHONPATH=./ python ./test/test_ppocr_rec.py
```