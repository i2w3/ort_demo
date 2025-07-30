import time

import cv2
import numpy as np

import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))

class MultiLabelClassification:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.classes = self.load_classes("/home/tuf/code/ort_demo/models/resnet18.txt")

    def load_classes(self, classes_path):
        with open(classes_path, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f]
        return classes

    def __call__(self, tensor):
        ort_inputs:dict[str, np.ndarray] = {i.name: tensor for i in self.session.get_inputs()}
        results: list[np.ndarray] = self.session.run(None, ort_inputs)
        outputs = []
        for i, result in enumerate(results):
            output = []
            # softmax
            for index, score in enumerate(result[0]):
                if score > 0.25:
                    output.append({'class': self.classes[index],'score': score,})
            outputs.append(output)
        return outputs
    
def preprocess(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 获取原始尺寸
    original_h, original_w = image.shape[:2]
    target_size = 224
    
    # 计算缩放比例，保持长宽比
    scale = min(target_size / original_w, target_size / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    
    # 缩放图像
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 创建224x224的画布，填充为0（黑色）
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # 将缩放后的图像放置到左上角 (0, 0)
    canvas[0:new_h, 0:new_w] = resized_image
    # cv2.imwrite("demo.jpg", cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    # 归一化
    input_data = canvas.astype(np.float32) / 255.0
    input_data = (input_data - MEAN) / STD
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    
    return input_data


def detailed_benchmark(detector, tensor, num_runs=10):
    """详细的性能测试，包括预处理和后处理时间"""
    print(f"开始详细性能测试 ({num_runs} 次运行)...")
    
    # 预热
    _ = detector(tensor)
    
    total_times = []
    inference_only_times = []
    
    for i in range(num_runs):
        # 测试完整流程时间
        total_start = time.time()
        
        # 仅测试模型推理时间
        ort_inputs = {inp.name: tensor for inp in detector.session.get_inputs()}
        
        inference_start = time.time()
        raw_results = detector.session.run(None, ort_inputs)
        inference_end = time.time()
        
        # 后处理
        outputs = []
        for result in raw_results:
            output = []
            for index, score in enumerate(result[0]):
                if score > 0.25:
                    output.append({'class': detector.classes[index], 'score': score})
            outputs.append(output)
        
        total_end = time.time()
        
        total_time = (total_end - total_start) * 1000
        inference_time = (inference_end - inference_start) * 1000
        
        total_times.append(total_time)
        inference_only_times.append(inference_time)
        
        print(f"第 {i+1} 次 - 总时间: {total_time:.2f} ms, 纯推理: {inference_time:.2f} ms")
    
    print("\n=== 详细性能统计 ===")
    print(f"总流程平均时间: {np.mean(total_times):.2f} ms")
    print(f"纯推理平均时间: {np.mean(inference_only_times):.2f} ms")
    print(f"后处理平均时间: {np.mean(total_times) - np.mean(inference_only_times):.2f} ms")
    print(f"FPS (基于总时间): {1000 / np.mean(total_times):.1f}")
    print(f"FPS (基于推理时间): {1000 / np.mean(inference_only_times):.1f}")
    
    return {
        'total_times': total_times,
        'inference_times': inference_only_times,
        'avg_total': np.mean(total_times),
        'avg_inference': np.mean(inference_only_times)
    }


if __name__ == "__main__":
    detector = MultiLabelClassification("/home/tuf/code/ort_demo/models/resnet18.onnx")
    image = cv2.imread("/home/tuf/code/ort_demo/images/1.png")
    tensor = preprocess(image)
    # results = detector(tensor)
    # print(results)

    detailed_stats = detailed_benchmark(detector, tensor, num_runs=10)