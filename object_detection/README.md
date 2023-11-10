# Inference Optimization for Object Detection <br>

## Introduction: <br>
Inference optimization is crucial for enhancing the efficiency and speed of deep learning models, especially when deploying them in real-world applications. Optimized inference reduces computational resource requirements, enabling models to run faster and consume fewer computational resources. In this project, I have employed a pre-trained YOLOv8 model sourced from Ultralytics and customized it to recognize various fashion items, creating what I refer to as the baseline model. The performance of this model was rigorously evaluated using metrics such as box loss, class loss, direction/offset loss, mean Average Precision at IoU=0.50 (mAP50), and mean Average Precision at IoU=0.50 to 0.95 (mAP50-95). Subsequently, I applied advanced inference acceleration techniques including TorchScript and ONNX to further enhance model efficiency. The culmination of this project involved a comparative analysis of the model sizes and inference durations across the baseline, TorchScript, and ONNX models.

## Dataset: <br> 
Colorful Fashion Dataset For Object Detection: https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection
The classes of the this dataset: sunglass, hat, jacket, shirt, pants, shorts, skirt, dress, bag, shoes.

## Custom Model - YOLOv8 for fashion Object Detection: <br> 
The YOLOv8 model was trained for fashion product detection using a dataset defined in `data.yaml`. Training was conducted over 5 epochs with a batch size of 8 and an image size of 128, utilizing a Python 3.10.12 and PyTorch 2.1.0 environment on a CPU (Intel Xeon 2.20GHz). The model's performance was evaluated based on several metrics, with the following key results observed: <br>

- Box Loss (`box_loss`): There was a consistent decrease in box loss over the epochs, starting from 1.719 in the first epoch to 1.354 in the final epoch, indicating improved bounding box prediction accuracy over time. <br>
  
- Class Loss (`cls_loss`): Similar to box loss, class loss saw a reduction from 1.968 to 1.018, reflecting enhanced classification performance as training progressed. <br>
  
- Direction/Offset Loss (`dfl_loss`): This metric also showed a downward trend from 1.203 to 1.040, suggesting improved localization of object detection throughout the training. <br>
  
- Mean Average Precision at IoU=0.50 (mAP50): The mAP50 metric increased from 0.535 to 0.656, demonstrating a notable improvement in model precision for detecting objects with at least 50% overlap with ground truths. <br>
  
- Mean Average Precision at IoU=0.50 to 0.95 (mAP50-95): The mAP50-95 metric similarly improved from 0.306 to 0.408, indicating better precision across a range of IoU thresholds. <br>

The inference performance post-training showed a processing speed of 105.8ms per image, with additional negligible preprocessing and postprocessing times. After optimization, the model size was reduced to 52.0MB. Detailed class-wise performance showed varying precision and recall rates, with `pants` achieving the highest mAP50 of 0.908 and `sunglass` showing the lowest at 0.0251. <br>

The overall training process took approximately 0.985 hours, and the validation results post-training confirmed the model's effectiveness, with consistent precision and recall across classes, resulting in a balanced mAP that underscores the model's robustness for fashion product detection tasks. <br>

The parameters yielding best results were saved as our baseline/custom model for further inference. <br>

#### Preparation for Inference: <br>
- Load and convert an image to RGB format, then resize it to 128x128 pixels for model compatibility. <br>
- Transform the resized image into a tensor and normalize it using predefined mean and standard deviation values. <br>
- Add a batch dimension to the preprocessed image tensor to prepare it for input into the model. <br>

#### Approach 1 - TorchScript Inference: <br>
- Convert the PyTorch model to TorchScript format using tracing, which records operations on a provided input tensor, and save the traced model. <br>
- Evaluate the TorchScript model's inference time by recording the duration it takes to process the given input tensor without gradient calculations. <br>
- Measure the inference time of the TorchScript model by executing the session and recording the start and end times. <br>
- Determine the file size of the TorchScript model in MB. <br>


#### Approach 2 - PyTorch Quantization <br>
- Export the PyTorch model `model_bsl` to the ONNX format with the specified input tensor and save it. <br>
- Initialize an ONNX Runtime session to load the exported ONNX model for inference. <br>
- Convert the PyTorch tensor to a NumPy array to match ONNX Runtime's expected input format. <br>
- Measure the inference time of the ONNX model by executing the session and recording the start and end times. <br>
- Determine the file size of the ONNX model in MB. <br>


#### Conclusion: <br>
I compared the model sizes and the inference times for all 3 models: baseline, TorchScript, and ONNX. In conclusion, the evaluative comparison of YOLOv8 baseline, TorchScript, and ONNX models illuminates distinct trade-offs between inference speed and storage size. The baseline YOLOv8 model requires the least storage at 49.58 MB but lags in inference performance, clocking in at 0.32 seconds. TorchScript moderately improves on this with a 0.28-second inference time, yet doubles the storage requirement. ONNX stands out by significantly enhancing inference speed to 0.07 seconds—a reduction of approximately 78% from the baseline and 75% from TorchScript—while maintaining a model size comparable to TorchScript. This performance profile underscores the importance of ONNX in applications where rapid inference is paramount, despite similar storage demands as TorchScript. Hence, ONNX emerges as the optimal choice for efficiency in computational performance without a considerable increase in model size.


## References: <br>
1) Fashion Object Detection - YOLOv8: (https://www.kaggle.com/code/rohitgadhwar/fashion-object-detection-yolov8) <br>
