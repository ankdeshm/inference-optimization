# Inference Optimization for Image Classification <br>

## Introduction: <br>
Inference optimization is crucial for enhancing the efficiency and speed of deep learning models, especially when deploying them in real-world applications. Optimized inference reduces computational resource requirements, enabling models to run faster and consume fewer computational resources. PyTorch provides powerful tools to achieve inference optimization, such as quantization and TorchScript. Quantization allows for the conversion of high-precision floating-point models to low-precision representations, reducing memory and computation requirements. TorchScript, on the other hand, enables the compilation of PyTorch models into a serialized format, which can be executed more efficiently and integrated into various deployment environments, making it essential for efficient and scalable model inference in production settings. In this assignment I have used the code snippets from Ref [1] and [2] to build a simple CNN model for image classification in PyTorch. The code to test the model size, latency, inference time and accuracy is written with the help of PyTorch documentation [3]. <br>

## Dataset: <br> 
CIFAR10 Here are some key details about the CIFAR-10 dataset: - Number of images: 60,000 - Number of classes: 10 - Number of images per class: 6,000 - Image size: 32x32 pixels - Color channels: 3 (RGB) - Training set: 50,000 images (5,000 per class) - Test set: 10,000 images (1,000 per class)

### Approach 1 - PyTorch vs TorchScript Inference: <br>
1. Build a simple convolutional neural network using PyTorch [1] <br>
2. Find accuracy on the test dataset = 63.14% <br>
3. Create a serialized and optimized TorchScript representation of a PyTorch model. <br>
4. Find accuracy on the test dataset = 63.14% <br>
5. There is no change in the accuracy for PyTorch and TorchScript. So, TorchScript inference does not necessarily improve the accuracy of the model.<br>
6. Compare the average inference time using PyTorch model and its TorchScript representation. <br>
7. Average PyTorch Inference Time: 0.00937 seconds <br>
8. Average TorchScript Inference Time: 0.00891 seconds <br>
9. Inference time for TorchScript is slightly less than that of PyTorch. <br>
10. Measure and compare the latency of PyTorch and TorchScript-optimzed model. <br>
11. Latency (PyTorch): 0.00686 seconds <br>
12. Latency (TorchScript): 0.00210 seconds <br>
13. Measure and compare the model size of PyTorch and TorchScript-optimized model. <br>
14. PyTorch Model Size: 12.28 MB <br>
15. TorchScript Model Size: 12.29 MB <br>
16. The model size for this particular model is almost equal in PyTorch and TorchScript. <br>

## Conclusion: <br>
I compared various parameters for inference optimization using PyTorch & TorchScript and observed that while the accuracy and the model size remained almost constant, TorchScript performed significantly better in terms of inference time and latency. <br>

## Approach 2 - PyTorch Quantization <br>
1. Build a simple convolutional neural network using PyTorch [2]. <br>
2. Apply dynamic quantization to the PyTorch model. <br>
3. Check model size for the models with and without quantization. <br>
4. Size without quantization: 13 MB. <br>
5. Size with quantization: 3 MB. <br>
6. Measure the time taken for inference of the model with and without quantization. <br>
7. Average time per inference (FP32): 0.002512 seconds <br>
8. Average time per inference (INT8): 0.000844 seconds <br>
9. Hence the quantized model is faster. <br>


## Conclusion: <br>
I compared various parameters for inference optimization using PyTorch & its quantized version by applying dynamic quantization and observed that the quantized model outperformed the non-quantized version of the model in terms of inference time and latency. <br>


## References: <br>
1) CNN in PyTorch for image classification: https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48 <br>
2) PyTorch Quantization Code Reference: https://gist.github.com/LilitYolyan/96ea2c9eaad511d3b0ffa87eff805e09#file-net-py <br>
3) PyTorch documentation: https://pytorch.org/docs/stable/nn.html <br>
