# TorchScript Inference <br>

## Inference via TorchScript: 

### Introduction: <br>
With TorchScript, PyTorch aims to create a unified framework from research to production. TorchScript takes our PyTorch modules as input and convert them into a production-friendly format. It will run the models faster and independent of the Python runtime. To focus on the production use case, PyTorch uses 'Script mode' which has 2 components PyTorch JIT and TorchScript. <br>

### Example 1: <br>
In the first example, I have utilized BERT(Bidirectional Encoder Representations from Transformers) from the transformer’s library provided by HuggingFace. <br>

#### Steps: <br>
1) Initialize the BERT model/tokenizers and create a sample data for inference <br>
2) Prepare PyTorch models for inference on CPU/GPU <br>
3) Model/Data should be on the same device for training/inference to happen. cuda() transfers the model/data from CPU to GPU. <br>
4) Prepares TorchScript modules (torch.jit.trace) for inference on CPU/GPU <br>
5) Compare the speed of BERT and TorchScript <br>
6) Save the model in *.pt format which is ready for deployment <br>

#### Results:<br>
Module <br>
BERT <br> 
Latency on CPU (ms): 88.82  <br> 
Latency on GPU (ms): 18.77  <br>  

Module <br>
TorchScript <br>    
Latency on CPU (ms): 86.93 <br>
Latency on GPU (ms): 9.32 <br>

#### Conclusion: <br>
On CPU the runtimes are similar but on GPU TorchScript clearly outperforms PyTorch.<br>


### Example 2: <br>
In the second example, I have utilized ResNet, short for Residual Networks. 

#### Steps: <br>
1) Initialize PyTorch ResNet <br>
2) Prepare PyTorch ResNet model for inference on CPU/GPU <br>
3) Initialize and prepare TorchScript modules (torch.jit.script ) for inference on CPU/GPU <br>
4) Compare the speed of PyTorch ResNet and TorchScript <br>

#### Results: <br>
Module <br>
ResNet <br>
Latency on CPU (ms): 92.92 <br>
Latency on GPU (ms): 9.04

Module <br>
TorchScript <br>
Latency on CPU (ms): 89.58 <br>
Latency on GPU (ms): 2.53  <br>

#### Conclusion: <br>
TorchScript significantly outperforms the PyTorch implementation on GPU. As demonstrated in 2 different ways above, TorchScript is a great way to improve the inference improvement as compared to the original PyTorch inference. <br>

## References: <br>
1) https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#basics-of-torchscript <br>
2) https://towardsdatascience.com/pytorch-jit-and-torchscript-c2a77bac0fff <br>
