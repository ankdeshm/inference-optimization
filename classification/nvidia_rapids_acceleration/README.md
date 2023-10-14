# Accelerate classification via NVIDIA optimizations

## Dataset: MNIST Digit Recognition Using KNN (Source: Kaggle)

##  Introduction: 
NVIDIA’s RAPIDS offers an efficient way to enhance the speed of ML procedures such as regression, classification etc. The betterment in the speed of operations in RAPIDS is due to GPU acceleration. While Python libraries run on CPU, RAPIDS libraries run on GPUs. In this notebook, I have performed the 3 separate comparisons to prove the time efficiency of RAPIDS. <br>
1) Pandas vs RAPIDS CuDF for data processing <br>
2) Sklearn’s KNN vs RAPIDS CuML for classification model using KNN <br>
3) XGBoost on CPU vs RAPIDS GPU-accelerated XGBoost for regression model <br>

## Process/Environment Setup: 
• Import Dataset MNIST Digit Recognition <br>
• Install RAPIDS-Colab files and test the GPU. For accelaration, we need to make sure we have one of the RAPIDS compatible GPUs available in colab such as Tesla T4, P4, or P100. So, if you are allocated a GPU other than these, you’ll have to keep changing the runtime type to GPU until you are issued on the RAPIDS compatible GPUs. <br>
• Update the Colab environment and restart the kernel. The session will be crashed during this execution. <br>
• Install CondaColab. This will restart the kernel one last time and the session will crash. <br>
• Install RAPIDS most recent stable version. This process takes about 15-10 minutes. Also it’s important to note that. When working on colab, you’ll have to install RAPIDS and the necessary libraries every time you run the GPU. This is one of the disadvantages of using Colab over Kaggle. <br>
• Once the installation is done, we can use cuML, CuDF and XGBoost to accelerate the respective functionalities.<br>
