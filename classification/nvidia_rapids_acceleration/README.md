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

## Step-by-step tutorial: <br>

### Pandas vs RAPIDS CuDF for data processing
Part 1 (Before Acceleration): <br>
• Import necessary modules like time, pandas, etc <br>
• Start the timer <br>
• Load the train_sample.csv file using pandas <br>
• End the timer <br>
• Store the elapsed time in a variable called CPU_Time <br>

Part 2 (After Acceleration): <br>
• Import necessary modules like time, CuDF, etc <br>
• Start the timer <br>
• Load the train_sample.csv file using CuDF <br>
• End the timer <br>
• Store the elapsed time in a variable called GPU_Time <br> 

Part 3: <br>
Compare CPU_Time and GPU_Time and visualize: <br>
CPU Time: 0.21895456314086914 <br>
GPU Time: 0.09745955467224121 <br>
CPU Time/GPU Time: 2.246619778606813 <br>

### Conclusion: 
The time taken by accelerated CuDF is less than half of the time taken by pandas to process the same file. This is extremely useful especially while reading the big files in ML models. <br>

### Sklearn’s KNN vs RAPIDS CuML for Classification model using KNN 

Part 1 (Before Acceleration): <br>
• from sklearn.neighbors import KNeighborsClassifier <br>
• Start the timer <br>
• Load training data using pandas <br>
• Create feature and target arrays for training data <br>
• Load testing data using pandas <br>
• Create feature and target arrays for testing data <br>
• Fit a KNN model from Sklearn <br>
• Predict on dataset which model has not seen before <br>
• End the timer <br>
• Store the elapsed time in a variable called CPU_Time <br>

Part 2 (After Acceleration): <br>
• Import cudf, from cuml.neighbors import KNeighborsClassifier <br>
• Start the timer <br>
• Load training data using pandas <br>
• Create feature and target arrays for training data <br>
• Load testing data using pandas <br>
• Create feature and target arrays for testing data <br>
• Fit a KNN model from CuML <br>
• Predict on dataset which model has not seen before <br>
• End the timer <br>
• Store the elapsed time in a variable called GPU_Time <br>

Part 3: • Compare CPU_Time and GPU_Time and visualize <br>
Time taken on CPU = 2.1018526554107666 <br>
Time taken on GPU = 0.5795042514801025 <br>
CPU Time to GPU time ratio: 3.6269840127013016 <br>

### Conclusion: 
The time taken by accelerated CuML is much lesser than the time taken by sklearn to build a KNN model and predict the classes for same dataset. <br>

### XGBoost on CPU vs RAPIDS GPU-accelerated XGBoost for Regression model 
Since this is a regression problem, I have used a different dataset here which is an in-built dataset from sklearn library about California Housing. So the following model is about Housing Price Prediction (Regression Model) using XGBoost. <br> 

Part 1 (Build a model): <br>
• Import xgboost, numpy and sklearn datasets <br>
• Create input features and the output predictor <br>
• Train the XGboost model <br>

Part 2 (Before Acceleration): <br>
• Enable CPU accelerated prediction <br>
• Start the timer <br>
• Predict on dataset which model has not seen before <br>
• End the timer <br>
• Store the elapsed time in a variable called CPU_Time <br>

Part 3 (After Acceleration): <br>
• Enable GPU accelerated prediction <br>
• Start the timer <br>
• Predict on dataset which model has not seen before <br>
• End the timer
• Store the elapsed time in a variable called GPU_Time <br>

Part 3: <br>
• Compare CPU_Time and GPU_Time and visualize <br>

### Conclusion: <br>
RAPIDS XGBoost clearly beats the CPU time in case of regression model also. As demonstrated in 3 different ways above, NVIDIA RAPIDS is a great way to accelerate classification/regression models on GPU. <br>

## References: <br>
1) Dataset: https://www.kaggle.com/datasets/srinivasav22/mnist-digit-recognition-using-knn <br>
2) Environment Setup: https://colab.research.google.com/drive/1xnTpVS194BJ0pOPuxN4GOmypdu2RvwdH#scrollTo=GzdJ7SnJ4QDD <br>
3) RAPIDS for ML: https://github.com/rapidsai/cuml <br>
