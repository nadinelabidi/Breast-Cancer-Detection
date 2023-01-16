# Breast-Cancer-Detection
## Predict Breast Cancer using SageMaker’s Linear-Learner with features derived from images of Breast Mass
We have taken  UCI’S breast cancer diagnostic dataset. The purpose here is to use this data set to build a predictve model of whether a breast mass image indicates benign or malignant tumor. We will use aws SageMaker to build our moderl.
Amazon SageMaker helps data scientists and developers to prepare, build, train, and deploy high-quality ML models quickly by bringing together a broad set of capabilities purpose-built for ML.

### Create an Amazon SageMaker Notebook Instance
An Amazon SageMaker notebook instance is a fully managed machine learning (ML) Amazon Elastic Compute Cloud (Amazon EC2) compute instance that runs the Jupyter Notebook App. You use the notebook instance to create and manage Jupyter notebooks for preprocessing data and to train and deploy machine learning models.
We will start by creating a (ml.t2.medium,Amazon Linux2,Jupyter lab 3) instance.
1. Open the [Amazon SageMaker console](https://console.aws.amazon.com/sagemaker/)
2. Choose Notebook instances, and then choose Create notebook instance
3. On the Create notebook instance page, provide the necessary information
4. For IAM role, choose Create a new role, and then choose Create role. This IAM role automatically gets permissions to access any S3 bucket that has sagemaker in the name. It gets these permissions through the AmazonSageMakerFullAccess policy, which SageMaker attaches to the role.

### Create a Jupyter Notebook
To start scripting for training and deploying your model, create a Jupyter notebook in the SageMaker notebook instance. Using the Jupyter notebook, you can conduct machine learning (ML) experiments for training and inference while accessing the SageMaker features and the AWS infrastructure.

##### For this notebook, we will start with importing the necessary packages
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import boto3
import re
import json
import sagemaker.amazon.common as smac
import sagemaker
``` 
##### Get the execution role for the sagemakersession and fix the file were your data  ```
# get the execution role for the sagemaker session
role = sagemaker.get_execution_role()
# get the region of the current session
region = boto3.Session().region_name

# get the name of the bucket for the current session: were we will upload and store data for training,modeling ..
bucket = sagemaker.Session().default_bucket()
# fix the file name within the bucket were all data and tasks can be located together
prefix = (
    "sagemaker/breast-cancer-prediction"  # place to upload training files within the bucket
)
``` 
3. we will get the Wisconsin Breast Cancer dataset from the UCI Machine Learning Repository
     - Dataset Information
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
     - Attribute Information
| ID number    |  Diagnosis (M = malignant, B = benign)   | radius     |  texture           |  perimeter         |  area |  smoothness              | compactness | concavity | concave points|symmetry | fractal dimension |radius_se| texture_se| perimeter_se|  |  area_se |  smoothness_se  | compactness_se | concavity_se | concave points_se|symmetry_se | fractal dimension_se| radius_worst  |  texture _worst  |  perimeter_worst  |  area_worst |  smoothness  _worst | compactness_worst | concavity_worst | concave points_worst|symmetry _worst| fractal dimension_worst|

```
#get the data
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data')
# set the columns name
data.columns = ["id","diagnosis","radius","texture","perimeter","area","smoothness",
                "compactness","concavity","concave points","symmetry","fractal_dimension",
                "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                "concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
                "perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
                "concave points_worst","symmetry_worst","fractal_dimension_worst"] 

#save the data
data.to_csv("data.csv", sep=',', index=False)
```


   



