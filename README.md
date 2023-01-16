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
```
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
##### we will get the Wisconsin Breast Cancer dataset from the UCI Machine Learning Repository

###### Dataset Information

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

######Attribute Information

| ID number    |  Diagnosis (M = malignant, B = benign)   | radius     |  texture           |  perimeter         |  area |  smoothness              | compactness | concavity | concave points|symmetry | fractal dimension |radius_se| texture_se| perimeter_se|  area_se |  smoothness_se  | compactness_se | concavity_se | concave points_se|symmetry_se | fractal dimension_se| radius_worst  | texture_worst| perimeter_worst|  area_worst |  smoothness_worst| compactness_worst | concavity_worst| concave points_worst|symmetry_worst | fractal dimension_wort|

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
##### for the data preparation, check the [notebook](https://github.com/nadinelabidi/Breast-Cancer-Detection/blob/main/Breast%20Cancer%20detection.ipynb)

##### Linear Learner Algorithm
Our problem is a classification problem and for that we will use linear models.
Linear models are supervised learning algorithms used for solving either classification or regression problems.
The Amazon SageMaker linear learner algorithm provides a solution for both classification and regression problems. With the SageMaker algorithm, you can simultaneously explore different training objectives and choose the best solution from a validation set. supports both recordIO-wrapped protobuf and CSV formats. For the application/x-recordio-protobuf input type, only Float32 tensors are supported. 

Amazon SageMaker’s Linear Learner actually fits many models in parallel, each with slightly different hyperparameters, and then returns the one with the best fit. for that we will create a job that will train and validate "32" models
   

```
linear_job = "linear_learner_job"

print("Job name is:", linear_job)

linear_training_params = {
    "RoleArn": role,
    "TrainingJobName": linear_job,
    "AlgorithmSpecification": {"TrainingImage": container, "TrainingInputMode": "File"},
    "ResourceConfig": {"InstanceCount": 1, "InstanceType": "ml.c4.2xlarge", "VolumeSizeInGB": 10},
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/train/".format(bucket, prefix),
                    "S3DataDistributionType": "ShardedByS3Key",
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None",
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/validation/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None",
        },
    ],
    "OutputDataConfig": {"S3OutputPath": "s3://{}/{}/".format(bucket, prefix)},
    "HyperParameters": {
        "feature_dim": "30",
        "mini_batch_size": "100",
        "predictor_type": "regressor",
        "epochs": "10",
        "num_models": "32",
        "loss": "absolute_loss",
    },
    "StoppingCondition": {"MaxRuntimeInSeconds": 60 * 60},
}
```
##### training the linear model
```
linear_hosting_container = {
    "Image": container,
    "ModelDataUrl": sm.describe_training_job(TrainingJobName=linear_job)["ModelArtifacts"][
        "S3ModelArtifacts"
    ],
}

create_model_response = sm.create_model(
    ModelName=linear_job, ExecutionRoleArn=role, PrimaryContainer=linear_hosting_container
)

print(create_model_response["ModelArn"])
```
##### Now that we’ve trained the linear algorithm on our data, let’s setup a model which can later be hosted. We will:
###### Point to the scoring container
###### Point to the model.tar.gz that came from training  
###### Create the hosting model
```
linear_endpoint_config = "DEMO-linear-endpoint-config-" + time.strftime(
    "%Y-%m-%d-%H-%M-%S", time.gmtime()
)
print(linear_endpoint_config)
create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName=linear_endpoint_config,
    ProductionVariants=[
        {
            "InstanceType": "ml.m4.xlarge",
            "InitialInstanceCount": 1,
            "ModelName": linear_job,
            "VariantName": "AllTraffic",
        }
    ],
)

print("Endpoint Config Arn: " + create_endpoint_config_response["EndpointConfigArn"])
```

