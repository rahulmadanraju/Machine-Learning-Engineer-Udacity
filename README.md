# Machine Learning Engineer - Udacity

## Syllabus

<br>
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/syl1.JPG" />
<br>


<br>
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/syl2.JPG" />
<br>


# Machine Learning Deployment using AWS SageMaker

Code and associated files 

This repository contains code and associated files for deploying ML models using AWS SageMaker. This repository consists of a number of project notebooks for various coding exercises, mini-projects, and project files that will be used to supplement the lessons of the Nanodegree.


### Project 1 - Sentiment Analysis

[Sentiment Analysis Web App](https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/tree/master/Sentiment_Analysis_Project)
is a notebook and collection of Python files to be completed. The result is a deployed RNN performing sentiment analysis on movie reviews complete with publicly accessible API and a simple web page which interacts with the deployed endpoint. This project assumes that you have some familiarity with SageMaker. Completing the XGBoost Sentiment Analysis notebook should suffice but also making a comparative study with LSTM model gives a description of analysis made.

<br>
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer---Nano-Degree---Udacity-Projects/blob/master/Sentiment_Analysis_Project/reviews.JPG" />
<br>


# Machine Learning, Deployment Case Studies

This repository contains associated files for deploying ML models using AWS SageMaker. This repository consists of a number of project  notebooks for various case studies, code exercises, and project files

### Tutorials

* [Population Segmentation](https://github.com/udacity/ML_SageMaker_Studies/tree/master/Population_Segmentation): Learn how to build and deploy unsupervised models in SageMaker. In this example, you'll cluster US Census data; reducing the dimensionality of data using PCA and the clustering the resulting, top components with k-means.
* [Payment Fraud Detection](https://github.com/udacity/ML_SageMaker_Studies/tree/master/Payment_Fraud_Detection): Learn how to build and deploy a supervised, LinearLearner model in SageMaker. You'll tune a model and handle a case of class imbalance to train a model to detect cases of credit card fraud.
* [Deploy a Custom PyTorch Model (Moon Data)](https://github.com/udacity/ML_SageMaker_Studies/tree/master/Moon_Data): Train and deploy a custom PyTorch neural network that classifies "moon" data; binary data distributed in moon-like shapes.
* [Time Series Forecasting](https://github.com/udacity/ML_SageMaker_Studies/tree/master/Time_Series_Forecasting): Learn to analyze time series data and format it for training a [DeepAR](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html) algorithm; a forecasting algorithm that utilizes a recurrent neural network. Train a model to predict household energy consumption patterns and evaluate the results.

<br>
<img src="https://github.com/rahulmadanraju/ML_SageMaker_Studies/blob/master/Time_Series_Forecasting/notebook_ims/example_applications.png" />
<br>

### Project 2 - Plagiarism Detection

[Plagiarism Detector](https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/tree/master/Project_Plagiarism_Detection): Building an end-to-end plagiarism classification model. Applying the skills to clean data, extract meaningful features, and deploying a plagiarism classifier in SageMaker.

In this project, you will be tasked with building a plagiarism detector that examines a text file and performs binary classification; labeling that file as either *plagiarized* or *not*, depending on how similar that text file is to a provided source text. Detecting plagiarism is an active area of research; the task is non-trivial and the differences between paraphrased answers and original work are often not so obvious.

This project will be broken down into three main notebooks:

**Notebook 1: Data Exploration**
* Load in the corpus of plagiarism text data.
* Explore the existing data features and the data distribution.
* This first notebook is **not** required in your final project submission.

<br>
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/graphpd.JPG" />
<br>

**Notebook 2: Feature Engineering**

* Clean and pre-process the text data.
* Define features for comparing the similarity of an answer text and a source text, and extract similarity features.
* Select "good" features, by analyzing the correlations between different features.
* Create train/test `.csv` files that hold the relevant features and class labels for train/test data points.

<br>
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/pddata.JPG" />
<br>

**Notebook 3: Train and Deploy Your Model in SageMaker**

* Upload your train/test feature data to S3.
* Define a binary classification model and a training script.
* Train your model and deploy it using SageMaker.
* Evaluate your deployed classifier.

 Overall accuracy score:
<br>
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/pd.JPG" />
<br>

**Notebook 4: Capstone: Dog Breed Classification using Transfer Learning**

Dog Breed Classification

<p align="center"> 
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images_Report/Beagle_01197.jpg" />  
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images_Report/Alaskan_malamute_00330.jpg" />
<p>

### Domain Background

In Dog Breed Classification, we see the dataset contains images of dogs and humans to which we have to classify dogs based on their breeds. Then why Humans? the images of the humans are used to see what category of dog breeds will they be classified (for fun purpose)
Also, when observed, we categorize such works to the field of Computer Vision and Machine Learning, to which there are various works carried on in the above project. 

Recently in 2019, Punyanuch Borwarnginn et al proposed the work on dog breed classification using the different approaches to classify them based on their breeds to tackle population control, disease breakout, vaccination control and legal ownership. He and his co-workers used 1. Histogram Oriented Gradient and 2. Convolutional Neural Network using Transfer Learning for the classification purpose. On making a comparative study, they found that the Neural Nets had a better performance compared to the HOG descriptor. 

(### Breakthrough Conventional Based Approach for Dog Breed Classification Using CNN with Transfer Learning)

Looking into the work carried above, we see how the dog breed classification can be used for determining various information. This can be further expanded to determine information on the 1. Types of Dog Breeds preferably chosen by humans 2. Demographic Needs for the Dogs 3. Behavior analysis of Dogs at different demographic locations, etc.


### Problem Statement

Here the goal is to create a Dog Breed Classifier and build an application for the same. The tasks involved are:
1. Download and Process the Images of the Dogs and Humans
2. Detect the Dogs and Humans using the Pre-Trained models such as Haarcascade and VGG-16
3. Build and train a classifier to classify dog breeds from scratch
4. Also, train the model using transfer learning with an efficiency to be used for application too.
5. Using the App, predict the breed of the dog and also the category of dog breed the human resembles.

It is an application that can be quite handy to recognize the breeds of unknown dogs for the user and also have fun by creating a resemblance of a dog to the given human images.

### Datasets and Inputs

Here, in the Dog Breed Classification, the dataset contains the images of Dogs and Humans. There are a total of 133 breeds, 8351 images for dogs. Using these images as data, it has to be processed according to our needs and a model has to be designed to train our machine. 
 
In our case, we observe that the split of train and test data is 90%-10%, i.e. 90% for training and 10% for testing purposes. In the training data, we have reserved another 10% for validation. The resultant split of data can be observed in the below graph.

From the below plot we can observe that a total of 6680 images will be used to train our machine, to further fine-tune the parameters we use another 835 images for validating it. And, lastly, we will be using 836 images to test our model's performance for various evaluation of metrics.

<<p align="center"> >
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/TVT.png" />
<p>

On observing the distribution of data within each class, the number of images are beyond certain threshold value (i.e. nearly 40 per class). Though the data is not distributed evenly along the graph, the number of images are sufficient to predict the class of a particular breed and look balanced.

<p align="center"> 
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/download.png" />
<p>



### Solution Statement

The user end application will be designed which will be useful to identify the dog breeds. To tackle this, we need to train our model to identify the dog based on certain features. The convolutional neural network can be used to train the model and based on the evaluation and performance of the model, an application will be built for the user experience.

### Benchmark Model

To tackle such data, we use a benchmark model to build a basic pipeline and a well-versioned model to improvise our classification rate. Such a methodology is carried to tune our model for better prediction of results. The benchmark model helps us to make a comparison and reduce the overfitting or underfitting condition and tune the model for a better outcome. Logistic Regression is one such example of the benchmark. We can also use the predefined image classifiers such as ImageNet, ResNet, VGG16, etc. to classify our images and later optimize our pipeline for better evaluation of metrics.

In the works of Dog Identification in Kaggle, we see that "Mohamed Sheik Ibrahim" used the VGG19, a predefined base model and carried various processing techniques such as data augmentation to improvise the results obtained from the predefined model. He also used logistic regression to classify the images of dogs and achieved an accuracy of 68%.

Considering another work performed on the same data, using the inception v3, the pre-trained model for image classification, Carey B achieved an accuracy of 87%, which is considered to be a good classification rate based on the performance of the model. 

Using the above understandings, We will be using VGG16 for our data for classifying the breeds of the Dogs, Later we build a Convolutional Neural Network and tune the parameters, make a comparative study and analyse the performance of the model. 

### Evaluation Metrics

The evaluation metrics that can be used to evaluate the performance of the machine learning model are:

Accuracy: The ratio of correct predictions to the total size of the data (i.e. (TP+TN)/Data Size)

Confusion matrix:

Recall: The ratio of true positives to the true positive and false negative (i.e. TP/(TP+FN))

Precision: The ratio of true positives to the true positive and false positive (i.e. TP/(TP+FP))
<p align="center"> 
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images_Report/CF.png" /> 
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/pre-rec.png" />
<p>
 
### Project Design
 
The following project can be designed according to the below workflow.

1. Data loading and exploration
2. Data augmentation and processing
3. Detect the Dog and Humans using the given detector algorithms
4. Build a CNN training model from scratch using Deep Learning framework - Pytorch
5. Build a training model using the Transfer Learning
6. Evaluate the performance of the model
7. Design and build an application using the above model for the user experience
8. Test the working model of the application using user inputs

The above steps are structured for the implementation of the project and any additional steps carried will be reported after the project implementation.

## Results

Detection algorithm comparison: Haarcascades vs Local Binary Pattern (LBP) cascades

Pre-trained model: VGG 16

Custom model accuracy: 12%

The output meets the requirement of the task with a prediction accuracy of **85%**.

The result of the above implementation is shown below - 
<p align="center"> 
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images_Report/1.PNG" />
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images_Report/2.PNG" />
<p>
---
