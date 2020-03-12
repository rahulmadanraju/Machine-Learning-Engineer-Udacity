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

* [Population Segmentation]: Learn how to build and deploy unsupervised models in SageMaker. In this example, you'll cluster US Census data; reducing the dimensionality of data using PCA and the clustering the resulting, top components with k-means.
* [Payment Fraud Detection]: Learn how to build and deploy a supervised, LinearLearner model in SageMaker. You'll tune a model and handle a case of class imbalance to train a model to detect cases of credit card fraud.
* [Deploy a Custom PyTorch Model (Moon Data)]: Train and deploy a custom PyTorch neural network that classifies "moon" data; binary data distributed in moon-like shapes.
* [Time Series Forecasting]: Learn to analyze time series data and format it for training a [DeepAR](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html) algorithm; a forecasting algorithm that utilizes a recurrent neural network. Train a model to predict household energy consumption patterns and evaluate the results.

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

**Notebook 4: Capstone: Dog Breed Classification using Transfer Learning

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

