# Machine Learning Engineer Nanodegree
## Capstone Proposal
Rahul Madan Raju  
January 25, 2020

## Proposal for Capstone Project
Dog Breed Classification

<br>
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images/Beagle_01197.jpg" />  
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images/Alaskan_malamute_00330.jpg" />
<br>

### Domain Background

In Dog Breed Classification, we see the dataset contains images of dogs and humans to which we have to classify dogs based on their breeds. Then why Humans? the images of the humans are used to see what category of dog breeds will they be classified (for fun purpose)
Also, when observed, we categorise such works to the field of Computer Vision and Machine Learning, to which there are various works carried on in the above project. 

Recently in 2019, Punyanuch Borwarnginn et al proposed the work on dog breed classifcation using the different approaches to classify them based on their breeds to tackle population control, disease breakout, vaccination control and legal ownnership. He and his co workers used 1. Histogram Oriented Gradient and 2. Convolutional Neural Network using Transfer Learning for the classification purpose. On making a comparative study, they found that the Neural Nets had a better performance compared to the HOG descriptor. 

(### Breakthrough Conventional Based Approach for Dog Breed Classification Using CNN with Transfer Learning)

Looking into the work carried above, we see how the dog breed classifcation can be used for determining various information. This can be further expanded to determine information on the 1. Types of Dog Breeds preferably chosen by humans 2. Demographic Needs for the Dogs 3. Behaviour analysis of Dogs at different demographic locations, etc.


### Problem Statement

Here the goal is to create a Dog Breed Classifier and build an application for the same. The tasks involved are:
1. Download and Process the Images of the Dogs and Humans
2. Detect the Dogs and Humans using the Pre-Trained models such as Haarcascade and VGG-16
3. Build and train a classifier to calssify dog breeds from scratch
4. Also, train the model using transfer learning with an effieciency to be used for application too.
5. Using the App, predict the breed of the dog and also the category of dog breed the human resembles to.

It is application which can be quite handy to recognise the breeds of unknown dogs for the user and also have fun by creating a resemblance of a dog to the given human images.

### Datasets and Inputs

Here, in the Dog Breed Classification, the dataset contains the images of Dogs and Humans. There are a total of 133 breeds, 8351 images for dogs. Using these images as data, it has to be processed according to our needs and a model has to be designed to train our machine. 
 
In our case we observe that the split of train and test data is 90%-10%, i.e. 90% for training and 10% for testing purpose. In the training data we have reserved an other 10% for validation. The resultant split of data can be observed in the below graph.

From the below plot we can observe that, a total of 6680 images will be used to train our machine, to further fine tune our parameters we use an other 835 images for validating it. And, lastly, we will be using 836 images to test our models performance for various evaluation of metrics.

<br>
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/TVT.png" />
<br>

On observing the distribution of data within each class, the number of images are beyond certain threshold value (i.e. nearly 40 per class). Though the data is not distributed evenly along the graph, the number of images are sufficient to predict the class of a particular breed and look balanced.

<br>
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/download.png" />
<br>



### Solution Statement

The user end application will be designed which will be useful to identify the dog breeds. To tackle this, we need tp train our model to identiy the dog based on the certain features. The convolutional neural network can be used to train the model and based on the evaluation and performance of the model, an application will be built for the user experience.

### Benchmark Model

To tackle such data, we use a benchmark model to build a basic pipeline and a well versioned model to improvise our classification rate. Such a methodology is carried to tune our model for better prediction of reults. The benchmark model helps us to make comparison and reduce the overfitting or underfitting condition and tune the model for better outcome. Logistic Regression is one such example of the bench mark. We can also us the pre defined image classifiers such as ImageNet, ResNet, VGG16, etc. to classify our images and later optimize our pipeline for better evaluation of metrics.

In the works of Dog Identifcation in Kaggle, we see that "Mohamed Sheik Ibrahim" used the VGG19, a predefined base model and carried various processing techniques such as data augmentation to improvise the results obtained from the predefined model. He also used  logistic regression to classify the images of dogs and acheived an accuracy of 68%.

Considering an other work perfomed on the same data, using the inception v3, the pre-trained model for image classification, Carey B achieved an accuracy of 87%, whihc is consdered to be a good classification rate based on the performance of the model. 

Using the above understandings, We will be using VGG16 for our data for classfying the breeds of the Dogs, Later we build a Convolutional Neural Network and tune the parameters, make a comaprative study and anlyse the performance of the model. 

As observed the input is expected to be a dog image or a human image. The output is the class of breed the dog belongs, and the type of dog the human resembles.

### Evaluation Metrics

The evaluation metrics that can be used to evaluate the performance of the machine learning model are:

Accuracy: The ratio of correct predictions to the total size of the data (i.e. (TP+TN)/Data Size)

Confusion matrix:

Recall: The ratio of true positives to the true positive and false negative (i.e. TP/(TP+FN))

Precision: The ratio of true positives to the true positive and false positive (i.e. TP/(TP+FP))

<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images/CF.png" /> 

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

-----
