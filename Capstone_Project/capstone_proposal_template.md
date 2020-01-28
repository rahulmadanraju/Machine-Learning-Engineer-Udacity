# Machine Learning Engineer Nanodegree
## Capstone Proposal
Rahul Madan Raju  
January 25, 2020

## Proposal for Capstone Project
Dog Breed Classification

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

Here, in the Dog Breed Classification, the dataset contains the images of Dogs and Humans. There are a total of 8351 images for dogs and 13233 images for humans. Using these images as data, it has to be processed according to our needs and a model has to be designed to train our machine. 

The input to the neural net is either an image of a dog or human to which we expect the output as the breed of the dog (for dog input) or tyoe if dog the human resembles to (for a human image input). 

### Benchmark Model

To tackle such data, it is preferably good to go with neural networks. The extraction of features for an image data is quite tedious. However, on using the convolutional neural networks, the features are easily extracted through it. Neural nets try to find a pattern to recognise features for each category of data and classify them based on the similarity measures.

As observed the input is expected to be a fog image or a human image. The output is the class of breed the dog belongs, and the type of dog the human resembles.

### Evaluation Metrics

The evaluation metrics that can be used to evaluate the performance of the machine learning model are:

Accuracy: The ratio of correct predictions to the total size of the data (i.e. (TP+TN)/Data Size)

Confusion matrix:

Recall: The ratio of true positives to the true positive and false negative (i.e. TP/(TP+FN))

Precision: The ratio of true positives to the true positive and false positive (i.e. TP/(TP+FP))


### Project Design

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
