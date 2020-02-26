# Machine Learning Engineer Nanodegree
## Capstone Project : Dog Breed Classification
Rahul Madan Raju  
February , 2020

## I. Definition
_(approx. 1-2 pages)_

Classification is the process of the classifying things based on similarity of features. It is a supervised learning approach in which the computer program learns from the data input given to it and then uses this learning to classify new observation. Classification is one of the several methods intended to make the analysis of very large datasets effective.

Some of the examples of classification using machine learning are:

- Sign Language Indentification
- Speech Recognition
- Object Classification

Here, we will be discussing about: Dog Breed Classification

### Project Overview

In Dog Breed Classification, we will be classifying dogs based on their breeds. 

We see the dataset contains images of dogs and humans to which we have to classify dogs based on their breeds. Then why Humans? the images of the humans are used to see what category of dog breeds will they be classified (for fun purpose)


<p align="center"> 
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images_Report/Beagle_01197.jpg" />  
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images_Report/Alaskan_malamute_00330.jpg" />
<p>

Also, when observed, we categorize such works to the field of Computer Vision and Machine Learning, to which there are various works carried on in the above project. 

Recently in 2019, Punyanuch Borwarnginn et al proposed the work on dog breed classification using the different approaches to classify them based on their breeds to tackle population control, disease breakout, vaccination control and legal ownership. He and his co-workers used 

- Histogram Oriented Gradient
- Convolutional Neural Network using Transfer Learning for the classification purpose

On making a comparative study, they found that the Neural Nets had a better performance compared to the HOG descriptor.

### Problem Statement

Here the goal is to create a Dog Breed Classifier and build an application for the same. The tasks involved are:
- Download and process the Images of the Dogs and Humans
- Detect the Dogs and Humans using the detector algorithms such as haarcascades and local binary pattern cascades
- Build and train a classifier to classify dog breeds using a pre-trained model (VGG-16 or RESTNET50) and custom model
- Also, train the model using transfer learning with an efficiency to be used for application
- Using the App, predict the breed of the dog and also the category of dog breed the human resembles

It is an application that can be quite handy to recognize the breeds of unknown dogs for the user and also have fun by creating a resemblance of a dog to the given human image

### Metrics

The evaluation metrics that can be used to evaluate the performance of the machine learning models are:
- Accuracy: The ratio of correct predictions to the total size of the data (i.e. (TP+TN)/Data Size)
- Recall: The ratio of true positives to the true positive and false negative (i.e. TP/(TP+FN))
- Precision: The ratio of true positives to the true positive and false positive (i.e. TP/(TP+FP))

<p align="center"> 
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images_Report/CF.png" /> 
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/pre-rec.png" />
<p>
  
In our case we will be using the accuracy as the metric of measurement to evaluate the performance of the model.

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration

Here, in the Dog Breed Classification, the dataset contains the images of Dogs and Humans. There are a total of 133 breeds, 8351 images for dogs. Using these images as data, it has to be processed according to our needs and a model has to be designed to train our machine.

In our case, we observe that the split of train and test data is 90%-10%, i.e. 90% for training and 10% for testing purposes. In the training data, we have reserved another 10% for validation. The resultant split of data can be observed in the below graph.

From the below plot we can observe that a total of 6680 images will be used to train our machine, to further fine-tune the parameters we use another 835 images for validating it. And, lastly, we will be using 836 images to test our model's performance for the evaluation of metric i.e. Accuracy.

<p align="center"> >
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/TVT.png" />
<p>

### Exploratory Visualization

The study on distribution of data gives us the information on balance or imbalance in the data. If there is a imbalance in the data beyond a certain treshold, we must see that the data is balanced by adding relevant images. If the balance in the data is comparatively near to the treshold, it is good to carry forward with the operation. Let uss see how it works with our data in the below figure. The plot shows a clear descriotion on breed class with number of dogs.

<p align="center"> 
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Images/download.png" />
<p>

On observing the distribution of data within each class, the number of images are beyond certain threshold value (i.e. nearly 40 per class). Though the data is not distributed evenly along the graph, the number of images are sufficient to predict the class of a particular breed and looks balanced.

Provided the data is small, not sufficient and the model is getting to overfit. In such case, we need to augment the data to increase the number of image samples for the model. Therefore, augmentation makes an impact of bringing down the overfit condition and gives transparency to the model to make a predictive decision on test samples.

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

From exploring the data, we visualize the actual distribution of data in each class. In order to classify the images based on their breeds, we need a classification algorithm to do the work. In this case, we use the pre-trained model such as VGG16 and a custom model for classification purpose. The intention of using the VGG16 is the need of performing transfer learning to our self built custom algorithm.

VGG-16 Algorithm Model

VGG 16 is a type of Convolutional Neural Network trained on a dataset of over 14 million images belonging to 1000 classes. The algorithm was proposed by K. Simonyan and A. Zisserman in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The network consists of:
- A stack of convolutional layers 
- Three fully connected layers
    The first two FCN consists of 4096 channels followed by 1000 channels for the last FCN as each belong to one class. 
- The last layer is a softmax layer which gives a probabilistic distribution of the respective class.

<p align="center"> >
<img src="https://github.com/rahulmadanraju/Machine-Learning-Engineer-Udacity/blob/master/Capstone_Project/Images_Report/VGG16.jpg" />
<p>
  
Some of the drawbacks of the VG166 are:
  - It is slow to train
  - The network architecture weights themselves are quite large
  
 Custom Built Algorithm Model
 
Based on the design of VGG16, we build our model on similar terms. In our model we have 5 convolutional layers followed by the 3 fully connected layers. In our algorithm we use the relu as the activation function and a pooling layer to reduce the dimension of the image data. In the first FCN there are 25088 channels which is converging to 133 in the third FCN. Where 133 belongs to the number of classes from the last fully connected network. We also have dropout layers in our model to make sure the model does not get into overfit condition. 
  
### Benchmark

To tackle such data, we use a benchmark model to build a basic pipeline and a well-versioned model to improvise our classification rate. Such a methodology is carried to tune our model for better prediction of results. The benchmark model helps us to make a comparison and reduce the overfitting or underfitting condition and tune the model for a better outcome. Logistic Regression, KNN are such examples of the benchmark. We can also use the predefined image classifiers such as ImageNet, ResNet, VGG16, etc. to classify our images and later optimize our pipeline for better evaluation of metrics.

In the works of Dog Identification in Kaggle, we see that "Mohamed Sheik Ibrahim" used the VGG19, a predefined base model and carried various processing techniques such as data augmentation to improvise the results obtained from the predefined model. He also used logistic regression to classify the images of dogs and achieved an accuracy of 68%.

Considering another work performed on the same data, using the inception v3, the pre-trained model for image classification, Carey B achieved an accuracy of 83%, which is considered to be a good classification rate based on the performance of the model. 

Using the above understandings, We will be using VGG16 for our data for classifying the breeds of the Dogs, Later we build a Convolutional Neural Network and tune the parameters, Using transfer learning through these models, make a comparative study and analyse the performance of the model.

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing

In the pre-processing part we will be doing following:

- As the size/resolution of the images remain different we need to resize the images to the same scale. Therefore, studying the input requirements of the VGG-16, we know that it requires an image of the size 224. As the reason we are resizing the images of the data to 255 along with a center crop of 224 as needed fr VGG16.
- Also, as mentioned above, we will augment the data to increase the number of images. In data augmentation, we will be increasing the number of images through transformations of the image. The transformations such as rotation, translation, etc are performed so that the number of images are increased.



### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
