## Project: Build a Traffic Sign Recognition Program

Overview
---
In this project, I trained and validated a model to classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Following training I then tried the model on images of German traffic signs found on the web.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/distribution.png "Distribution of Classes"
[image3]: ./examples/grayscale.png "Grayscale images"
[image4]: ./examples/training_curves.png "Training Curves"
[image5]: ./examples/webSigns.png "Signs from the web"
[image6]: ./examples/classifications.png "Classifications"
[image7]: ./examples/conv1.png "Convolution layer 1"
[image8]: ./examples/conv2.png "Convolution layer 2"

### Data Set Summary & Exploration

#### 1. Summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Samples of the RGB images from the training set are shown below:

![alt text][image1]

The distribution of classes is shown in the following graph. The most important observation is that the distributions are fairly similar.

![alt text][image2]

### Design and Test of Model Architecture

#### 1. Preprocessing

As a first step, I decided to convert the images to grayscale simplify the complexity of the neural network

Examples of traffic sign images after grayscaling are shown below.

![alt text][image3]

As a last step, I normalized the image data to aid in the stochastic gradient descent.

I considered adding additional data by  using OpenCV functions such as the rotate method on the training data. However, the inital testing of the neural network showed acceptable performance so I chose not to augment the training set in this way.

#### 2. The model architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 
