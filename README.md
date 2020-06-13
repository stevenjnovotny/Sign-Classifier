## Project: Build a Traffic Sign Recognition Program

Overview
---
In this project, I trained and validated a model to classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Following training, I applied the model to images of German traffic signs found on the web.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualize the convolutional layers

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

As a first step, I decided to convert the images to grayscale to simplify the complexity of the neural network

Examples of traffic sign images after grayscaling are shown below.

![alt text][image3]

As a last step, I normalized the image data to aid in the stochastic gradient descent.

I considered adding additional data by  using OpenCV functions such as the rotate method on the training data. However, the initial testing of the neural network showed acceptable performance so I chose not to augment the training set in this way.

#### 2. The model architecture 

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 Grayscale image                       | 
| Convolution 5x5       | 1x1 stride, 'VALID' padding, outputs 28x28x12 |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 14x14x12                 |
| Convolution 5x5       | 1x1 stride, 'VALID' padding, outputs 10x10x32 |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, output 5x5x32                     |
| Flattening            | Output 800                                    |
| Fully connected       | Output 120                                    |
| RELU                  |                                               |
| Fully connected       | Output 84                                     |
| RELU                  |                                               |
| Dropout               | prob = 0.5 for train; 1.0 for test            |
| Fully connected       | Output 43 (number of classes)                 |

#### 3. Training the Model.

To train the model, I used a learning rate of 0.0005, batch size of 128, and 25 epochs. These parameters were determined through multiple runs and evaluation of the learning curve as well as the resulting accuracy. Dropout was added to address signs of overfitting I observed in the training curves. 

As an optimizer, I used the AdamOptimizer as it was based on stochastic gradient descent. As a loss function, I chose cross entropy as implemented with softmax_cross_entropy_with_logits().

The training curves are shown below:

![alt text][image4]

#### 4. Approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My approach was to start with the LeNet-5 architecture and apply that to grayscale and normalized versions of the data. The LeNet architecture was chosen because the street signs seemed comparable to the characters used in the paper. The initial results gave me validation accuracy of approximately 0.91. Based on a need to improve that result and because I saw signs of overfitting in the training curves (validation loss increasing while training loss decreased) I chose to do the following:
* increase the number of filters (features) in the convolutional layers
* decrease the learning rate
* increase the number of epochs
* add a dropout function following the second fully connected layer

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.95 
* test set accuracy of 0.936

### Test the Model on New Images

#### 1. German traffic signs found on the web.

I found six German street signs on the internet. They are varied in lighting conditions, background, and clutter. In addition, most have only symbols while one has text. The signs are shown below.

![alt text][image5]

Applying the model to these signs provides the following classifications. Shown in the graphs are the top five most probable classifications (as determined through the application of a softmax function). 

![alt text][image6]

The model correctly identified all six signs, but mistook the 50 km/hr for the 30 km/hr. This suggests that the model is focusing on aspects of the signs' morphology as opposed to the details and structure of the text characters. This could also be a result of the granulation and pixilation of the input images. Additionally, the following section identifies the features that cause activation. Perhaps limiting the max pooling would help with the finer details.

### Visualizing the Neural Network 

The following shows a visualization of the neural network's first two layers. The first image shows the activation features after the first convolutional layer and max pooling, and the second image shows the same for the second convolutional layer.

![alt text][image7]

![alt text][image8]

## Conclusions

I beleive bettere refinement of the model would improve the performance of the classifier. Specifically, increasing the number of nuerons in the hidden layers could have an impact on the fine details (such as text) that the model is currently missing. It also seems  that adjusting the strides in the max pooling could keep more information.

In addition, I believe the generation of additional data through rotation, scaling and noise addition could craete a more robust model.