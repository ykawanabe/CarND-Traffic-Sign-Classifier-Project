#**Traffic Sign Recognition**

This is Project 2 Traffic Sign Recognition of [Self-Driving car course on Udacity](https://www.udacity.com/drive).

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/image_preprocessing.png "Preprocessing"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/newImages.png "Traffic Signs"
[image9]: ./examples/signs.png "Sign images"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/yusuke-kawanabe/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data are distributed. There are distribution is a little bit unbalanced by the order of 10.

![alt text][image1]

They are sample images for each classes.

![alt text][image9]

Numbers corresponds to traffic sign names as shown below.

| Index | Sign Name |
|:-----|:-----------|
| 0 | Speed limit (20km/h) |
| 1 | Speed limit (30km/h) |
| 2 | Speed limit (50km/h) |
| 3 | Speed limit (60km/h) |
| 4 | Speed limit (70km/h) |
| 5 | Speed limit (80km/h) |
| 6 | End of speed limit (80km/h) |
| 7 | Speed limit (100km/h) |
| 8 | Speed limit (120km/h) |
| 9 | No passing |
| 10 | No passing for vehicles over 3.5 metric tons |
| 11 | Right-of-way at the next intersection |
| 12 | Priority road |
| 13 | Yield |
| 14 | Stop |
| 15 | No vehicles |
| 16 | Vehicles over 3.5 metric tons prohibited |
| 17 | No entry |
| 18 | General caution |
| 19 | Dangerous curve to the left |
| 20 | Dangerous curve to the right |
| 21 | Double curve |
| 22 | Bumpy road |
| 23 | Slippery road |
| 24 | Road narrows on the right |
| 25 | Road work |
| 26 | Traffic signals |
| 27 | Pedestrians |
| 28 | Children crossing |
| 29 | Bicycles crossing |
| 30 | Beware of ice/snow |
| 31 | Wild animals crossing |
| 32 | End of all speed and passing limits |
| 33 | Turn right ahead |
| 34 | Turn left ahead |
| 35 | Ahead only |
| 36 | Go straight or right |
| 37 | Go straight or left |
| 38 | Keep right |
| 39 | Keep left |
| 40 | Roundabout mandatory |
| 41 | End of no passing |
| 42 | End of no passing by vehicles over 3.5 metric tons |

###Design and Test a Model Architecture

#### Preprocess

![alt text][image2]

The image above illustrates image preprocess techniques that I tried and applied.

As a first step, I decided to sharpen the picture because edges contain more information as signs are designed to stand out from the background to avoid horrible car accidents.
Here is an example of a traffic sign image before and after sharpening.

Secondly, I applied histogram equalization because different training data has different contrast.

As the last step, I normalized the image data because good data set should have mean of zero and equal variance across variables.

I decided not to use grayscale as color have important data in traffic signs.

#### Model architecture

My final model consisted of the following layers:

| Layer                 |     Description                                |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x3 RGB image                               |
| Convolution 3x3         | 1x1 stride, valid padding, outputs 32x32x16     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 15x15x64                 |
| Convolution 3x3         | 1x1 stride, valid padding, outputs 12x12x32     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 6x6x32                 |
| Flatten        | 1152 outputs               |
| Fully connected                | outputs 120 |
| RELU |  |
| Dropout | Dropout rate 0.5 |
| Fully connected | outputs 84 |
| RELU |  |
| Dropout | Dropout rate 0.5 |
| Fully connected | outputs 43 |


To train the model, I used an LeNet architecture. As the number of class is bigger than the LeNet lab problem, I decided to widen the neural network so that the network can hold more complex information.

To avoid overfitting, I decided to use dropout.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.965
* test set accuracy of 0.938004732131958

I chose the same LeNet architecture as [the LeNet Lab](https://github.com/udacity/CarND-LeNet-Lab) at first and it required more than 80 epochs to reach to the accuracy 0.93. This can lead to overfitting as we are feeding the same data again and again.
I decided to widen the neural network because the traffic sign classifier problem has more classes and needs more information in the network.

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4]

The third and firth image might be difficult to classify because the sign is relatively small and as the result blur in the picutre.

Here are the results of the prediction:

| Image                    |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| Children crossing | Children crossing             |
| Stop                 | Stop                                         |
| Speed limit (30km/h)                    | Speed limit (50km/h)   |
| No passing for vehicles over 3.5 metric tons | No entry |
| Roundabout mandatory  | Roundabout mandatory |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

The code for making predictions on my final model is located in the cells below "Output Top 5 Softmax Probabilities For Each Image Found on the Web" title in the Ipython notebook.

For the first image, the model is relatively sure that this is a children crossing sign (probability of 0.85), and the image does contain a children crossing sign. The top five soft max probabilities were

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .85                     | Children crossing |
| .05                     | Beware of ice/snow                                         |
| .02                    | Road narrows on the right                                            |
| .02                      | General caution                                     |
| .02                    | Pedestrians                                  |


For the second image the model is sure that this is a stop sign.

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.00                     | Stop |
| .00                     | Yield |
| .00                    | No entry |
| .00                      | Ahead only  |
| .00                    | Priority road |

and so forth.
