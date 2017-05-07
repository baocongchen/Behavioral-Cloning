# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goal of this project is build an autonomous vehicle by using driving data. To accomplish this goal, I do the following steps:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 2x2 filter sizes and depths between 16 and 48

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using lambda.

#### 2. Attempts to reduce overfitting in the model

I used a dropout layer in order to reduce overfitting. In addition, the model was trained and validated on different data sets to ensure that the model was not overfitting. Number of epochs was limited to 25. The model was also tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Learning rate was set to be 0.0001 which is neither too high nor too low.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving in the middle of the road, teach it to steer back whenever it deviates from the center.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try various combinations of layers.

My first step was to use a convolution neural network model similar to the one used in [Nvidia's End to End Learning for Self-Driving Cars
](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The vehicle lost its direction, so I decided to modify it until it could drive through track 1. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Validation loss remained around 0.02. The simulated self-driving car popped up onto ledges, so I decided to modify the strides in such a way that the ratio between vertical filters and horizontal filters is between 0.2 and 0.5 corresponding to the ratio of the height and width of the input image. I also increased the number of filters to capture more features and used simulator to test the effect of the modification.

To combat the overfitting, I used a dropout layer with a rate of 0.5 as the last layer in the network. I restricted the number of epochs to 25 so that the model would not memorize data but learn from it.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture () consisted of a convolution neural network with the following layers and layer sizes:


|Layer (type)         |            Output Shape     |     Param #  |   Connected to                 |    
====================================================================================================
|cropping2d_1 (Cropping2D)   |     (None, 14, 64, 3)   |  0         |  cropping2d_input_1[0][0]     |    
____________________________________________________________________________________________________
|lambda_1 (Lambda)           |     (None, 14, 64, 3)    | 0         |  cropping2d_1[0][0]           |    
____________________________________________________________________________________________________
|convolution2d_1 (Convolution2D) |  (None, 7, 32, 12)  |   588      |   lambda_1[0][0]              |     
____________________________________________________________________________________________________
|activation_1 (Activation)       | (None, 7, 32, 12)   |  0         |  convolution2d_1[0][0]        |   
____________________________________________________________________________________________________
| convolution2d_2 (Convolution2D) | (None, 4, 16, 24)  |   1176     |   activation_1[0][0]          |    
____________________________________________________________________________________________________
|activation_2 (Activation)       | (None, 4, 16, 24)   |  0         |  convolution2d_2[0][0]        |    
____________________________________________________________________________________________________
|convolution2d_3 (Convolution2D) | (None, 2, 8, 36)    |  3492      |  activation_2[0][0]           |    
____________________________________________________________________________________________________
|flatten_1 (Flatten)            |  (None, 576)        |   0         |  convolution2d_3[0][0]        |    
____________________________________________________________________________________________________
|dense_1 (Dense)                |  (None, 512)        |   295424    |  flatten_1[0][0]              |    
____________________________________________________________________________________________________
|activation_3 (Activation)      |  (None, 512)        |   0         |  dense_1[0][0]                |    
____________________________________________________________________________________________________
|dropout_1 (Dropout)            |  (None, 512)        |   0         |  activation_3[0][0]           |    
____________________________________________________________________________________________________
|dense_2 (Dense)                |  (None, 1)          |   513       |  dropout_1[0][0]              |    
====================================================================================================
Total params: 301,193


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back whenever it deviates from the center. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would help prevent steering bias to the left. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 7491 number of data points. I then preprocessed this data by resizing it to 32x64x3 and cropping 12px from the top.


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was set to 25 to avoid overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
