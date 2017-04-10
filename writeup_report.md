#**Behavioral Cloning** 

##Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

My additional goal is to build and train CNN that drives fine on both tracks supplied by simulator, without pre recording any special driving behavior.

[//]: # (Image References)

[track1raw]: ./examples/track1raw.png "Track 1 samples"
[track2raw]: ./examples/track2raw.png "Track 2 samples"
[distort]: ./examples/distort.png "Sample distortion"
[light]: ./examples/light.png "Light random adjustment"
[sidecams]: ./examples/sidecams.png "Side cameras samples"
[cropping]: ./examples/cropping.png "Cropped images"
[hist1]: ./examples/hist1.png "Track 1"
[hist1b]: ./examples/hist1b.png "Track 1 balanced"
[hist2]: ./examples/hist2.png "Track 2"
[hist2b]: ./examples/hist2b.png "Track 2 balanced"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* labutils.py containing routines to generate samples, load\save\threshold\balance dataset etc. 
* init.py containing the script to run initial training
* finetune.py containing the script to run additional training a.k.a. "fine tune"
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* stats.py containing the script to analize datasets, prepare materials for this report, and do some experiments with balancing and thresholding data
* video.mp4 containing a video of driving in autonomous mode on Track 1 for one lap on 30mph
* t1-30mph-b.mp4 same as previous, but driving reverse direction
* t2-20mph-f.mp4 containing a video of driving in autonomous mode on Track 2 for one lap on 20mph
* t1-30mph-b.mp4 same as previous, but driving reverse direction
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model includes convolutional layers with 5x5 and 3x3 filter sizes and depths between 3 and 48 (model.py lines 73-88) 

The model includes RELU layers to introduce nonlinearity (code line 77), and the data is normalized in the model using a Keras lambda layer (code line 71). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 78). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (init.py 42-51).The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 166).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
Data was collected by driving on first track 4-6 laps forward direction, 4-6 laps reverse direction and same on Track 2.
I decided not to record special driving behavior but instead record driving "as is" on both tracks in both directions to experiment how well model can learn from "raw" data. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep it as simple as possible because firstly the only option for training was modest Core 2 Duo CPU 2.2Ghz.

So i started with just one output dense layer to see how far i can go with it. 
I had data from just 2 laps of Track 1, because driving track 2 were practically impossible on my laptop. 
After few tens of trials i implemented few data augmentation utilities (more info in next sections), wich improved performance of my simplest NN ever so it could drive about quarter of lap on first track with speed 9mph.

Then i've decided to try approach similar to [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)  
I thought this model might be appropriate because it is designed to solve very similar task. 

NVidia model was to complex to train so i decided to take only 2 convolutional layers with filter 5x5 and only one with filter 3x3. I "replaced" missing layers with max pooling with size 2x2. 
I didn't want resize my images to not increase decision time during autonomous driving, so the input shape of the model was 65x320x3 - size of images after cropping. I increased sizes of classification layers hoping compensate decreasion in features generalization.
I thought that grayscaling might improve generalization as well, but keeping in mind processing time limit, i decided to add one convolutional layer with filter size 1x1 and depth 3, hoping that NNC will learn to do something similar to grayscaling. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
To combat the overfitting, I added dropout layers after each trainable layers except first one (conv 1x1), and last two classification layers. 

I trained model using Adam optimizer with batch size 128, 100 steps per epoch, 5 epoch. Each epoch time was about 2500sec. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.

Luckily for me to me aws approved access to g2.2xlarge instance, and i got faster laptop.
So first i decided to record more data, and recorded 4-6 laps on first track in both directions and same on Track 2.

I trained model with this data and run simulator on new laptop. The vehicle was able to drive autonomously around the track without leaving the road on 30mph.

But my goal was to develop the model that can drive on both tracks in both directions.
And here was a problem, few tens of trials showed that model couldn't generalize both tracks.

I tried original nvidia model. With same result. This told me that something wrong with data used to learning.
I implemented few tools to analyze datasets, actually angles distributions, and dataset balancing routine.

Then i prepared balanced datasets for both tracks and trained model on mixture of tracks: whole data of track 2, mixed with data from track 1 in proportion 2:1.

At the end of the process, the vehicle is able to drive autonomously around both tracks in both directions without leaving the road.

I noticed few potentially dangerous spots on track 1, and tried different approaches, with different models to generalize better with no luck, so i consider final model as "the best i can for now".

I've recorded video clips of autonomous driving on 
[Track 1 30 mph forward](./video.mp4)
[Track 1 30 mph reverse](./t1-30mph-b.mp4)
[Track 2 20 mph forward](./t2-20mph-f.mp4)
[Track 2 20 mph reverse](./t2-20mph-b.mp4)

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers

| Layer (type)                | Output Shape           | Param #      |
|:---------------------------:|:----------------------:|-------------:|
|lambda_1 (Lambda)            |(None, 65, 320, 3)      |  0           |
|:---------------------------:|:----------------------:|:------------:|
|conv2d_1 (Conv2D)            |(None, 65, 320, 3)      |  12          |
|:---------------------------:|:----------------------:|:------------:|
|conv2d_2 (Conv2D)            |(None, 31, 158, 24)     |  1824        |
|:---------------------------:|:----------------------:|:------------:|
|activation_1 (Activation)    |(None, 31, 158, 24)     |  0           |
|:---------------------------:|:----------------------:|:------------:|
|dropout_1 (Dropout)          |(None, 31, 158, 24)     |  0           |
|:---------------------------:|:----------------------:|:------------:|
|conv2d_3 (Conv2D)            |(None, 14, 77, 36)      |  21636       |
|:---------------------------:|:----------------------:|:------------:|
|activation_2 (Activation)    |(None, 14, 77, 36)      |  0           |
|:---------------------------:|:----------------------:|:------------:|
|max_pooling2d_1 (MaxPooling2 |(None, 7, 38, 36)       |  0           |
|:---------------------------:|:----------------------:|:------------:|
|dropout_2 (Dropout)          |(None, 7, 38, 36)       |  0           |
|:---------------------------:|:----------------------:|:------------:|
|conv2d_4 (Conv2D)            |(None, 5, 36, 48)       |  15600       |
|:---------------------------:|:----------------------:|:------------:|
|activation_3 (Activation)    |(None, 5, 36, 48)       |  0           |
|:---------------------------:|:----------------------:|:------------:|
|max_pooling2d_2 (MaxPooling2 |(None, 2, 18, 48)       |  0           |
|:---------------------------:|:----------------------:|:------------:|
|dropout_3 (Dropout)          |(None, 2, 18, 48)       |  0           |
|:---------------------------:|:----------------------:|:------------:|
|flatten_1 (Flatten)          |(None, 1728)            |  0           |
|:---------------------------:|:----------------------:|:------------:|
|dense_1 (Dense)              |(None, 512)             |  885248      |
|:---------------------------:|:----------------------:|:------------:|
|dropout_4 (Dropout)          |(None, 512)             |  0           |
|:---------------------------:|:----------------------:|:------------:|
|activation_4 (Activation)    |(None, 512)             |  0           |
|:---------------------------:|:----------------------:|:------------:|
|dense_2 (Dense)              |(None, 128)             |  65664       |
|:---------------------------:|:----------------------:|:------------:|
|dropout_5 (Dropout)          |(None, 128)             |  0           |
|:---------------------------:|:----------------------:|:------------:|
|activation_5 (Activation)    |(None, 128)             |  0           |
|:---------------------------:|:----------------------:|:------------:|
|dense_3 (Dense)              |(None, 16)              |  2064        |
|:---------------------------:|:----------------------:|:------------:|
|activation_6 (Activation)    |(None, 16)              |  0           |
|:---------------------------:|:----------------------:|:------------:|
|dense_4 (Dense)              |(None, 1)               |  17          |

Total trainable params: 992065

Model includes 8 trainable layers:  

| Layer (type)                | Specification                    |
|:---------------------------:|:--------------------------------:|
|conv2d_1 (Conv2D)            |filter 1x1, depth 3,  strides 1x1 |
|conv2d_2 (Conv2D)            |filter 5x5, depth 24, strides 2x2 |
|conv2d_3 (Conv2D)            |filter 5x5, depth 36, strides 2x2 |
|conv2d_4 (Conv2D)            |filter 3x3, depth 48, strides 1x1 |
|dense_1 (Dense)              |1728 to 512                       |
|dense_2 (Dense)              |512 to 128                        |
|dense_3 (Dense)              |128 to 16                         |
|dense_4 (Dense)              |16 to 1                           |

####3. Creation of the Training Set & Training Process

As i mentioned above, i didn't record any special driving behavior, instead i recorded 4-6 laps in both directions on both tracks.
Actually track 2 is complex enough and normal driving log has practically all driving examples i can imagine.

Samples recorded on track 1
![alt text][track1raw]

Samples recorded on track 2

![alt text][track2raw]

To augment the dataset, i adopted my 'distort' routine from traffic signs classification project. 
This routine (labutils.py line 44) adds small random rotations, horizontal and vertical shifts to the image, and corrects angle according to horizontal shift applied.

Same samples of Track 2 as above, with random distortion applied:
![alt text][distort]

I also add random light adjustments to images (labutils.py line 66). 

Same samples of Track 1 as above, with random light adjustment applied:
![alt text][light]

I use part of samples with negligible angles as source for new samples with side cameras images and angles adjustment.
I implemnted this by randomly selecting left or right camera image and adding correction to the steering angle (labutils.py line 175).

Examples of generating new samples from sidecamera images:
![alt text][sidecams]

I cropped images to CNN can concentrate on road boarders andcurvature (labutils.py line 25).

Examples of final images:
![alt text][cropping]
 
After the collection process, I had following datasets

Track 1 
----
Log size: 18450
Angles median freq: 24
Angles mean freq: 186
Angles max freq: 6171
Negligible (< 0.0005rad) angles freq: 5621
Valuable left angles freq: 6618 \ 0.5158624990256451
Valuable right angles freq: 6211 \ 0.484137500974355

![alt text][hist1]

Track 2
----
Log size: 12879
Angles median freq: 87
Angles mean freq: 130
Angles max freq: 4648
Negligible (< 0.0005rad) angles freq: 4486
Valuable left angles freq: 4287 \ 0.510782795186465
Valuable right angles freq: 4106 \ 0.4892172048135351

![alt text][hist2]
 
As it is seen from histograms, datasets are greatly unbalanced. 
I balanced datasets by undersampling frequent angles and oversampling rare angles. 
This results to following datasets: 

Track 1 balanced 
----
Log size: 19991
Angles median freq: 164
Angles mean freq: 201
Angles max freq: 965
Negligible angles freq: 529
Valuable left angles freq: 10062 \ 0.5170075017983763
Valuable right angles freq: 9400 \ 0.48299249820162365

![alt text][hist1b]

Track 2 balanced 
----
Log size: 12859
Angles median freq: 124
Angles mean freq: 129
Angles max freq: 484
Negligible angles freq: 358
Valuable left angles freq: 6385 \ 0.5107591392688585
Valuable right angles freq: 6116 \ 0.48924086073114154

![alt text][hist2b]

I trained final model on mixture of whole balanced data set of Track 2 and balanced data set of Track 1 with proportion 2:1 (model.py line 169), so the final data set contained 19289 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used generator (model.py 14) for partially lodding and adding random distortion to this data.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by absence of validation loss improvement with bigger number of epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
