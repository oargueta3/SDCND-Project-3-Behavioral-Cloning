## Behavioral Cloning: An End-to-End Deep Learning Steering System
### Overview
The objective of this project was to develop an End-to-End Deep Learning algorithm to clone human driving behavior similar to the one presented by NVidia paper titled [End-to-End Deep Learning for Self Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). An End-to-End approach only uses a neural network to determine the desired output of a system. 

                          [INPUT(Center Image) -----> NEURAL NET -------> OUTPUT(Steering Angle)]

For this project images from front-facing camera mounted on a car were fed to neural network that predicted the corresponding steering angle of the a car. To achieve this, data was collected on a driving simulator that records front camera images and a corresponding steering angle. The camera images were used a training features and steering angles as training labels. The simulator contains two different tracks. Only Track 1 was used to collect training data. After training the neural network, it's performance was validated on Track 1 and the networks's ability to generalize was evaluated on Track 2. The trained network was able to succesfully drive around both tracks without ever leaving the boundaries of the road nor exhibiting 'unsafe' driving behavior.
    
  **Click here to see results**

### Included Files

This project was written in Python using the [KERAS](https://keras.io/) deep learning API. The follwing files were used to create an test the model.

1. `Behavioral Cloning-Oscar Argueta.ipynb`: Used to develop and tune the model. Detailed explanations and figures can be found in this jupyter notebook 
2. `drive.py`: File that loads the neural net model and communicates with the simulator to excecute predicted steering angles
3. `model.py`: File that was used to make and train a KERAS model for a modified version of the LeNet Architecture. Saves model weights 
4. `model.h5`: File that contains the trained weights
5. `model.json`: File contains a JSON representation of the neural net that can be used to predict steering angles in realtime
6. `utils.py`: File contains helper methods to process and batch data


## Data Collection and Preparation

Driving data was collected over 2 separate recording sessions on **TRACK 1**. The keyboard arrow keys were was used to control the car, but unfortunately produced unreliable steering data. The data recorded was comoposed as follows:

**Data Set 1**: 4 laps of centered driving  

**Data Set 2**: 4 laps of centered driving (*oppossite direction*)

Each data set has an associated ``driving_log.csv`` file that contains the recorded center, left, and right images file paths, throttle, steering angle, and speed. In the real word it is not safe and legally practical to simulate the off center shifts required to train a car to recuperate from such shift. To simulate such behavior the Left and Right images are used. To train the vehicle to recuperate from these we assign a correction angle, the angle required for the car to recenter on the road, to each left and right image. The correction angle was empirically selected to be 0.25, and the corresponding steering angles are calculated as follows:

[left image angle = steering angle + 0.25]
[right image angle = steering angle - 0.25]

![Left-Center-Image](readme_images/image_1.png 
              
              
The model was succesfully trained with this data, but given the poor interface for recording (arrow keys), the data set provided by Udacity was used to train the final model as it produced better results. To compare why one data set produces better results a histogram of both data sets were plotted

## Neural Net Architecture

## Training

## Results
videos

## How to run


Thank you!