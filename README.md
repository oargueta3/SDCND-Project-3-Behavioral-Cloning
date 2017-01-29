## Behavioral Cloning: An End-to-End Deep Learning Steering System
### Overview
The objective of this project was to develop an End-to-End Deep Learning algorithm to clone human driving behavior similar to the one presented by NVidia paper titled [End-to-End Deep Learning for Self Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). An End-to-End approach only uses a neural network to determine the desired output of a system. 

INPUT(IMAGES) -----> NEURAL NET -------> OUTPUT(STEERING ANGLE)

For this project, images of a front-facing camera mounted on a car were fed to neural network that determined the corresponding steering angle of the a car. To achieve this, data was collected on a driving simulator that records front camera images and a corresponding steering angle. The camera images were used a training features and steering angles as training labels. The simulator contains two different tracks. Only Track 1 was used to collect training data. After training the neural network, it's performance was validated on Track 1 and the networks's ability to generalize was evaluated on Track 2. The trained network was able to succesfully drive around both tracks without ever leaving the boundaries of the road nor exhibiting 'unsafe' driving behavior.

                                         [diagram of overview of system] 
                                        

### Included Files

This project was written in python using the [KERAS](https://keras.io/) deep learning API. The follwing files were used to create an test the model.

1. `Behavioral Cloning-Oscar Argueta.ipynb`: Used to develop and tune the model. Detail explanations and figures can be found in this notebook 
2. `drive.py`: File that loads the neural net model and communicates with the simulator to excecute predicted angles
3. `model.py`: File that was used to make and train the KERAS model for a modified version of the LeNet. saves the model weights 
4. `model.h5`: File that contains the trained weights
5. `model.json`: File contains a json to
6. `utils.py`