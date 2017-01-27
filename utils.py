import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_data_logs(PATHS):
    """ Takes in a list of file paths for the recorded training data
        and returns a data frame with the information from all 'drive_log.csv'
        files. Adds 2 additional columns or left and right image steering angle:
        
        left image steer angle = center image steer angle + angle_delta
        right image steer angle = center image steer angle - angle_delta
        
        angle_delta: the angle required to steer the car back to the center """
    column_names = ['Center Image','Left Image', 'Right Image',
                    'Steering Angle', 'Throttle', 'Break', 'Speed']
    full_data_log = pd.DataFrame(columns=column_names)
                    
    # Load and merge data logs
    for path in PATHS:
        data_log = pd.read_csv(path + 'driving_log.csv')
        data_log.columns = column_names
        data_log['Center Image'] = path + data_log['Center Image'].apply(str.strip)
        data_log['Left Image'] = path + data_log['Left Image'].apply(str.strip)
        data_log['Right Image'] = path + data_log['Right Image'].apply(str.strip)
        full_data_log = pd.concat([full_data_log, data_log])

    # Remove data with no throttle
    full_data_log = full_data_log[full_data_log['Throttle'] > 0].reset_index(drop=True)
    
    # Add left and right image steering angles with corresponding angle correction
    angle_delta = 0.25
    full_data_log['Left Steering Angle'] = full_data_log['Steering Angle'] + angle_delta
    full_data_log['Right Steering Angle'] = full_data_log['Steering Angle'] - angle_delta  
    return full_data_log


def resample_zeros(data_set, fraction=0.5):
    """Return a new data frame with all non-zero angle samples
        concatenated with a random subset of zero angle data."""
    data_set_non_zero = data_set[data_set["Steering Angle"] != 0.0]
    data_set_zero = data_set[data_set["Steering Angle"] == 0.0].sample(frac=fraction, random_state= 8809)
    return pd.concat([data_set_non_zero, data_set_zero]).reset_index(drop=True)


def preprocess_image(image, new_row_size, new_col_size):
    """ Takes in an image(numpy array) and converts it HSV color space 
        only keeping the S channel. Resizes the image to new_row_size x new_col_size,
        and crops image so that the part horizon and the hood of the car are gone."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1]
    image = image[40:138]                                  # crop car hood and part of the horizon
    image = cv2.resize(image,(new_row_size, new_col_size), interpolation=cv2.INTER_AREA)
    return image


def load_data_from_data_log(data_log, new_row_size=64, new_col_size=64):
    """ Takes in a dataframe containing training data information and
        returns preprocessed image data and labels in numpy arrays. Images are resized
        to new_row_size x new_col_size"""
    num_samples = len(data_log)
    label_data = np.zeros(num_samples*3, dtype=np.float32)
    image_data = np.zeros((num_samples*3, new_row_size, new_col_size), dtype=np.float32)
    
    # Load center image data
    for index in range(0, num_samples):
        image_c = cv2.imread(data_log['Center Image'][index])
        label_data[index] = data_log['Steering Angle'][index]
        image_data[index] = preprocess_image(image_c, new_row_size, new_col_size)
    
    # Load left image data
    for read_index, write_index in zip(range(num_samples), range(num_samples, num_samples*2)):
        image_l = cv2.imread(data_log['Left Image'][read_index])
        label_data[write_index] = data_log['Left Steering Angle'][read_index]
        image_data[write_index] = preprocess_image(image_l, new_row_size, new_col_size)
    
    # Load right image data
    for read_index, write_index in zip(range(num_samples), range(num_samples*2, num_samples*3)):
        image_r = cv2.imread(data_log['Right Image'][read_index])
        label_data[write_index] = data_log['Right Steering Angle'][read_index]
        image_data[write_index] = preprocess_image(image_r, new_row_size, new_col_size)
    
    # Add additional dimension to image data - e.g (None,64,64) to (None,64,64,1)
    # in order to feed itto the neural net model
    image_data = np.reshape(image_data, (image_data.shape[0],new_row_size,
                                         new_col_size,-1))
    return image_data, label_data