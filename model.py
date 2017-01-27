import json
from utils import *
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ELU, Flatten
from keras.optimizers import Adam

#=====================================
# Define  Neural Network Architecture
#=====================================
def LeNet_Drives():
    # model paramaters
    input_shape = (64, 64, 1)
    filter_size = 3
    pool_size = (4,4)
    
    # Create model
    model = Sequential()
    
    # Normalization layer
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=input_shape))
    
    # Convolution 1
    model.add(Convolution2D(6, filter_size, filter_size, init='he_normal', border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pool_size))
    
    # Convolution 2
    model.add(Convolution2D(16, filter_size, filter_size, init='he_normal', border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=pool_size))
    
    # Flatten
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(120, init='he_normal'))      # fc1
    model.add(ELU())
    model.add(Dense(84, init='he_normal'))       # fc2
    model.add(ELU())
    model.add(Dense(1, init='he_normal'))        # output
    return model

def save_model(model, json_file, weights_file):
    # Save model architecture
    with open(json_file,'w' ) as f:
        json.dump(model.to_json(), f)
    
    # Save model weights
    model.save_weights(weights_file)


if __name__ == "__main__":
    PATHS = ['data/']
    # new image dimesions for neural net
    new_col_size = 64
    new_row_size = 64
    
    #=========================================
    # Prepare data to train the Neural Network
    #=========================================
    print("Preprocessing Data....")
    print(" ")
    
    # Load data logs into a pandas dataframe
    full_data_log = load_data_logs(PATHS)
    
    # Down sample 60% of the zero angle data
    re_full_data_log = resample_zeros(full_data_log, 0.40)
    
    # Preprocess images and load train and test features into numpy arrays
    train_features, train_labels = load_data_from_data_log(re_full_data_log, new_row_size, new_col_size)
    
    # Create validation set with a 90-10 split
    train_features, train_labels = shuffle(train_features, train_labels)
    train_features, valid_features, train_labels, valid_labels = train_test_split(
                                                                              train_features,
                                                                              train_labels,
                                                                              test_size=0.10,
                                                                            random_state=832289)
    #=============================================
    # Train and Save the Neural Network Parameters
    #=============================================
    print("Training Model....")
    print(" ")
    model = LeNet_Drives()

    # Compile model using mean square error as the loss parameter
    # Optimizer - Adam: values represent default settings
    adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse', metric=['accuracy'])

    # Train model for 10 Epochs
    history = model.fit(train_features, train_labels,
                   batch_size=128, nb_epoch=10, shuffle=True,
                   validation_data=(valid_features, valid_labels))

    # Save model architecture and weights
    save_model(model, 'model.json', 'model.h5')
    print(" ")
    print("Model saved....")
