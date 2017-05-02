import os
os.environ["THEANO_FLAGS"] = "device=gpu0"

from keras.models import Model
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, Activation
from keras.layers import Input, Convolution2D, Dense
from PIL import Image
import numpy as np

def FullConnected(weights_path=''):

    input_shape = (1, 4096)
    img_input = Input(shape=input_shape)

    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(4096, activation='relu', name='fc3')(x)
    x = Dense(2, activation='softmax', name='fc4')(x)

    # Create model
    model = Model(img_input, x)


    if weights_path:
        model.load_weights(weights_path)

    return model

