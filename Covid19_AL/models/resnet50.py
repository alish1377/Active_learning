"""
This module contains models for resent50
It's but an example. Modify it as you wish.
"""

import tensorflow as tf
from keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.resnet50 import preprocess_input


class Resnet50:
    def __init__(self, image_size, n_classes=4):
        self.input_shape = (image_size[0], image_size[1], 3)
        self.n_classes = n_classes

    def get_model(self) -> Model:

        data_input = Input(self.input_shape)
        x = preprocess_input(data_input)

        # get the pretrained model
        base_model = tf.keras.applications.ResNet50(input_shape=self.input_shape,
                                                    include_top=False,
                                                    weights='imagenet')
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        # pass training False if you are going to fine tune
        x = base_model(x, training=False)
        # x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dropout(.3)(x)
        x = Dense(self.n_classes, activation='softmax')(x)
        model = Model(data_input, x)
        model.summary()
        return model


