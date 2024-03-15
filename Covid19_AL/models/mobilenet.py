import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dropout


class MobileNet:
    def __init__(self, image_size=(200, 200), n_classes=4, channels=3, **kwargs):
        self.input_shape = (image_size[0], image_size[1], channels)
        self.n_classes = n_classes

    def get_model(self) -> Model:
        # define unput and preprocess
        inputs = Input(self.input_shape)
        x = preprocess_input(inputs)

        # imports the mobilenet model and discards the last 1000 neuron layer.
        base_model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet',
                                                               input_shape=self.input_shape,
                                                               include_top=False)
        # freeze the model
        base_model.trainable = False

        x = base_model(x)
        # reduce dimension
        x = GlobalAveragePooling2D()(x)
        # we add dense layers so that the model can learn more complex functions and classify for better results.
        x = Dense(1024, activation='relu')(x)
        x = Dropout(rate=0.3)(x)
        x = Dense(1024, activation='relu')(x)  # dense layer 2
        x = Dropout(rate=0.3)(x)
        x = Dense(512, activation='relu')(x)  # dense layer 3
        x = Dense(self.n_classes, activation='softmax')(x)  # final layer with softmax activation

        model = Model(inputs, x)
        return model
