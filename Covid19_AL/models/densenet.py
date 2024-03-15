import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Flatten
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dropout


class DenseNet:
    def __init__(self, image_size=(200, 200), n_classes=4, channels=3, **kwargs):
        self.input_shape = (image_size[0], image_size[1], channels)
        self.n_classes = n_classes

    def get_model(self) -> Model:
        # define input and preprocess
        inputs = Input(self.input_shape)
        x = preprocess_input(inputs)

        base_model = tf.keras.applications.DenseNet121(weights='imagenet',
                                                       input_shape=self.input_shape,
                                                       include_top=False)
        # freeze the model
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        x = base_model(x)
        # reduce dimension
        x = GlobalAveragePooling2D()(x)
       # x = Dropout(.6)(x)
        #x = Dense(1024, activation='relu')(x)
        x = Dropout(.6)(x)
        x = Dense(self.n_classes, activation='softmax')(x)

        model = Model(inputs, x)
        model.summary()

        return model
