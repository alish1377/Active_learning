"""
This module contains models for resent50
It's but an example. Modify it as you wish.
"""

import math
from tensorflow import keras
from tensorflow.keras import layers


class Resnet18:
    def __init__(self, image_size, n_classes=10):
        self.input_shape = (image_size[0], image_size[1], 3)
        self.n_classes = n_classes

    def get_model(self) -> keras.Model:
        kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

        def conv3x3(x, out_planes, stride=1, name=None):
            x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
            return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False,
                                kernel_initializer=kaiming_normal, name=name)(x)


        def basic_block(x, planes, stride=1, downsample=None, name=None):
            identity = x

            out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
            out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
            out = layers.ReLU(name=f'{name}.relu1')(out)

            out = conv3x3(out, planes, name=f'{name}.conv2')
            out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

            if downsample is not None:
                for layer in downsample:
                    identity = layer(identity)

            out = layers.Add(name=f'{name}.add')([identity, out])
            out = layers.ReLU(name=f'{name}.relu2')(out)

            return out


        def make_layer(x, planes, blocks, stride=1, name=None):
            downsample = None
            inplanes = x.shape[3]
            if stride != 1 or inplanes != planes:
                downsample = [
                    layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False,
                                kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
                    layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
                ]

            x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
            for i in range(1, blocks):
                x = basic_block(x, planes, name=f'{name}.{i}')

            return x


        def resnet(x, blocks_per_layer, num_classes=self.n_classes):
            x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
            x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal,
                            name='conv1')(x)
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
            x = layers.ReLU(name='relu1')(x)
            x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
            x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

            x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
            x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
            x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
            x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

            x = layers.GlobalAveragePooling2D(name='avgpool')(x)
            initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
            x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)
            x = layers.Softmax()(x)
            return x


        inputs = keras.Input(shape=self.input_shape)
        outputs = resnet(inputs, [2, 2, 2, 2], self.n_classes)
        model = keras.Model(inputs, outputs)
        return model



