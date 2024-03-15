import math
import os

import cv2
import numpy as np
import tensorflow as tf
from albumentations import (
    RandomBrightness, RandomContrast, Sharpen, Emboss, PiecewiseAffine,
    ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, Flip, OneOf, Compose
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

"""
CREATING CUSTOM DATA GENERATOR
"""


class CustomImageGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 img_list=None,
                 labels=None,
                 shuffle=True,
                 batch_size=32,
                 img_size=(224, 224),
                 n_channels=3,
                 aug_prob=0.5,
                 n_classes=4,
                 class_name_map={}
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.aug_prob = aug_prob
        self.aug_func = self.aug_func(self.aug_prob)
        self.img_ids = img_list
        self.labels = labels
        self.class_name_map = class_name_map
        self.n_classes = n_classes
        # self.classes = np.array([self.class_name_map[l] for l in self.labels])
        self.classes = None
        self.indexes = np.arange(len(self.img_ids))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.floor(len(self.img_ids) / self.batch_size)

    def get_image(self, name):
        img = cv2.imread(name)[..., ::-1]
        img = cv2.resize(img, self.img_size)
        return img

    def __data_generation(self, list_id_temp, labels):
        x = np.zeros((self.batch_size, *self.img_size, self.n_channels))
        y = np.zeros(self.batch_size, dtype=float)
        for i, (name, label) in enumerate(zip(list_id_temp, labels)):
            img = self.get_image(name)
            img = self.aug_func(image=img)['image']
            x[i] = img
            y[i] = self.class_name_map[label]
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_ids))
        self.classes = np.array([self.class_name_map[l] for l in self.labels])[:len(self) * self.batch_size]
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        list_id_temp = [self.img_ids[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]
        x, y = self.__data_generation(list_id_temp, labels)
        self.classes[idx * self.batch_size:(idx + 1) * self.batch_size] = y
        # one-hot encode
        y = to_categorical(y, self.n_classes)
        return x, y

    @staticmethod
    def get_label_list(img_dir, class_name_map):
        if class_name_map is None:
            class_name_map = {class_name: en for en, class_name in enumerate(os.listdir(img_dir))}
        labels = []
        img_list = []
        for class_name in os.listdir(img_dir):
            class_dir = os.path.join(img_dir, class_name)
            label = class_name
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img_list.append(img_path)
                labels.append(label)
        return img_list, labels, class_name_map

    @staticmethod
    def aug_func(p=0.5):
        return Compose([Flip(p=0.5),
                        GaussNoise(p=0.2),
                        OneOf([
                            MotionBlur(p=.2),
                            MedianBlur(blur_limit=3, p=.1),
                            Blur(blur_limit=3, p=.1),
                        ], p=0.2),
                        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
                        OneOf([
                            OpticalDistortion(p=0.3),
                            GridDistortion(p=.1),
                            PiecewiseAffine(p=0.3),
                        ], p=0.2),
                        OneOf([
                            Sharpen(),
                            Emboss(),
                            RandomContrast(),
                            RandomBrightness(),
                        ], p=0.3),
                        HueSaturationValue(p=0.3),
                        ], p=p)


# it return list of train, pool, val, test for Active_Learning
def get_loader(train_path='data/data/train',
               val_path='data/data/val',
               test_path='data/data/test',
               pool_size=0.8,
               class_name_map=None
               ):
    train_list, y_train, class_name_map = CustomImageGenerator.get_label_list(train_path, class_name_map)
    # split train and pool
    train_list, pool_list, y_train, y_pool = train_test_split(train_list, y_train, test_size=pool_size, random_state=42)
    val_list, y_val, class_name_map = CustomImageGenerator.get_label_list(val_path, class_name_map)
    test_list, y_test, class_name_map = CustomImageGenerator.get_label_list(test_path, class_name_map)

    train = (train_list, y_train)
    pool = (pool_list, y_pool)
    val = (val_list, y_val)
    test = (test_list, y_test)
    return train, pool, val, test, class_name_map
