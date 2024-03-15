import sys

sys.path.insert(0, '..')
from active_learning.data_loader import CustomImageGenerator
from models import load_model
import cv2
from tensorflow.keras.utils import to_categorical
import numpy as np


def retrain(img, true_label, iter):
    img = cv2.imread(img)
    X, y_labels = np.array(img), np.array(true_label)
    for x in range(iter):
        new_img = CustomImageGenerator.aug_func(p=1)(image=img)['image']
        X.append(new_img, axis=0)
        y_labels.append(true_label, axis=0)
    print(y_labels)
    model = load_model(model_name='resnet50', image_size=(240, 320, 3))
    model.load_weights('../' + 'weights/mymodel.h5', {'image_size': (240, 320, 3)})
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y_labels,
              epochs=5, batch_size=32)
    model.save('../weights/retrained_model.h5')


y = to_categorical([3], num_classes=4)
retrain('../streamlit/files/random-images/_0_5239.jpeg', y, 300)
