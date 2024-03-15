from .query_strategies.uncertainty_sampling import UncertaintySampling
from .data_loader import CustomImageGenerator

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model


class AL:
    def __init__(self,
                 model,
                 x_train,
                 y_train,
                 x_pool,
                 y_pool,
                 x_val,
                 y_val,
                 batch_size=32,
                 img_size=(224, 224),
                 init_epochs=1,
                 query_strategy="lc"):
        self.x, self.y = None, None
        self.model: Model = model
        self.x_pool = x_pool
        self.y_pool = y_pool
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.img_size = img_size,
        self.method = query_strategy
        self.query_func = self.query()
        self.train(x_train, y_train, epochs=init_epochs, add=False, remove=False)

    def query(self):
        if self.method == 'lc':  # least confident
            method = UncertaintySampling(self.method)
            return method.make_query

    def _add_samples(self, x, y):
        self.x = np.concatenate([self.x, x])
        self.y = np.concatenate([self.y, y])

    def _remove(self, query_index):
        self.x_pool = np.delete(self.x_pool, query_index, axis=0)
        self.y_pool = np.delete(self.y_pool, query_index, axis=0)

    @staticmethod
    def _aug_func(image):
        return CustomImageGenerator.aug_func(p=1)(image=image)['image']

    def _get_oversample(self, x, y, factor):
        """
        Augments new images from the input ones

        Parameters
        ----------
        x: The input images
        y: The input labels
        factor: Defines the output images. Number of generated image equal to len(x) * factor

        Returns
        -------
        The newly generated/augmented images
        """
        new_x, new_y = [], []
        for i, (img, label) in enumerate(zip(x, y)):
            for j in range(factor):
                new_img = self._aug_func(image=img)
                new_x.append(new_img)
                new_y.append(label)

        return new_x, new_y

    @staticmethod
    def break_callbacks():
        early_stopping = EarlyStopping(patience=1)
        return [early_stopping]

    def train(self, x, y, epochs=1, add=True, remove=True, query_index=None, oversampling_factor=5, only_new=False):

        oversampled_list, y_oversampled = [], []
        # Only augmented images
        # to oversample & augment images
        if oversampling_factor > 1:
            oversampled_list, y_oversampled = self._get_oversample(x=x,
                                                                   y=y,
                                                                   factor=oversampling_factor - 1)

        if only_new:
            x_final, y_final = oversampled_list + x, y_oversampled + y
        else:
            x_final, y_final = self.x_train + oversampled_list + x, self.y_train + y_oversampled + y

        self.model.fit(x_final, y_final, validation_data=(self.x_val, self.y_val),
                       epochs=epochs, callbacks=self.break_callbacks(), batch_size=self.batch_size)

        # to remove augmented images
        if oversampling_factor > 1:
            del oversampled_list, y_oversampled, x_final, y_final

        if add:
            self._add_samples(x, y)

        if remove and query_index is not None:
            self._remove(query_index)
        return self

    def evaluate(self):
        if self.x_val is not None and self.y_val is not None:
            results = self.model.evaluate(self.x_val, self.y_val)
            print(f'print validation-loss: {results[0]:.4f} validation-acc: {results[1]:.2f}')
            return results

    def fit(self, each_query_epochs, query_size=None, only_new=False, n_queries=None, oversampling_factor=5):
        query_size = self.batch_size if query_size is None else query_size
        n_queries = len(self.x_pool) // query_size if n_queries is None else n_queries

        for idx in range(n_queries):
            print('Query %d/%d' % (idx + 1, n_queries))
            query_index = self.query_func(self.model, self.x_pool, query_batch_size=query_size, verbose=1)
            if len(query_index) < self.batch_size:
                return self
            x, y = self.x_pool[query_index], self.y_pool[query_index]
            self.train(x, y, epochs=each_query_epochs, query_index=query_index,
                       only_new=only_new, oversampling_factor=oversampling_factor)
        return self
