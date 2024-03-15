from .query_strategies.Strategies import Strategies
from .data_loader import CustomImageGenerator
from deep_utils import remove_create
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
import cv2
import os
import shutil
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


class AL:
    def __init__(self,
                 model,
                 train_list,
                 y_train,
                 pool_list,
                 y_pool,
                 val_list,
                 y_val,
                 df,
                 df_best,
                 batch_size=32,
                 img_size=(224, 224),
                 init_epochs=1,
                 gen_args=None,
                 query_strategy="lc",
                 model_path = None,
                 initial_model_path=None,
                 initial_history = None,
                 temp_dir='temp'):
        """

        Parameters
        ----------
        model
        train_list
        y_train
        pool_list
        y_pool
        val_list
        y_val
        batch_size
        img_size
        init_epochs
        gen_args
        query_strategy
        temp_dir: The directory for the generated temporary images
        """
        self.model: Model = model
        self.pool_list = pool_list
        self.y_pool = y_pool
        self.train_list: list = []
        self.y_train: list = []
        self.val_list = val_list
        self.y_val = y_val
        self.df = df
        self.df_best = df_best
        self.batch_size = batch_size
        self.img_size = img_size
        self.gen_arg = gen_args  # arguments to make CustomDataGenerator
        self.method = query_strategy
        self.temp_dir = temp_dir
        self.model_path = model_path
        self.initial_model_path = initial_model_path
        self.initial_history = initial_history
        self.query_func = self.query()
        self.val_gen = CustomImageGenerator(img_list=self.val_list,
                                            labels=self.y_val,
                                            **self.gen_arg)

        print('Training the initial samples')

        self.train(train_list, y_train, epochs=init_epochs, add=True, remove=False, factor_oversampling=1, is_init=True)

    def query(self):  # least confident
        method = Strategies(self.method)
        return method.make_query

    def _add_samples(self, x, y):
        self.train_list = self.train_list + x
        self.y_train = self.y_train + y

    @staticmethod
    def _remove_from_list(lst, query_index):
        """
        Parameters
        ----------
        lst: The input list
        query_index: The removing indices

        Returns
        -------
        Returns the final list after removing the specific indices.
        """
        indices = sorted(query_index, reverse=True)
        for idx in indices:
            lst.pop(idx)
        return lst

    def _remove(self, query_index):
        """
        Removes the queried items from the pool

        Parameters
        ----------
        query_index: The indices of the query images

        Returns
        -------
        None
        """
        self.pool_list = self._remove_from_list(self.pool_list, query_index)
        self.y_pool = self._remove_from_list(self.y_pool, query_index)

    @staticmethod
    def _aug_func(image):
        return CustomImageGenerator.aug_func(p=1)(image=image)['image']

    def _oversample(self, x_list, y, factor, temp_dir='temp'):
        """

        Parameters
        ----------
        x_list: the input image's address
        y: the labels of the input images
        factor: Defines the number of output images. Number of generated image equal to len(x) * factor


        Returns
        -------
        The address to the newly generated images and the corresponding labels.
        """
        # make a temp file
        remove_create(temp_dir)
        new_x_list, new_y = [], []
        if factor <= 0:
            return new_x_list, new_y
        for i, (name, label) in enumerate(zip(x_list, y)):
            img = cv2.imread(name)[..., ::-1]
            img = cv2.resize(img, self.img_size)

            # saving new augmented images to new path
            name = os.path.split(name)[1]
            name, ext = os.path.splitext(name)

            for j in range(factor):
                # augmenting img
                new_img = self._aug_func(image=img)
                new_path = os.path.join(temp_dir, f'{name}_{j}' + ext)
                cv2.imwrite(new_path, new_img)
                new_x_list.append(new_path)
                new_y.append(label)

        return new_x_list, new_y

    def _remove_oversample(self, factor: int):
        """

        Parameters
        ----------
        factor: This is a flag to determine whether to remove the temp_dir or not

        """
        if factor > 0:
            shutil.rmtree(self.temp_dir)

    @staticmethod
    def callbacks(model_path):
        early_stopping = EarlyStopping(patience=1)
        check_point = ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True)
        return [early_stopping, check_point]

    def train(self, x_list, y, epochs, add=True, remove=True, query_index=None, factor_oversampling=5,
              only_new=False, save_model=False, is_init=False):
        oversampled_list, y_oversampled = self._oversample(x_list=x_list, y=y, factor=factor_oversampling - 1)

        if only_new:
            final_list, y_final = oversampled_list + x_list, y_oversampled + y
        else:
            final_list, y_final = self.train_list + oversampled_list + x_list, self.y_train + y_oversampled + y

        train_gen = CustomImageGenerator(img_list=final_list, labels=y_final, **self.gen_arg)
        print()

        if is_init:
            if os.path.exists(self.initial_model_path):
                model = load_model(self.initial_model_path)
                history = np.load(self.initial_history,allow_pickle='TRUE').item()
                print(history.history)
            else:
                checkpoint = ModelCheckpoint(self.initial_model_path, monitor='val_acc', save_best_only=True)
                history = self.model.fit(train_gen, validation_data=self.val_gen,
                                        epochs=epochs, callbacks=[checkpoint], batch_size=self.batch_size, workers=4)
                self.model = load_model(self.initial_model_path)
                np.save(self.initial_history ,history)
                print(history.history)
        else:
            history = self.model.fit(train_gen, validation_data=self.val_gen,
                                        epochs=epochs, callbacks=self.callbacks(self.model_path), batch_size=self.batch_size, workers=4)
            self.model = load_model(self.model_path)

        if save_model:
            self.model.save('../weights/retrained_model.h5')
        # to remove augmented images

        self._remove_oversample(factor_oversampling)
        # del oversampled_list, y_oversampled, final_list, y_final

        if add:
            self._add_samples(x_list, y)

        if remove and query_index is not None:
            self._remove(query_index)
        self.df = pd.concat([self.df, pd.DataFrame(history.history)], ignore_index=True)
        for k, v in history.history.items():
          if 'loss' in k:
            history.history[k] = [min(v)]
          if 'acc' in k:
            history.history[k] = [max(v)]
        self.df_best = pd.concat([self.df_best, pd.DataFrame(history.history)], ignore_index=True)

    def evaluate(self):
        if self.val_list is not None:
            results = self.model.evaluate(self.val_gen)
            print(f'print validation-loss: {results[0]:.4f} validation-acc: {results[1]:.2f}')
            return results

    def fit(self, each_query_epochs, query_size=100, only_new=True, n_queries=None, factor_oversampling=5
            , uncertainty_rate=1):
        query_size = self.batch_size if query_size is None else query_size
        n_queries = len(self.pool_list) // query_size if n_queries is None else n_queries
        for idx in range(n_queries):
            print('Query %d/%d' % (idx + 1, n_queries))
            pool_gen = CustomImageGenerator(img_list=self.pool_list, labels=self.y_pool, **self.gen_arg)
            query_index = self.query_func(self.model, pool_gen, query_batch_size=query_size,
                                          uncertain_size=uncertainty_rate, verbose=1)
            if len(query_index) < self.batch_size:
                return self
            x_list = [self.pool_list[i] for i in query_index]
            y = [self.y_pool[i] for i in query_index]

            self.train(x_list, y, epochs=each_query_epochs, query_index=query_index,
                       only_new=only_new, factor_oversampling=factor_oversampling, is_init=False)
        file_name = 'metric_logs' + "_" + str(datetime.now().date()) + "_" + str(datetime.now().time())
        file_name = file_name.replace(':', '.')
        self.df.to_csv(file_name)
        self.df_best.to_csv("best_" + file_name)
        self.plotting(range(len(self.df_best.index)), self.df_best['loss'].tolist(), self.df_best['val_loss'].tolist(), 'loss')
        self.plotting(range(len(self.df_best.index)), self.df_best['acc'].tolist(), self.df_best['val_acc'].tolist(), 'acc')

    def plotting(self, x, y, val_y, metric):
        if self.method == 'random':
            np.save(metric+'_train', y)
            np.save(metric+'_val', val_y)
            plt.plot(x, y, label='random train')
            plt.plot(x, val_y, label='random val')
            plt.xlabel("number of queries")
            plt.ylabel(metric)
            plt.show()
        else:
            try:
                y_random = np.load(metric + '_train.npy')
                y_rand_val = np.load(metric+'_val.npy')
                plt.plot(x, y_random, label='random train')
                plt.plot(x, y_rand_val, label='random val')
            except:
                print('NO DATA AVAILABLE FROM RANDOM QUERIES')
            fig = plt.plot(x, y, label=self.method+' train')
            plt.plot(x, val_y, label=self.method+' val')
            plt.xlabel("number of queries")
            plt.ylabel(metric)
            plt.legend()
            plt.show()
