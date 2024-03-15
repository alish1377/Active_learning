from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import tensorflow as tf


def get_decay_scheduler(decay_rate=-0.5, min_lr=0):
    def scheduler(epoch, lr):
        new_lr = lr * tf.math.exp(decay_rate)
        if new_lr > min_lr:
            lr = new_lr
        print(f"learning rate:  {lr}")
        return lr

    return scheduler


def get_callbacks(model_path, early_stopping_p, save_weights_only=True, plateau_min_lr=0.0001,
                  # set default scheduler , you can pass your custom one
                  scheduler=get_decay_scheduler(), **kwargs):
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=save_weights_only,
                                 )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,  # new_lr = lr * factor
                                  patience=4,  # number of epochs with no improvment
                                  min_lr=plateau_min_lr,  # lower bound on the learning rate
                                  mode='min',
                                  verbose=1
                                  )
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_p, verbose=1)
    return checkpoint, reduce_lr, early_stopping, learning_rate_scheduler
