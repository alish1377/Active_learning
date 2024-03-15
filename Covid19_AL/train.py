import os
import sys
from datetime import datetime
from models import load_model
from params import get_args
from data.data_loader import get_loader
from tensorflow.keras.optimizers import RMSprop
import mlflow
from utils.callbacks import get_callbacks, get_decay_scheduler
from utils.mlflow_handler import MLFlowHandler
from utils.utils import get_gpu_grower
from utils.plots import get_plots

get_gpu_grower()


def train():
    model_name = sys.argv[2]
    print(f"Chosen Model: {model_name}")
    args = get_args(model_name)
    print(f"Arguments: {args}")

    id_ = model_name + "_" + str(datetime.now().date()) + "_" + str(datetime.now().time())
    # file with ':' in their name  not allowed in Windows.
    id_ = id_.replace(':', '.')
    weight_path = os.path.join(os.getcwd(), 'weights', id_) + ".h5"
    mlflow_handler = MLFlowHandler(model_name=model_name, run_name=id_, mlflow_source=args.mlflow_source,
                                   run_ngrok=args.run_ngrok, ngrok_token=args.ngrok_token)
    mlflow_handler.start_run(args)

    # Loading Data
    train_loader, valid_loader, test_loader = get_loader(train_path=args.train_path,
                                                         val_path=args.val_path,
                                                         test_path=args.test_path,
                                                         batch_size=args.batch_size,
                                                         target_size=tuple(args.target_size)
                                                         )
    print("Loading Data is Done!")
    print(f"data classes: {test_loader.class_indices}")
    # Loading Model
    model = load_model(model_name=model_name,
                       image_size=args.target_size,
                       n_classes=args.n_classes,
                       )
    print("Loading Model is Done!")
    lr_scheduler = get_decay_scheduler(decay_rate=args.lr_scheduler_decay, min_lr=args.lr_scheduler_min)
    checkpoint, reduce_lr, early_stopping, learning_rate_scheduler = get_callbacks(weight_path,
                                                                                   early_stopping_p=5,
                                                                                   mlflow=mlflow,
                                                                                   scheduler=lr_scheduler)

    # -------------------------------------------------------------------

    # Training
    opt = RMSprop(learning_rate=args.lr)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'sparse_categorical_accuracy']
                  )
    print("Training Model...")
    model.fit(train_loader,
              batch_size=args.batch_size,
              steps_per_epoch=(train_loader.n // args.batch_size),
              epochs=args.epochs,
              validation_data=valid_loader,
              validation_batch_size=args.batch_size,
              callbacks=[learning_rate_scheduler, checkpoint, reduce_lr, mlflow_handler.mlflow_logger]
              )
    print("Training Model is Done!")
    print("Evaluating the test data")
    get_plots(model, test_loader, args, mlflow_handler)
    if args.fine_tune:
        print("running fine tune")
        # Unfreeze the base_model. Note that it keeps running in inference mode
        # since we passed `training=False` when calling it. This means that
        # the batchnorm layers will not update their batch statistics.
        # This prevents the batchnorm layers from undoing all the training
        # we've done so far.
        model.trainable = True
        model.summary()
        # Training
        opt = RMSprop(learning_rate=args.ft_lr)
        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', 'sparse_categorical_accuracy']
                      )
        train_loader.reset()
        valid_loader.reset()
        model.fit(train_loader,
                  batch_size=args.batch_size,
                  steps_per_epoch=(train_loader.n // args.batch_size),
                  epochs=args.ft_epoch,
                  validation_data=valid_loader,
                  validation_batch_size=args.batch_size,
                  callbacks=[mlflow_handler.mlflow_logger]
                  )
        print("Evaluating fine tuned model")
        get_plots(model, test_loader, args, mlflow_handler)

    mlflow_handler.end_run()


if __name__ == '__main__':
    train()
