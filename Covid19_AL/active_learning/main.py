from active_learning.al_custom_generator import AL
from active_learning.data_loader import CustomImageGenerator, get_loader
from models import load_model
from tensorflow.keras.optimizers import RMSprop
from argparse import ArgumentParser
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_al(args):
    print(f"Arguments: {args}")

    # train = ( train_list, y_train ) and so on..
    train, pool, val, test, class_name_map = get_loader(train_path=args.train_path,
                                                        val_path=args.val_path,
                                                        test_path=args.test_path,
                                                        pool_size=args.pool_size
                                                        )

    generator_args = {'shuffle': not args.no_shuffle,
                      'batch_size': args.batch_size,
                      'img_size': tuple(args.target_size),
                      'n_channels': args.img_channels,
                      'aug_prob': args.aug_prob,
                      'n_classes': args.n_classes,
                      'class_name_map': class_name_map
                      }
    print(f'Size of initial train dataset {len(train[0])}')
    print(f'Size of initial pool dataset {len(pool[0])}')
    print(f'Size of validation  dataset {len(val[0])}')
    print(f'Size of test dataset {len(test[0])}')
    print("Loading list of dataset is Done!")

    model = load_model(model_name=args.model,
                       image_size=args.target_size,
                       n_classes=args.n_classes,
                       )
    print("Loading Model is Done!")

    opt = RMSprop(learning_rate=args.lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='acc')

    ac_learner = AL(model,
                    train_list=train[0],
                    y_train=train[1],
                    pool_list=pool[0],
                    y_pool=pool[1],
                    val_list=val[0],
                    y_val=val[1],
                    df=pd.DataFrame(),
                    df_best = pd.DataFrame(),
                    query_strategy='random',
                    batch_size=args.batch_size,
                    init_epochs=args.init_epochs,
                    model_path = args.model_path,
                    initial_model_path = args.initial_model_path,
                    initial_history = args.initial_history,
                    gen_args=generator_args
                    )
    print("Learner is made!")

    ac_learner.fit(each_query_epochs=args.each_query_epochs,
                   query_size=args.query_batch_size,
                   only_new=args.only_new,
                   n_queries=args.n_queries,
                   uncertainty_rate=args.uncertainty_rate,
                   factor_oversampling=args.factor_oversampling)

    # evaluate Test dataset with plots
    generator_args['shuffle'] = False
    test_gen = CustomImageGenerator(img_list=test[0], labels=test[1], **generator_args)

    # ---------Temporary-to-evaluate-start------------------------------------------------------------------------------
    print('Making  prediction ')
    n_classes = args.n_classes
    y_pred, y_true = None, None
    for x, y in test_gen:
        predictions = model.predict(x, batch_size=1)
        temp = np.argmax(predictions, axis=1)
        y = np.argmax(y, axis=1)
        if y_true is None:
            y_true = y
            y_pred = temp
        else:
            y_true = np.concatenate([y_true, y])
            y_pred = np.concatenate([y_pred, temp])

    correct_count = 0.0
    for i, y1 in enumerate(y_pred):
        if y1 == y_true[i]:
            correct_count = correct_count + 1
    print(f"Test Accuracy:  {correct_count / len(y_pred)}")
    # Metrics: Confusion Matrix
    con_mat = confusion_matrix(y_true, y_pred)
    print(con_mat)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=[i for i in range(n_classes)], columns=[i for i in range(n_classes)])
    plt.figure(figsize=(n_classes, n_classes))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')
    plt.show()
    report = classification_report(y_true, y_pred)
    print(report)

    # ---------Temporary-to-evaluate-end------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Model name. Default = resnet50', required=False)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Define the size of every training batch. Default = 32')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Choose verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default = 1')
    parser.add_argument('--lr', type=float, default=0.001, help='Set learning rate.', required=False)
    parser.add_argument('--train_path', type=str, default='/content/data/train',
                        help='Path to folder containing train dataset directory.',
                        required=False)
    parser.add_argument('--val_path', type=str, default='/content/data/val',
                        help='Path to folder containing val dataset directory.',
                        required=False)
    parser.add_argument('--test_path', type=str, default='/content/data/test',
                        help='Path to folder containing test dataset directory.',
                        required=False)
    parser.add_argument('--target_size', type=list, nargs=2, default=[32, 32], help='Image size for model.',
                        required=False)
    parser.add_argument('--img_channels', type=int, default=3, help='Number of Image channels for model.',
                        required=False)
    parser.add_argument('--n_classes', type=int, default=4, help='Number of classes', required=False)

    # active_learning parameters
    parser.add_argument('--pool_size', type=float, default=0.93, help='Pool size to separate pool from train.',
                        required=False)
    parser.add_argument('--aug_prob', type=float, default=0.5, help='Probability of applying the augmentation',
                        required=False)

    parser.add_argument('--no_shuffle', action='store_true',
                        help='False, if batches between epochs shouldn\'t look alike.',
                        required=False)

    parser.add_argument('--n_queries', type=int, default=26, help='Number of queries.', required=False)
    parser.add_argument('--init_epochs', type=int, default=20, help='Number of epochs in initial training',
                        required=False)
    parser.add_argument('--uncertainty_rate', type=float, default=1,
                        help='proportion of query that you want to be uncertainty(for CEAL)',
                        required=False)
    parser.add_argument('--each_query_epochs', type=int, default=5, help='Number of epochs in each query.',
                        required=False)
    parser.add_argument('--query_batch_size', type=int, default=100,
                        help='Number of instances selected to be labelled in each query',
                        required=False)
    parser.add_argument('--only_new', action='store_true',
                        help='True, If only selected instances use to train model.',
                        required=False)
    parser.add_argument('--factor_oversampling', type=int, default=3,
                        help='Factor by which number of new labelled-images will be oversampled.',
                        required=False)
    parser.add_argument('--initial_model_path', type=str, default='/content/drive/MyDrive/Covid19/initial_model.hdf5',
                        help='initial model path for loading in another strategy',
                        required=False)
    parser.add_argument('--model_path', type=str, default='/content/drive/MyDrive/Covid19/each_epoch_model.hdf5',
                        help='best model for each query',
                        required=False)     
    parser.add_argument('--initial_history', type=str, default='/content/drive/MyDrive/Covid19/initial_history.npy',
                        help='initial history for join as first accuracy',
                        required=False)


    args = parser.parse_args()
    run_al(args)
