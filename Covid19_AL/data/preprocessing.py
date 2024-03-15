import random
from os.path import join
import os
import zipfile
import pandas as pd
from argparse import ArgumentParser
import shutil
from imutils import paths


def preprocessing(kaggle_user="monibaravan",
                  kaggle_key="7a03bfd58536ced3a30c3b5c742096dd",
                  val_size=0.2,
                  augmented_samples=True,
                  augmented_test=True,
                  dataset_path='blood-cells.zip',
                  train_path='data/train',
                  val_path='data/val',
                  test_path='data/test'
                  ):
    print('If you are living in Iran, turn on you VPN!!!\n')
    os.environ['KAGGLE_USERNAME'] = kaggle_user  # username from the json file
    os.environ['KAGGLE_KEY'] = kaggle_key  # key from the json file
    if not os.path.exists(dataset_path):
        print(f"downloading dataset")
        os.system('kaggle datasets download -d paultimothymooney/blood-cells')
    else:
        print('Dataset exists')

    with zipfile.ZipFile(dataset_path) as file:
        print(f"extracting files from {dataset_path} ")
        file.extractall()
    # remove path if exist
    for p in [train_path, val_path, test_path]:
        print(f"removing {p}")
        shutil.rmtree(p, ignore_errors=True)

    if augmented_samples:
        data_path = "dataset2-master/dataset2-master/images"
        print(f"moving  train images to {train_path}")
        shutil.copytree(join(data_path, 'TRAIN'), train_path)
        if augmented_test:
            print(f"moving augmented test images to {test_path}")
            shutil.copytree(join(data_path, 'TEST'), test_path)
        else:
            print(f"moving non augmented test images to {test_path}")
            shutil.copytree(join(data_path, 'TEST_SIMPLE'), test_path)
        # Generate validation set from train set
        # first get all train images paths and randomly shuffle
        train_images = list(paths.list_images(train_path))
        random.seed(42)
        random.shuffle(train_images)
        # selecting a subset of that images
        i = int(len(train_images) * val_size)
        validation_img = train_images[:i]
        # ensure validation path has created
        os.makedirs(val_path, exist_ok=True)
        # moving images from train directory to validation folder
        print(f"moving validation data into {val_path}")
        for p in validation_img:
            img_path = os.path.normpath(p).split(os.sep)
            dir_path = join(val_path, *img_path[-2: -1])
            os.makedirs(dir_path, exist_ok=True)
            img_path = join(*img_path[-2:])
            shutil.move(p, join(val_path, img_path))
    else:
        extract_no_augmention(train_path, val_path, test_path, val_size)
    # remove unnecessary directories
    shutil.rmtree("dataset2-master")
    shutil.rmtree("dataset-master")


def extract_no_augmention(train_path, val_path, test_path, val_split):
    data_path = "dataset-master/dataset-master/JPEGImages"
    label_csv_path = "dataset-master/dataset-master/labels.csv"
    labels_df = pd.read_csv(label_csv_path)
    train_split = 0.8
    # grab the paths to all input images in the original input directory
    # and shuffle them
    imagePaths = list(paths.list_images(data_path))
    random.seed(42)
    random.shuffle(imagePaths)

    # compute the training and testing split
    i = int(len(imagePaths) * train_split)
    trainPaths = imagePaths[:i]
    testPaths = imagePaths[i:]

    # we'll be using part of the training data for validation
    i = int(len(trainPaths) * val_split)
    valPaths = trainPaths[:i]
    trainPaths = trainPaths[i:]

    # define the datasets that we'll be building
    datasets = [
        ("training", trainPaths, train_path),
        ("validation", valPaths, val_path),
        ("testing", testPaths, test_path)
    ]
    # loop over the datasets
    for (dType, imagePaths, baseOutput) in datasets:
        # show which data split we are creating
        print("[INFO] building '{}' split".format(dType))
        # if the output base output directory does not exist, create it
        if not os.path.exists(baseOutput):
            print("[INFO] 'creating {}' directory".format(baseOutput))
            os.makedirs(baseOutput)
        # loop over the input image paths
        for inputPath in imagePaths:
            # extract the filename of the input image and extract the
            # class label ("0" for "negative" and "1" for "positive")
            filename = inputPath.split(os.path.sep)[-1]
            # int() to remove  zeros 00
            label_no = int(filename[-9:-4])

            try:
                label_names = labels_df.loc[labels_df['Image'] == label_no]['Category'].values[0].replace(' ', '') \
                    .split(',')
                print(label_names, label_no)
            except Exception:
                print(f"error cl {label_no} ")
                # break

            for label_name in label_names:
                # build the path to the label directory
                labelPath = os.path.sep.join([baseOutput, label_name])
                # if the label output directory does not exist, create it
                if not os.path.exists(labelPath):
                    print("[INFO] 'creating {}' directory".format(labelPath))
                    os.makedirs(labelPath)
                # construct the path to the destination image and then copy
                # the image itself
                p = os.path.sep.join([labelPath, filename])
                shutil.copy2(inputPath, p)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--kaggle_user", default="monibaravan", type=str)
    parser.add_argument("--kaggle_key", default="7a03bfd58536ced3a30c3b5c742096dd", type=str)
    parser.add_argument('--train_path', type=str, default='data/data/train',
                        help='Path  train dataset directory.',
                        required=False)
    parser.add_argument('--dataset_path', type=str, default='blood-cells.zip',
                        help='dataset path',
                        required=False)
    parser.add_argument('--val_path', type=str, default='data/data/val',
                        help='Path to folder containing val dataset directory.',
                        required=False)
    parser.add_argument('--test_path', type=str, default='data/data/test',
                        help='Path to folder containing test dataset directory.',
                        required=False)
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='validation size',
                        required=False)
    parser.add_argument('--augmented_samples', type=bool, default=True,
                        help='Use augmented data (dataset2-master)',
                        required=False)
    parser.add_argument('--augmented_test', type=bool, default=True,
                        help='Use augmented test data',
                        required=False)

    args = parser.parse_args()
    preprocessing(
        kaggle_user=args.kaggle_user,
        kaggle_key=args.kaggle_key,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        dataset_path=args.dataset_path,
        val_size=args.val_size,
        augmented_test=args.augmented_test,
        augmented_samples=args.augmented_samples
    )
