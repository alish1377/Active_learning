# import kaggle
import zipfile
from sklearn.model_selection import train_test_split
import os
from argparse import ArgumentParser
import shutil
import random


def copy_data(list_img, set_path, extracted_data_path, cls, data_size):
    new_path = os.path.join(set_path,cls)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    random.seed(4)
    random.shuffle(list_img)
    list_img = list_img[:: data_size]
    print(f'len {set_path.split("/")[-1]} for class {cls} is :' , len(list_img))
    for img in list_img:
        shutil.move(os.path.join(extracted_data_path, cls, img), new_path)

def MED_preprocessing(
    kaggle_user = 'monibaravan',
    kaggle_key = '7a03bfd58536ced3a30c3b5c742096dd',
    data_path = '/content/dataset',
    train_path = '/content/data/train',
    test_path = '/content/data/test',
    val_path = '/content/data/val',
    test_size = 0.166666,
    val_size = 0.2,
    data_size = 20):


    os.environ['KAGGLE_USERNAME'] = kaggle_user # username from the json file
    os.environ['KAGGLE_KEY'] = kaggle_key # key from the json file
    os.system('kaggle datasets download -d tawsifurrahman/covid19-radiography-database')
    os.makedirs(data_path)

    zip_path = 'covid19-radiography-database.zip'
    extracted_data_path = data_path
    with zipfile.ZipFile(zip_path) as file:
        print(f"extracting files from {zip_path} ")
        file.extractall(extracted_data_path)

    main_data_path = os.path.join(extracted_data_path, 'COVID-19_Radiography_Dataset')
    classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    for cls in classes:
        img_lists = os.listdir(os.path.join(main_data_path,cls))
        train_img_list, test_img_list = train_test_split(img_lists, random_state=200, test_size = test_size)
        train_img_list, val_img_list = train_test_split(train_img_list, random_state=200, test_size = val_size )
        copy_data(train_img_list, train_path, main_data_path, cls, data_size)
        copy_data(val_img_list, val_path, main_data_path, cls, data_size)
        copy_data(test_img_list, test_path, main_data_path, cls, data_size)

    shutil.rmtree(extracted_data_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--kaggle_user", default="monibaravan", type=str)
    parser.add_argument("--kaggle_key", default="7a03bfd58536ced3a30c3b5c742096dd", type=str)
    parser.add_argument('--data_path', type=str, default='/content/dataset',
                        help='dataset path',
                        required=False)
    parser.add_argument('--train_path', type=str, default='/content/data/train',
                        help='Path  train dataset directory.',
                        required=False)
    parser.add_argument('--val_path', type=str, default='/content/data/val',
                        help='Path to folder containing val dataset directory.',
                        required=False)
    parser.add_argument('--test_path', type=str, default='/content/data/test',
                        help='Path to folder containing test dataset directory.',
                        required=False)
    parser.add_argument('--test_size', type=float, default=0.1666666,
                        help='test size',
                        required=False)
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='validation size',
                        required=False)
    parser.add_argument('--data_size', type=float, default=5,
                        help='part of main dataset we choose for task(1/20 part of dataset)',
                        required=False)    

    args = parser.parse_args()

    MED_preprocessing(
        kaggle_user = args.kaggle_user,
        kaggle_key = args.kaggle_key,
        data_path = args.data_path,
        train_path = args.train_path,
        test_path = args.test_path,
        val_path = args.val_path,
        test_size = args.test_size,
        val_size = args.val_size,
        data_size = args.data_size
    )