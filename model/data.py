from os.path import join
from os import getcwd, mkdir, listdir
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from dataset import DatasetFromFolder


def input_transform():
    return Compose([
        ToTensor(),
    ])


def target_transform():
    return Compose([
        ToTensor(),
    ])


def train_test_split(train_ratio=0.8):
    root_dir = join(getcwd(), "data", "all_data")
    all_dir = join(root_dir, "train")
    gt_dir = join(root_dir, "label")
    test_dir = join(root_dir, "test_data")
    train_data = join(root_dir, "train_data")

    all_data, gt_data = sorted(listdir(all_dir)), sorted(listdir(gt_dir))
    train_data, test_data = all_data[:int(len(all_data)*train_ratio)], all_data[int(len(all_data)*train_ratio):] 
    gt_train, gt_test = gt_data[:int(len(gt_data)*train_ratio)], gt_data[int(len(gt_data)*train_ratio):]

    return (train_data, gt_train), (test_data, gt_test)

def get_training_set():
    root_dir = join(getcwd(), "data", "all_data")
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(),
                             target_transform=target_transform())


def get_test_set():
    root_dir = join(getcwd(), "data", "all_data")
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(),
                             target_transform=target_transform())