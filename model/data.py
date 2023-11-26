from os.path import join
from os import getcwd, listdir
import torch
from torchvision.transforms import Compose, ToTensor, Resize, GaussianBlur 
from dataset import DatasetFromFolder
from scipy.ndimage import gaussian_filter


'''
# Directory layout #

data
├── all_data
    └── train
        ├── data
        └── labels
    └── validation
        ├── data
        └── labels
    └── test
        ├── data
        └── labels
├── 2000_Full_Channel
├── 2000_Full_Channel_Img
├── 2000_Full_Channel_Pooled
├── channelData
├── channelData_Img
├── channelData_Pooled
├── dataSent
├── dataSent_Img
├── dataSent_Pooled
'''

def input_transform():
    return Compose([
        ToTensor(),
        GaussianBlur(9, 1),
        #Resize((49,16), antialias=True),
        Resize((48,16), antialias=True),
    ])


def target_transform():
    return Compose([
        ToTensor(),
        #Resize((1568,512), antialias=True),
    ])

# Not used
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
    if torch.cuda.is_available():
        print("get_training_set: Cuda is available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return DatasetFromFolder(train_dir,
                             device,
                             input_transform=input_transform(),
                             target_transform=target_transform())


def get_validation_set():
    root_dir = join(getcwd(), "data", "all_data")
    validation_dir = join(root_dir, "validation")
    if torch.cuda.is_available():
        print("get_validation_set: Cuda is available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return DatasetFromFolder(validation_dir,
                             device,
                             input_transform=input_transform(),
                             target_transform=target_transform())

def get_test_set():
    root_dir = join(getcwd(), "data", "all_data")
    test_dir = join(root_dir, "test")
    if torch.cuda.is_available():
        print("get_test_set: Cuda is available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return DatasetFromFolder(test_dir,
                             device,
                             input_transform=input_transform(),
                             target_transform=target_transform())
