import torch.utils.data as data

from os import listdir
from os.path import join
import numpy as np


def is_array_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])


def load_array(filepath):
    return np.load(filepath)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_array_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_array(self.image_filenames[index])
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)