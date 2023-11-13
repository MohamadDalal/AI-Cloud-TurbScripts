import torch.utils.data as data

from os import listdir
from os.path import join, exists
import numpy as np


def is_array_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])


def load_array(filepath):
    return np.load(filepath)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None, image_filenames=None):
        super(DatasetFromFolder, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_array_file(x)]
        self.data_dir = join(image_dir, "data")
        self.label_dir = join(image_dir, "labels")
        if image_filenames is None:
            self.image_filenames = [x for x in sorted(listdir(self.data_dir)) if is_array_file(x)]
        else:
            self.image_filenames = [x for x in image_filenames if is_array_file(x)]
          
        index = 0      
        for _ in range(len(self.image_filenames)):
            if exists(join(self.label_dir, self.image_filenames[index])):
                index += 1
            else:
                print(f'No label array exists for slice {self.image_filenames[index]}')
                self.image_filenames.pop(index)
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_array(join(self.data_dir, self.image_filenames[index]))
        input = np.float32(input)
        target = load_array(join(self.label_dir, self.image_filenames[index]))
        # target = np.float32(np.mean(target, axis=2)) # Do we need it axis-averaged?
        target = np.float32(target)
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)