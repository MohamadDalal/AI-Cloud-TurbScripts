import os
import numpy as np
from h5py import File
from sys import stdout

'''
data
├── all_data
    └── train
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


def progress_bar(iteration:int, total:int, bar_length=50):
    '''
    Track progress of a loop.

    :iteration: Current iteration
    :total: Total iteration
    :bar_length: Length of a loading bar
    '''
    progress = iteration / total
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    percent = int(progress * 100)
    stdout.write(f'[{arrow}{spaces}] {percent}%\r')
    stdout.flush()

 
def load_data(folder_name):
    '''
    Load data given a directory path.

    :folder_name: Path to directory of files.

    :return: Tuple where the first element is filenames,
    and the second element is the corresponding data. 
    '''
    filenames = os.listdir(folder_name)
    data = [np.load(f"{folder_name}/{file}") for file in filenames]
    return tuple(filenames, data)


def get_2d_bulk(*args):
    '''
    Extracts 2D images from 3D arrays in a batch.

    :param args: A tuple of directory paths.
    The first element is the directory containing input 3D data,
    and the second element is the directory where 2D images will be saved.
    '''
    for dataset in args:
        in_data, out_data = dataset
        fnames, data = load_data(in_data)
        for fname, chunk in zip(fnames, data):
            axis_slice = [i for i, length in enumerate(chunk.shape) if length == 10][0]
            for idx in range(chunk.shape[axis_slice]):
                img = np.take(chunk, idx, axis=axis_slice)
                np.save(f"{out_data}/{fname[:-4]}{idx}", img)


def sort_imgs(*args):
    '''
    Sorts pooled images and original slices into train and label directories.

    :param args: A tuple of directory paths.
    The first element is the directory containing original 3D data,
    and the second element is the directory of 2D pooled images (saved from get_2d_bulk).
    '''
    all_data_dir = os.path.join(os.getcwd(), "data", "all_data")
    for dataset in args:
        original_data, img_data = dataset
        original_data_h5 = [x for x in os.listdir(original_data) if x.split(".")[-1] == "h5"]
        for chunk in original_data_h5:
            load_chunk = File(f"{original_data}/{chunk}")
            chunk_data = load_chunk[tuple(load_chunk.keys())[0]]
            # List of 2D imgs that correspond to the same chunk
            load_imgs = [np.load(f"{img_data}/{x}") for x in sorted(os.listdir(img_data)) if x.split(".")[0][:-1] == chunk.split(".")[0]]
            for i in range(len(load_imgs)):
                i_original = 3*i + 1
                np.save(f"{all_data_dir}/label/{chunk[:-3]}{i}.npy",chunk_data[:,:,i_original,:])
                np.save(f"{all_data_dir}/train/{chunk[:-3]}{i}.npy", load_imgs[i])
    

pwd = os.getcwd()
data_dir = os.path.join(pwd, "data")

channelData = os.path.join(data_dir, "channelData")
channelDataPooled = os.path.join(data_dir, "channelData_Pooled")
channelDataImg = os.path.join(data_dir, "channelData_Img") 

dataSent = os.path.join(data_dir, "dataSent")
dataSentPooled = os.path.join(data_dir, "dataSent_Pooled")
dataSentImg = os.path.join(data_dir, "dataSent_Img")

Full_Channel = os.path.join(data_dir, "2000_Full_Channel")
Full_ChannelPooled = os.path.join(data_dir, "2000_Full_Channel_Pooled")
Full_ChannelImg = os.path.join(data_dir, "2000_Full_Channel_Img")


get_2d_bulk((Full_ChannelPooled, Full_ChannelImg))
sort_imgs((dataSent, dataSentImg), (channelData, channelDataImg))
