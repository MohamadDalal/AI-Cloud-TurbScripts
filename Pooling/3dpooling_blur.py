import h5py
from sys import stdout
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv

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


def max_pool_3d(method, input_array, pool_size=(8, 8, 3)):
    '''
    3D pooling operation on a given 3D array.

    :method: Numpy method for pooling (np.mean, np.median, np.max, np.min,...)
    :input_array: Input 3D array
    :pool_size: Size of a pooling filter
    '''
    x, y, z = input_array.shape
    pool_x, pool_y, pool_z = pool_size

    output_x = x // pool_x
    output_y = y // pool_y
    output_z = z // pool_z

    pooled_array = np.zeros((output_x, output_y, output_z))

    for i in range(output_x):
        for j in range(output_y):
            for k in range(output_z):
                x_start = i * pool_x
                x_end = x_start + pool_x
                y_start = j * pool_y
                y_end = y_start + pool_y
                z_start = k * pool_z
                z_end = z_start + pool_z

                pooled_array[i, j, k] = method(input_array[x_start:x_end, y_start:y_end, z_start:z_end])

    return pooled_array


def batch_pool(input_dir, output_dir, method, pool_size=(8,8,3), loading_bar=True, slice_axis=2):
    '''
    3D pooling operation on a given 3D array.
    
    :input_dir: Input directory of 3D arrays
    :output_dir: Destination directory of pooled 3D arrays
    :method: Numpy method for pooling (np.mean, np.median, np.max, np.min,...)
    :pool_size: Size of a pooling filter
    :loading_bar: Display loading bar in terminal
    :slice_axis: Axis to make slices from
    '''
    all_file_list = os.listdir(input_dir)
    file_list_h5 = [x for x in all_file_list if x.split(".")[-1] == "h5"]
    data = [h5py.File(f"{input_dir}/{x}") for x in file_list_h5]
    count = 11 #iterator for Gaussian blur

    for i, x in enumerate(file_list_h5):
        channels = []
        input_array = data[i][tuple(data[i].keys())[0]]
        for j in range(input_array.shape[-1]):
            pooled_channel = max_pool_3d(method, input_array[...,j], pool_size) 
            channels.append(pooled_channel[..., np.newaxis])
        pooled_array = np.concatenate(channels, axis=3)
        print(f"{output_dir}/{x[:-3]}\t{i}")
        for j in range(pooled_array.shape[slice_axis]):
            #start = pool_size[slice_axis]*j
            #end = start + pool_size[slice_axis]
            blur_krnl_9 = cv.GaussianBlur(pooled_array.take(j, slice_axis),(5,5),1) #Gaussian blur on pooled_arrays
            for i in range(count):
                more_blur_krnl_9 = cv.GaussianBlur(blur_krnl_9,(9,9),1)
            np.save(f"{output_dir}/data/{x[:-3]}{j}.npy", more_blur_krnl_9) #changed this line to instead take
            #np.save(f"{output_dir}/labels/{x[:-3]}{j}.npy", np.take(input_array, np.arange(start,end), slice_axis))
        if loading_bar:
            progress_bar(i+1, len(file_list_h5))

    return None


work_dir = os.getcwd()
data_dir = os.path.join(work_dir, "data")

dSent = os.path.join(data_dir, "../dataSent")
dSent_Pooled = os.path.join(data_dir, "../dataSent_Pooled")

cData = os.path.join(data_dir, "channelData")
cData_Pooled = os.path.join(data_dir, "channelData_Pooled")

fChannel = os.path.join(data_dir, "2000_Full_Channel")
fChannel_Pooled = os.path.join(data_dir, "2000_Full_Channel_Pooled")

#inputDir = os.path.join(data_dir, "../dataSent")
inputDir = os.path.join(work_dir, "../../channelData")
outputDir = os.path.join(work_dir, "../model/data/all_data/train")
Path(os.path.join(outputDir, "data")).mkdir(parents=True, exist_ok=True)
#Path(os.path.join(outputDir, "labels")).mkdir(parents=True, exist_ok=True)

# batch_pool(input_dir=fChannel, output_dir=fChannel_Pooled, method=np.mean, pool_size=(31,31,3))
batch_pool(input_dir=inputDir, output_dir=outputDir, method=np.mean, pool_size=(8,8,3), slice_axis=2)
# batch_pool(input_dir=cData, output_dir=cData_Pooled, method=np.mean, pool_size=(31,31,3))

#-------------------------------------------------#
# # 3D VISUALIZATION
# input_array = np.mean(data[0][tuple(data[0].keys())[0]], axis=3)
# print(input_array.shape)
# pooled_array = max_pool_3d(np.max, input_array)
# x, y, z = np.meshgrid(np.arange(pooled_array.shape[2]), np.arange(pooled_array.shape[1]), np.arange(pooled_array.shape[0]))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c=pooled_array, cmap='viridis', marker="s", vmin=0, vmax=np.max(input_array))
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# fig.savefig("3dcubeh5data.png")

# # SHAPE IS: Z,Y,X -> visualizing half of X.
# fig, axes = plt.subplots(1,2)
# axes[0].imshow(pooled_array[:,:,pooled_array.shape[2]//2], vmin=0, vmax=np.max(input_array))
# axes[1].imshow(input_array[:,:,16], vmin=0, vmax=np.max(input_array))
# fig.savefig("3dpool.png")
