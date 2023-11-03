import h5py
from sys import stdout
import numpy as np
import os
import matplotlib.pyplot as plt


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


def max_pool_3d(method, input_array, pool_size=(3, 3, 3)):
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


def batch_pool(input_dir, output_dir, method, pool_size=(3,3,3), loading_bar=True):
    '''
    3D pooling operation on a given 3D array.
    
    :input_dir: Input directory of 3D arrays
    :output_dir: Destination directory of pooled 3D arrays
    :method: Numpy method for pooling (np.mean, np.median, np.max, np.min,...)
    :pool_size: Size of a pooling filter
    '''
    all_file_list = os.listdir(input_dir)
    file_list_h5 = [x for x in all_file_list if x.split(".")[-1] == "h5"]
    data = [h5py.File(f"{input_dir}/{x}") for x in file_list_h5]

    for i, x in enumerate(file_list_h5):
        input_array = np.mean(data[i][tuple(data[i].keys())[0]], axis=3)
        pooled_array = max_pool_3d(method, input_array, pool_size)

        np.save(f"{output_dir}/{x[:-3]}", pooled_array)
        if loading_bar:
            progress_bar(i+1, len(file_list_h5))

    return None


work_dir = os.getcwd()
data_dir = os.path.join(work_dir, "data")

dataSent = os.path.join(data_dir, "dataSent")
dataSent_Pooled = os.path.join(data_dir, "dataSent_Pooled")

FChannel = os.path.join(data_dir, "2000_Full_Channel")
FChannel_Pooled = os.path.join(data_dir, "2000_Full_Channel_Pooled")

batch_pool(input_dir=FChannel, output_dir=FChannel_Pooled, method=np.mean, pool_size=(3,3,3))

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