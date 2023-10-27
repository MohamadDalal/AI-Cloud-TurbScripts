import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


def max_pool_3d(method, input_array, pool_size=(3, 3, 3)):
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


# cwd = os.getcwd()
# cube_dir = os.path.join(cwd, "data", "FullCube_TimeIndex2000")
# slice_list = os.listdir(cube_dir)
# slice_list.sort()
# cube = [np.load(f"{cube_dir}/{slice_list[i]}", mmap_mode="r") for i in range(1)]

# input_array = cube[0]
# input_array = np.mean(input_array, axis=3)

# pool_size = (32, 32, 32)
# pooled_array = max_pool_3d(np.max, input_array, pool_size)
# print("Input shape:", input_array.shape)
# print("Pooled shape:", pooled_array.shape)

# fig, axes = plt.subplots(1,2)
# axes[0].imshow(pooled_array[:,pooled_array.shape[1]//2,:], vmin=0, vmax=np.max(input_array))
# axes[1].imshow(input_array[:,64,:], vmin=0, vmax=np.max(input_array))
# fig.savefig("3dpool.png")

work_dir = os.getcwd()
data_dir = os.path.join(work_dir, "data")

#-------------------------------------------------#
# # POOLING dataSent
# channel_dir = os.path.join(data_dir, "dataSent")
# pooled_dir = os.path.join(data_dir, "dataSent_Pooled")
# file_list = os.listdir(channel_dir)
# file_list = [x for x in file_list if x[-1]=="5"]
# data = [h5py.File(f"{channel_dir}/{x}") for x in file_list if x[-1] == "5"]

# print("\n",list(data[0].keys()))
# print(data[0][tuple(data[0].keys())[0]])

# for i, x in enumerate(file_list):
#     print(i,x)
#     input_array = np.mean(data[i][tuple(data[i].keys())[0]], axis=3)
#     print(input_array.shape)
#     pooled_array = max_pool_3d(np.max, input_array)
#     print(pooled_array.shape)
#     np.save(f"{pooled_dir}/{x}", pooled_array)
# print("Done")

#-------------------------------------------------#
# # POOLING channelData
# channel_dir = os.path.join(data_dir, "channelData")
# pooled_dir = os.path.join(data_dir, "channelData_Pooled")
# file_list = os.listdir(channel_dir)

# data = [h5py.File(f"{channel_dir}/{x}") for x in file_list]

# print("\n",list(data[0].keys()))
# print(data[0][tuple(data[0].keys())[0]])

# for i, x in enumerate(file_list):
#     print(i,x)
#     input_array = np.mean(data[i][tuple(data[i].keys())[0]], axis=3)
#     print(input_array.shape)
#     pooled_array = max_pool_3d(np.max, input_array)
#     print(pooled_array.shape)
#     np.save(f"{pooled_dir}/{x}", pooled_array)
# print("Done")

#-------------------------------------------------#
# # POOLING 2000_Full_Channel
# channel_dir = os.path.join(data_dir, "2000_Full_Channel")
# pooled_dir = os.path.join(data_dir, "2000_Full_Channel_Pooled")

# data = [h5py.File(f"{channel_dir}/2000_Full_Channel{x}.h5") for x in range(1, 17)]

# # print("\n",list(data[0].keys()))
# # print(data[0][tuple(data[0].keys())[0]])

# for x in range(16):

#     input_array = np.mean(data[x][tuple(data[x].keys())[0]], axis=3)
#     print(input_array.shape)
#     pooled_array = max_pool_3d(np.max, input_array)
#     print(pooled_array.shape)
#     np.save(f"{pooled_dir}/2000_Full_Channel{x + 1}.npy", pooled_array)

#----------------------------------------------#
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

#----------------------------------------------#
# TORCH - not working, didn't bother
# input_array = np.mean(input_array, axis=3)
# input_torch = torch.from_numpy(input_array)

# pool = torch.nn.MaxPool3d(4, 1, dilation=1)
# out = pool(input_torch)
# fig, axes = plt.subplots(1,2)
# axes[0].imshow(out[:,16,:,:])
# axes[1].imshow(input_array[:,64,:,:])
# fig.savefig("yoloTorch")
