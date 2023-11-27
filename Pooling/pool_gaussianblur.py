import h5py
from sys import stdout
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv


def pooling_3d(method, input_array, pool_size=(3, 3, 3)):
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



input_array = np.load('AI-Cloud-TurbScripts\Pooling\T3381-X1267-S32-A06.npy')

channels = []

for j in range(input_array.shape[-1]):
   
    pooled_channel = pooling_3d(method=np.mean, input_array=input_array[...,j], pool_size=(8,8,3)) 
    channels.append(pooled_channel[..., np.newaxis])
pooled_array = np.concatenate(channels, axis=3)

#print(input_array.shape)
print(pooled_channel[...,0].shape)
print(pooled_array[...,0,:].shape)

#plt.imshow(np.sum(pooled_array[...,0,:]**2, axis=2))
#plt.show()

#applying Gaussian blur to the average pooled image. Kernel size = 9x9 and sigma x = 0
blur_krnl_9 = cv.GaussianBlur(np.sum(pooled_array[...,0,:]**2, axis=2),(5,5),1)
blur_krnl_15 = cv.GaussianBlur(np.sum(pooled_array[...,0,:]**2, axis=2),(15,15),1)
blur_krnl_31 = cv.GaussianBlur(np.sum(pooled_array[...,0,:]**2, axis=2),(31,31),1)
blur_krnl_47 = cv.GaussianBlur(np.sum(pooled_array[...,0,:]**2, axis=2),(47,47),1)
blur_krnl_63 = cv.GaussianBlur(np.sum(pooled_array[...,0,:]**2, axis=2),(99,99),1)

more_blur_krnl_9 = cv.GaussianBlur(blur_krnl_9,(5,5),1)
more_blur_krnl_15 = cv. GaussianBlur(blur_krnl_15,(15,15),1)
more_blur_krnl_31 = cv.GaussianBlur(blur_krnl_31,(31,31),1)
more_blur_krnl_47 = cv. GaussianBlur(blur_krnl_47,(47,47),1)
more_blur_krnl_63 = cv. GaussianBlur(blur_krnl_63,(99,99),1)

count = 11

for i in range(count-1):
    more_blur_krnl_9 = cv.GaussianBlur(more_blur_krnl_9,(5,5),1)
    more_blur_krnl_15 = cv. GaussianBlur(more_blur_krnl_15,(15,15),1)
    more_blur_krnl_31 = cv.GaussianBlur(more_blur_krnl_31,(31,31),1)
    more_blur_krnl_47 = cv.GaussianBlur(more_blur_krnl_47,(47,47),1)
    more_blur_krnl_63 = cv. GaussianBlur(more_blur_krnl_63,(99,99),1)


#plotting average pooled image alongside Gaussian blurred image
fig,axes = plt.subplots(1,6)
axes[0].imshow(np.sum(pooled_array[...,0,:]**2, axis=2))
axes[0].set_title("Average pooled data")
axes[1].imshow(more_blur_krnl_9)
axes[1].set_title("Kernel size 9x9")
axes[2].imshow(more_blur_krnl_15)
axes[2].set_title("Kernel size 15x15")
axes[3].imshow(more_blur_krnl_31)
axes[3].set_title("Kernel size 31x31")
axes[4].imshow(more_blur_krnl_47)
axes[4].set_title("Kernel size 47x47")
axes[5].imshow(more_blur_krnl_63)
axes[5].set_title("Kernel size 99x99")
#fig.savefig("blurred_image.png")
plt.show()