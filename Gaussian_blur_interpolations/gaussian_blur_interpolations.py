import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import h5py as h5py 
import skimage as ski

# loading the 3D data
data = h5py.File("DNS.h5") 
original_array = data[tuple(data.keys())[0]][:]
#original_resolution = np.shape(original_array)
meanpooled_array = np.load('more_mean_pooled.npy')
maxpooled_array = np.load('more_max_pooled.npy')
#pooled_resolution = np.shape(meanpooled_array)
#print(original_resolution, pooled_resolution)

#averages over the 3 channels
original_array = np.mean(original_array, axis=3)
#taking slices of arrays in the middle and plotting them
fig, axes = plt.subplots(1,3)
axes[0].imshow(original_array[:,:,16], vmin=0, vmax=np.max(original_array))
axes[0].set_title("DNS")
axes[1].imshow(meanpooled_array[:,:,5], vmin=0, vmax=np.max(original_array))
axes[1].set_title("Mean pool")
axes[2].imshow(maxpooled_array[:,:,5], vmin=0, vmax=np.max(original_array))
axes[2].set_title("Max pool")

#saving image
fig.savefig("original_pooled.png")
plt.show()


#original_image = original_array[:,:,16]
mean_pooled = meanpooled_array[:,:,5]        #
max_pooled = maxpooled_array[:,:,5]
d1 = np.shape(mean_pooled)
d2 = np.shape(max_pooled)
print("The image dimensions of mean pooled and max pooled data are:", d1,d2)

#applying Gaussian blur to the average pooled image. Kernel size = 9x9 and sigma x = 0
blur = cv.GaussianBlur(mean_pooled,(9,9),0)

#plotting average pooled image alongside Gaussian blurred image
fig,axes = plt.subplots(1,2)
axes[0].imshow(mean_pooled)
axes[0].set_title("Mean Pooled")
axes[1].imshow(blur)
axes[1].set_title("Gaussian Blur")
fig.savefig("blurred_image.png")
plt.show()

#interpolating and resizing image to 125% of its original size 
img_resized_interlinear = cv.resize(blur, (20,62), interpolation=cv.INTER_LINEAR) #using interlinear interpolation
img_resized_intercubic = cv.resize(blur, (20,62), interpolation=cv.INTER_CUBIC) #using intercubic interpolation
img_resized_nearest = cv.resize(blur, (20,62), interpolation=cv.INTER_NEAREST) #using nearest neighbor interpolation

#rotating images (scrathed)
#array_rotated = cv.rotate(mean_pooled,cv.ROTATE_90_COUNTERCLOCKWISE)
#interlinear_rotated = cv.rotate(img_resized_interlinear,cv.ROTATE_90_CLOCKWISE)
#intercubic_rotated = cv.rotate(img_resized_intercubic,cv.ROTATE_90_CLOCKWISE)
#nearest_rotated = cv.rotate(img_resized_nearest,cv.ROTATE_90_CLOCKWISE)


#plotting downsampled image (no blur) and the blurred images with different interpolations
Titles =["Downsampled Image", "Bilinear Interpolation", "Bicubic Interpolation", "Nearest Neighbor Interpolation"]
images =[mean_pooled, img_resized_interlinear, img_resized_intercubic, img_resized_nearest]
count = 4


for i in range(count):
    plt.subplot(2, 2, i + 1)
    plt.title(Titles[i])
    plt.imshow(images[i])

plt.savefig('interpolations.png')
plt.show()
