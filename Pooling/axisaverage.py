import matplotlib.pyplot as plt
from typing import Union
import numpy as np
import os


class Cube:
    def __init__(self, cube):
        self.cube = cube
        self.pooled = None
        self.multipool = None


    @classmethod
    def load_from_dir(cls, cube_dir: str, slice_count: int = 1):
        """Load a Cube from directory.
        
        Keyword arguments:
        cube_dir -- directory of a full cube
        slice_count -- how many slices to take from cube
        """
        slice_list = os.listdir(cube_dir)
        slice_list.sort()
        cube = [np.load(f"{cube_dir}/{slice_list[i]}", mmap_mode="r") for i in range(slice_count)]
        return cls(np.concatenate(cube, axis=1))


    def normalize(self):
        """Normalize the Cube to [0, 1] range.

        Keyword arguments:
        axis -- axis on which to search for min/max values 
        """
        temp = np.mean(self.cube, axis=3)
        print(temp.shape)
        min_val, max_val = np.min(temp), np.max(temp)
        print(min_val, max_val)
        del(temp)
        self.cube = (self.cube - min_val) / (max_val - min_val)
        return None


    def pool_cube(self, cube, method=np.mean, pooling_axis: Union[int, tuple] = 1):
        """Pool the Cube to 2D array.

        Keyword arguments:
        cube -- array to be pooled
        method -- numpy pooling method (np.average, np.mean, np.median)
        pooling_axis -- squished axis for pooling
        """
        self.pooled = method(cube, axis=pooling_axis)
        return self.pooled


    def pool_multiple(self, method, pooling_axis: Union[int, tuple] = 1,
                      output_num = 1, slice_axis: int = 1):
        """Pool multiple Cube chunks.
        
        Keyword arguments:
        method -- numpy pooling method (np.average, np.mean, np.median)
        pooling_axis -- squished axis for pooling
        output_num -- number of pooled array outputed
        slice_axis -- axis on which the cube is split
        """

        self.multipool = []
        full_axis_size = self.cube.shape[slice_axis]
        slice_size = full_axis_size // output_num
        start = 0
        for i in range(output_num):
            end = min((i + 1) * slice_size, full_axis_size) # Ensure no out-of-bounds
            cube_slice = self.cube.take(np.arange(start, end), axis=slice_axis)
            self.multipool.append(self.pool_cube(cube_slice, method, pooling_axis))
            start = end
        return self.multipool


    def show_pooled(self, filename: str, show_original: bool = False):
        """Show pooled Cube image.

        Keyword arguments:
        filename -- filename that will be saved
        show_original -- boolean to show original next to pooled image
        """
        if self.multipool:
            fig, axes = plt.subplots(1, len(self.multipool) + (1 if show_original else 0))
            for i in range(len(self.multipool)):
                axes[i].imshow(self.multipool[i])
                axes[i].set_title(f"Slice {i + 1}/{len(self.multipool)}")
            if show_original:
                axes[-1].imshow(self.cube[:,self.cube.shape[1]//2,:,:])
            fig.savefig(f"./visualization/multi{filename}")
            return None


        fig, axes = plt.subplots(1, 2) if show_original else plt.subplots(1, 1)
        axes = [axes] if not show_original else axes

        axes[0].imshow(self.pooled)
        axes[0].set_title("Pooled Cube")

        if show_original:
            axes[1].imshow(self.cube[:,self.cube.shape[1]//2, :, :])
            axes[1].set_title("Original Cube")

        plt.tight_layout()
        fig.savefig(f"./visualization/{filename}")


cwd = os.getcwd()
cube_dir = os.path.join(cwd, "data", "FullCube_TimeIndex2000")

cube = Cube.load_from_dir(cube_dir, 1)
# cube.pool_cube(cube.cube, method=np.mean, pooling_axis=(1,3))
cube.pool_multiple(np.mean, pooling_axis=(1,3), output_num=2, slice_axis=1)
cube.show_pooled("pooled.png", show_original=False)
