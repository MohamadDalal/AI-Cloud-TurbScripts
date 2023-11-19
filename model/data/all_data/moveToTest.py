import numpy as np
import os

split_size = 0.25
input_dir = os.path.join(os.getcwd(), "train/data")
input_dir2 = os.path.join(os.getcwd(), "train/labels")
output_dir = os.path.join(os.getcwd(), "test/data")
output_dir2 = os.path.join(os.getcwd(), "test/labels")
all_file_list = os.listdir(input_dir)
file_list_npy = [x for x in all_file_list if x.split(".")[-1] == "npy"]
print(input_dir, output_dir)
print(len(file_list_npy))
RNG = np.random.default_rng(1234)
test_list_npy = RNG.choice(file_list_npy, int(len(file_list_npy)*split_size), replace=False)
for x in test_list_npy:
    #print(f"mv {input_dir}/{x}, {output_dir}/{x}")
    #print(f"mv {input_dir2}/{x}, {output_dir2}/{x}")
    if not os.path.isfile(f"{input_dir2}/{x}"):
        print(f"File {input_dir2}/{x} does not exist")
        continue
    else:
        os.popen(f"mv {input_dir}/{x} {output_dir}/{x}")
        os.popen(f"mv {input_dir2}/{x} {output_dir2}/{x}")