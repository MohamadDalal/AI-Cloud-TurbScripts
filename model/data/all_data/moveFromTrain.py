import numpy as np
import os

input_dir = os.path.join(os.getcwd(), "train/data")
input_dir2 = os.path.join(os.getcwd(), "train/labels")

validation_dir = os.path.join(os.getcwd(), "validation/data")
validation_dir2 = os.path.join(os.getcwd(), "validation/labels")

test_dir = os.path.join(os.getcwd(), "test/data")
test_dir2 = os.path.join(os.getcwd(), "test/labels")

validation_list = np.loadtxt("validation_chunks.csv")
test_list = np.loadtxt("test_chunks.csv")

for x in validation_list:
    for i in range(10):
        if not os.path.isfile(f"{input_dir}/{x}{i}.npy"):
            print(f"File {input_dir}/{x}{i}.npy does not exist")
            continue
        elif not os.path.isfile(f"{input_dir2}/{x}{i}.npy"):
            print(f"File {input_dir2}/{x}{i}.npy does not exist")
            continue
        else:
            os.popen(f"mv {input_dir}/{x}{i}.npy {validation_dir}/{x}{i}.npy")
            os.popen(f"mv {input_dir2}/{x}{i}.npy {validation_dir2}/{x}{i}.npy")

for x in test_list:
    for i in range(10):
        if not os.path.isfile(f"{input_dir}/{x}{i}.npy"):
            print(f"File {input_dir}/{x}{i}.npy does not exist")
            continue
        elif not os.path.isfile(f"{input_dir2}/{x}{i}.npy"):
            print(f"File {input_dir2}/{x}{i}.npy does not exist")
            continue
        else:
            os.popen(f"mv {input_dir}/{x}{i}.npy {test_dir}/{x}{i}.npy")
            os.popen(f"mv {input_dir2}/{x}{i}.npy {test_dir2}/{x}{i}.npy")