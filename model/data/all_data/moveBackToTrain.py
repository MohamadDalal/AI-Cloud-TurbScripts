import os

input_dir = os.path.join(os.getcwd(), "test/data")
input_dir2 = os.path.join(os.getcwd(), "test/labels")
output_dir = os.path.join(os.getcwd(), "train/data")
output_dir2 = os.path.join(os.getcwd(), "train/labels")
all_file_list = os.listdir(input_dir)
file_list_npy = [x for x in all_file_list if x.split(".")[-1] == "npy"]
print(input_dir, output_dir)
print(len(file_list_npy))
for x in file_list_npy:
    if not os.path.isfile(f"{input_dir2}/{x}"):
        print(f"File {input_dir2}/{x} does not exist")
        continue
    else:
        os.popen(f"mv {input_dir}/{x} {output_dir}/{x}")
        os.popen(f"mv {input_dir2}/{x} {output_dir2}/{x}")