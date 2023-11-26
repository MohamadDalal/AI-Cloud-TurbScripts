import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Creates an image to represent the data we query",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--csv-path", type=str, default="IndicesToQuery.csv",
                        help="Path of the csv file to use.")
    parser.add_argument("-o", "--output-path", type=str, default="IndicesToQuery.png",
                        help="Path to save image.")
    parser.add_argument("--true-color", type=tuple, default=(255,255,0),
                        help="Path to save image.")
    parser.add_argument("--false-color", type=tuple, default=(128,128,128),
                        help="Path to save image.")
    
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    #img = np.zeros((2048, 4000, 3))
    img = np.full((2048, 4000,3), args.false_color)
    for index, row in df.iterrows():
        #print(row)
        img[row["xStart"]:row["xEnd"], row["timestep"]] = args.true_color

    #Fig, ax = plt.subplots(figsize=(40,20))
    Fig, ax = plt.subplots(figsize=(20,10))
    ax.imshow(img)
    ax.set_title("Queried data points across time and X axis")
    ax.set_xlabel("time")
    ax.set_ylabel("X axis")
    #Fig.show()
    Fig.savefig(args.output_path, dpi=600)
    plt.show()


    