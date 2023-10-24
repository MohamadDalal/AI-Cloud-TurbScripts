import pandas as pd
import argparse
import h5py
from subprocess import Popen

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Runs getChannelSliceMultiprocessing.py using arguments from a csv to query data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("startIndex", type=int,
                        help="Index to start from in csv file.")
    parser.add_argument("stopIndex", type=int,
                        help="Index to stop at in csv file. Excluded")
    parser.add_argument("-c", "--csv-path", type=str, default="IndicesToQuery.csv",
                        help="Path of the csv file to use.")
    parser.add_argument("-o", "--output-path", type=str, default="channelData/",
                        help="Path to save the h5 outputs to.")
    parser.add_argument("-p", "--num-processors", type=int, default=8,
                        help="Number of processors to use. Use 0 for max.")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    #print(df)
    #print(df.iloc[1])
    for i in range(args.startIndex, args.stopIndex):
        values = df.iloc[i]
        if values["Queried"]:
            print(f"\nSlice {i} at index {values['xStart']} and time {values['timestep']} is already queried.")
        else:
            print(f"\nQuerying slice {i} at index {values['xStart']} and time {values['timestep']}.")
            #print(f"Size is {values['xEnd']-values['xStart']+1}")
            timestep = values["timestep"]
            index = values["xStart"]
            size = values['xEnd']-values['xStart']+1
            axis = 0
            output = f"{args.output_path}T{timestep}-X{index}-S{size}-A{axis}"
            print(timestep, index, size, axis)
            #process = Popen(["python", "getChannelSliceMultiprocessing.py", f"{timestep}", f"{index}", f"{output}",
            #                 "-s", f"{size}", "-a", f"{axis}", "--num-processors", f"{args.num_processors}"])
            #code = process.wait()
            code = 0
            if code == 0:
                try:
                    f = h5py.File(output+".h5", "r")
                    if f["xcoor"].shape[0] == 32:
                        df.loc[i, "Queried"] = True
                        df.to_csv(args.csv_path, index=False)
                        print("Sucessful")
                    else:
                        print(f"Generated h5dy file has wrong shape {f[tuple(f.keys())[0]].shape}")
                except Exception as e:
                    print(e)
            else:
                print(code)
                print("Failed to query")
    
