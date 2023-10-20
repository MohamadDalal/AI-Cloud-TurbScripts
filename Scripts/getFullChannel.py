import argparse
import pyJHTDB

def oddNumber(str):
    num = int(str)
    if num%2==0:
        #raise argparse.ArgumentTypeError
        raise ValueError
    else:
        return num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script querys a slice from the channel database in JHTDB",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("timestep", type=int,
                        help="The time step to be queried")
    parser.add_argument("index", type=int,
                        help="Index to slice at in the selected axis.")
    parser.add_argument("-a", "--axis", type=int, default='0',
                        help="The axis to slice on. X-axis=0, Y-axis=1, Z-axis=2")
    parser.add_argument("-s", "--size", type=oddNumber, default='31',
                        help="Slice size to be queried. Has to be an odd number.")
    args = parser.parse_args()
    print(args.timestep, args.index, args.axis, args.size)
    with open("authToken.txt", "r") as f:
        authToken = f.read()
    print(authToken)

    