import argparse
import numpy as np
import pyJHTDB
import ctypes

def setParams(timestep, index, axis, size):
    start = np.array((1, 1, 1, timestep), dtype=int)
    end = np.array((2048, 512, 1536, timestep), dtype=int)
    #end = np.array((32, 33, 32, timestep), dtype=int)
    step = np.ones(4, dtype=int)
    upperBound = np.array((2048, 512, 1536, 4000), dtype=int)
    start[axis] = index
    end[axis] = index + size - 1
    if timestep < 0 or timestep >= upperBound[3]:
        print("timestep provided is not within 1 and 4000")
        raise argparse.ArgumentError
    elif start[axis] < 1 or end[axis] > upperBound[axis]:
        print(f"index provided is not within 1 and {upperBound[axis]}")
        raise argparse.ArgumentError
    return start, end, step



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script querys a slice from the channel database in JHTDB",
                                     epilog="Note that the resulting h5 array has shape [zAxis,yAxis,xAxis,values]",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("timestep", type=int,
                        help="The time step to be queried")
    parser.add_argument("index", type=int,
                        help="Index to start slicing at in the selected axis.")
    parser.add_argument("output", type=str,
                        help="Name or path of the output file. Choosing name of already exisiting file raises an error.")
    parser.add_argument("-a", "--axis", type=int, default='0', choices=(0,1,2),
                        help="The axis to slice on. X-axis=0, Y-axis=1, Z-axis=2")
    parser.add_argument("-s", "--size", type=int, default='31',
                        help="Slice size to be queried. End index will be index + size.")
    parser.add_argument("-p", "--pressure", action='store_true',
                        help="Get pressure instead of velocity.")
    args = parser.parse_args()
    print(args.timestep, args.index, args.output, args.axis, args.size, args.pressure)
    with open("authToken.txt", "r") as f:
        authToken = f.read()
    print(authToken.strip())

    #startArr, endArr, stepArr = setParams(args.timestep, args.index, args.axis, args.size)
    function = "p" if args.pressure else "u"
    #print(startArr)
    #print(endArr)
    #print(stepArr)
    #print(function)

    lJHTDB = pyJHTDB.libJHTDB()
    lJHTDB.initialize()
    lJHTDB.lib.turblibSetExitOnError(ctypes.c_int(0));
    lJHTDB.add_token(authToken.strip())

    ## "filename" parameter is the file names of output files, if filename='N/A', no files will be written. 
    ##             For example, if filename='results', the function will write "results.h5" and "results.xmf".
    ## The function only returns the data at the last time step within [t_start:t_step:t_end]
    ## The function only returns the data in the last field. For example, result=p if field=[up].
    #result = lJHTDB.getbigCutout(
    #        data_set="channel", fields=function, filename=args.output,
    #        t_start=startArr[3], t_end=endArr[3], t_step=stepArr[3],
    #        start=startArr[:3], end=endArr[:3], step=stepArr[:3])
    startPoint = args.index
    for i in range((args.size//5)):
        print(f"Starting query at index {i}")
        startArr, endArr, stepArr = setParams(args.timestep, startPoint, args.axis, 5)
        temp = lJHTDB.getCutout(
                        data_set="channel", field=function, time_step=startArr[3],
                        start=startArr[:3], end=endArr[:3], step=stepArr[:3])
        if temp.shape[2-args.axis] != 5:
            print("Error. Queried array is of wrong shape: ", temp.shape)
            exit()
        else:
            np.save(f"T{args.timestep}-I{args.index}-S{args.size}-A{args.axis}-Temp{i}.npy", temp)
            startPoint += 5
    if (args.size % 5) != 0:
        print(f"Starting query at last index")
        startArr, endArr, stepArr = setParams(args.timestep, startPoint, args.axis, args.size-((args.size//5)*5))
        temp = lJHTDB.getCutout(
                        data_set="channel", field=function, time_step=startArr[3],
                        start=startArr[:3], end=endArr[:3], step=stepArr[:3])
        if temp.shape[2-args.axis] != 5:
            print("Error. Queried array is of wrong shape: ", temp.shape)
            exit()
        else:
            np.save(f"T{args.timestep}-I{args.index}-S{args.size}-A{args.axis}-TempLast.npy", temp)
    lJHTDB.finalize()

    FullArrShape = [2048, 512, 1536]
    FullArrShape[args.axis] = args.size
    if args.pressure:
        FullArrShape = tuple(list(np.array(FullArrShape)[::-1]) + [1])
    else:
        FullArrShape = tuple(list(np.array(FullArrShape)[::-1]) + [3])
    resultArr = np.zeros(FullArrShape, dtype=np.float32)
    for i in range((args.size//5)):
        if args.axis == 0:
            resultArr[:,:,i*5:(i+1)*5,:] = np.load(f"T{args.timestep}-I{args.index}-S{args.size}-A{args.axis}-Temp{i}.npy")
        elif args.axis == 1:
            resultArr[:,i*5:(i+1)*5,:,:] = np.load(f"T{args.timestep}-I{args.index}-S{args.size}-A{args.axis}-Temp{i}.npy")
        elif args.axis == 2:
            resultArr[i*5:(i+1)*5,:,:,:] = np.load(f"T{args.timestep}-I{args.index}-S{args.size}-A{args.axis}-Temp{i}.npy")
    if (args.size % 5) != 0:
        if args.axis == 0:
            resultArr[:,:,(i+1)*5:,:] = np.load(f"T{args.timestep}-I{args.index}-S{args.size}-A{args.axis}-TempLast.npy")
        elif args.axis == 1:
            resultArr[:,(i+1)*5:,:,:] = np.load(f"T{args.timestep}-I{args.index}-S{args.size}-A{args.axis}-TempLast.npy")
        elif args.axis == 2:
            resultArr[(i+1)*5:,:,:,:] = np.load(f"T{args.timestep}-I{args.index}-S{args.size}-A{args.axis}-TempLast.npy")

    # h5dy saving copied directly from get big cutout
    startArr, endArr, stepArr = setParams(args.timestep, args.index, args.axis, args.size)
    nl = '\r\n'
    hdf5_file, xdmf_file, shape=lJHTDB.hdf5_init(args.output, "channel",startArr[3],endArr[3],stepArr[3],
                                                 startArr[:3],endArr[:3],stepArr[:3],1,
                                                 np.arange(startArr[0], endArr[0]+1, stepArr[0]),
                                                 np.arange(startArr[1], endArr[1]+1, stepArr[1]),
                                                 np.arange(startArr[2], endArr[2]+1, stepArr[2]))
    VarName = "Pressure" if args.pressure else "Velocity"
    print(f"    <Grid Name=\"{VarName}\" GridType=\"Collection\" CollectionType=\"Temporal\">{nl}", file=xdmf_file)
    dim = 1 if args.pressure else 3
    lJHTDB.hdf5_writing(args.output,resultArr,"channel",VarName,dim,args.timestep,hdf5_file,xdmf_file,shape)
    print(f"    </Grid>{nl}", file=xdmf_file)
    lJHTDB.hdf5_end(hdf5_file,xdmf_file)
    #print(result.shape)

    