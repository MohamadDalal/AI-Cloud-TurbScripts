import h5py

if __name__=="__main__":
    #f = h5py.File("bigTest32MPFast.h5", "r")
    f = h5py.File("../dataSent/T3234-X181-S32-A0.h5", "r")
    print(f.keys())
    print(f[tuple(f.keys())[0]].shape)
    print(f["xcoor"].shape[0] == 32)
    print(f["ycoor"].shape)
    print(f["zcoor"].shape)