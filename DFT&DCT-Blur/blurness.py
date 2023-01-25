import cv2
from os import listdir
from os.path import join
import math
import numpy as np
import time
import multiprocessing 
from scipy import fftpack


def fourier(i, f1, DFTBlur, DCTBlur):
    img = cv2.imread(f1,0)
    height = img.shape[0]
    width = img.shape[1]

    f = np.fft.fft2(img)         
    x=np.log(1+np.abs(f))     
    fshift = 255* (x/np.max(x))
    fs = fshift - np.mean(fshift)
    k = np.sum(np.power(fs,3))
    m = (1/width*height)*np.sum(np.power(fs,2))
    dftBlur = (1/width*height)* (k/(np.power(math.sqrt(m),3)))
    
    c = fftpack.dct(fftpack.dct(img, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')
    x1=np.log(1+np.abs(c))     
    cshift = 255* (x1/np.max(x1))
    cs = cshift - np.mean(cshift)
    k1 = np.sum(np.power(cs,3))
    m1 = (1/width*height)*np.sum(np.power(cs,2))
    dctBlur = (1/width*height)* (k1/(np.power(math.sqrt(m1),3)))

    DFTBlur[i] = dftBlur
    DCTBlur[i] = dctBlur


if __name__ == '__main__': 
    folderpath = "D:\Droneflights\\15-02-2022\RCW18\RCW18GROUNDNUT\F3(40mtr)"
    manager = multiprocessing.Manager()
    
    
    imageList = []
    for f in listdir(folderpath):
        imageList.append(str(join(folderpath, f)))
    DFTBlur = manager.list(range(len(imageList)))
    DCTBlur = manager.list(range(len(imageList)))
    processes=[]
    import time
    start_time1 = time.time()

    batchsize = 10
    for i in range(batchsize, len(imageList), batchsize):
        for j in range(i-batchsize,i):
            p=multiprocessing.Process(target=fourier,args=(j,imageList[j],DFTBlur,DCTBlur))
            p.start()
            processes.append(p)
        for process in processes:
            process.join()

    for k in range(i,len(imageList)):
        p=multiprocessing.Process(target=fourier,args=(k,imageList[k],DFTBlur,DCTBlur))
        p.start()
        processes.append(p)
    for process in processes:
        process.join()
        
    print("--- %s seconds ---" % (time.time() - start_time1))
    print((sum(DFTBlur)/len(DFTBlur)),"dft")
    print((sum(DCTBlur)/len(DCTBlur)),"dct")
