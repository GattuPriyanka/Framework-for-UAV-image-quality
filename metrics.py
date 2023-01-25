# This Python file uses the following encoding: utf-8
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QFileDialog, QLabel
from PySide2.QtCore import Signal, QThread, Slot
import cv2
import os
import sys
from os import listdir
from os import path
from os.path import join
import math
import numpy as np
import multiprocessing 
from scipy import fftpack
from niqe import niqe
from libsvm import svmutil
from brisque import BRISQUE

batchsize = 5

# Method to compute Blur in images 
def fourier(i, f1, DFTBlur, DCTBlur):
        # read RGB image as grayscale
        img = cv2.imread(f1,0)
        # get height of the image
        height = img.shape[0]
        # get width of the image
        width = img.shape[1]

        # perform FFT using numpy's FFT function
        f = np.fft.fft2(img)         
        # shift the zero frequency component (DC component) of the result to the center
        x = np.log(1+np.abs(f))     
        fshift = 255* (x/np.max(x))
        fs = fshift - np.mean(fshift)
        k = np.sum(np.power(fs,3))
        m = (1/width*height)*np.sum(np.power(fs,2))
        # computing blur
        dftBlur = (1/width*height)* (k/(np.power(math.sqrt(m),3)))
        
        # # perform DCT using fftpack's DCT function
        c = fftpack.dct(fftpack.dct(img, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')
        x1 = np.log(1+np.abs(c))     
        cshift = 255* (x1/np.max(x1))
        cs = cshift - np.mean(cshift)
        k1 = np.sum(np.power(cs,3))
        m1 = (1/width*height)*np.sum(np.power(cs,2))
        # computing blur
        dctBlur = (1/width*height)* (k1/(np.power(math.sqrt(m1),3)))

        # append computed values to the lists
        DFTBlur[i] = dftBlur
        DCTBlur[i] = dctBlur


def computeBlurness(folderpath): 
        manager = multiprocessing.Manager()
        # list to hold image file paths
        imageList = []
        for f in listdir(folderpath):
                path = join(folderpath, f)
                newPath = path.replace(os.sep, '/')
                # imageList.append(str(join(self.path, f)))
                imageList.append(newPath)
        # creating lists to share between processes
        DFTBlur = manager.list(range(len(imageList)))
        DCTBlur = manager.list(range(len(imageList)))
        processes=[]

        # main loop
        for i in range(batchsize, len(imageList), batchsize):
                for j in range(i-batchsize,i):
                        p = multiprocessing.Process(target=fourier, args = (j, imageList[j], DFTBlur, DCTBlur))
                        p.start()
                        processes.append(p)
                for process in processes:
                        process.join()

        for k in range(i, len(imageList)):
                p=multiprocessing.Process(target=fourier, args = (k, imageList[k], DFTBlur, DCTBlur))
                p.start()
                processes.append(p)
        for process in processes:
                process.join()
        
        # compute avg and return the results 
        return sum(DFTBlur)/len(DFTBlur), sum(DCTBlur)/len(DCTBlur)


def pix(i, f1, OE, UE):
        # read RGB image as grayscale
        img = cv2.imread(f1, 0)
        # computing sum of all pixel intensities greater than 250
        pix_a = np.sum(img[:,:]>=250)
        # computing sum of all pixel intensities less than 5
        pix_b = np.sum(img[:,:]<=5)
        # appending results to lists
        OE[i] = pix_a
        UE[i] = pix_b


def computeNIQE(folderpath):
        manager = multiprocessing.Manager()
        # list to hold image file paths
        imageList = []
        for f in listdir(folderpath):
                imageList.append(str(join(folderpath, f)))
        # creating lists to share between processes
        BList = manager.list(range(len(imageList)))
        processes=[]

        # main loop
        for i in range(batchsize, len(imageList), batchsize):
                for j in range(i-batchsize,i):
                        p = multiprocessing.Process(target = niqe, args = (j, imageList[j], BList))
                        p.start()
                        processes.append(p)
                for process in processes:
                        process.join()

        for k in range(i, len(imageList)):
                # more details on NIQE method here: https://github.com/buyizhiyou/NRVQA
                p=multiprocessing.Process(target = niqe, args = (k, imageList[k], BList))
                p.start()
                processes.append(p)
        for process in processes:
                process.join()
        
        # compute avg and return the results
        return sum(BList)/len(BList)


def computePixelRanges(folderpath): 
        manager = multiprocessing.Manager()
        # list to hold image file paths
        imageList = []
        for f in listdir(folderpath):
                imageList.append(str(path.join(folderpath, f)))
        # creating lists to share between processes
        UE = manager.list(range(len(imageList)))
        OE = manager.list(range(len(imageList)))
        processes=[]

        # main loop
        for i in range(batchsize, len(imageList), batchsize):
                for j in range(i-batchsize,i):
                        p=multiprocessing.Process(target = pix, args = (j, imageList[j], UE, OE))
                        p.start()
                        processes.append(p)
                for process in processes:
                        process.join()

        for k in range(i, len(imageList)):
                p=multiprocessing.Process(target = pix, args = (k, imageList[k], UE, OE))
                p.start()
                processes.append(p)
        for process in processes:
                process.join()
        
        # compute avg and return the results
        return sum(OE)/len(OE), sum(UE)/len(UE)


def BrisqueMethod1(i, f1, BList):
        img = cv2.imread(f1)
        # BRISQUE() method is imported from pybrisque module
        # for more details: https://github.com/bukalapak/pybrisque
        BList[i] = BRISQUE().get_score(img)


def computeBRSIQUE(folderpath):
        manager = multiprocessing.Manager()
        # list to hold image file paths
        imageList = []
        for f in listdir(folderpath):
                imageList.append(str(join(folderpath, f)))

        # creating lists to share between processes
        BList = manager.list(range(len(imageList)))
        processes=[]
    
        # main loop
        for i in range(batchsize, len(imageList), batchsize):
                for j in range(i-batchsize,i):
                        p=multiprocessing.Process(target=BrisqueMethod1,args=(j,imageList[j],BList))
                        p.start()
                        processes.append(p)
                for process in processes:
                        process.join()

        for k in range(i,len(imageList)):
                p=multiprocessing.Process(target=BrisqueMethod1,args=(k,imageList[k],BList))
                p.start()
                processes.append(p)
        for process in processes:
                process.join()

        # compute avg and return the results
        return sum(BList)/len(BList)

class ComputeMetrics(QThread):
        completeMsg = Signal(str)
        def __init__(self, path):
                super().__init__()
                self.path = path

        def run(self):
                dft,dct = computeBlurness(self.path)
                oe,ue = computePixelRanges(self.path)
                n = computeNIQE(self.path)
                b = computeBRSIQUE(self.path)
                msg = "DFT Blur: " + str(dft) + "\n" + "DCT Blur: " + str(dct) + "\n" + "Pixels (0-5): " + str(ue) + "\n" + "Pixels (250-255): " + str(oe) + "\n" + "NIQE: " + str(n) + "\n" + "BRISQUE Method 1: " + str(b) 
                self.completeMsg.emit(msg)


class MainWindow(QMainWindow):
        def __init__(self):
                QMainWindow.__init__(self)
                self.left = 10
                self.top = 10
                self.width = 320
                self.height = 240
                self.initUI()
                

        def initUI(self):
                self.setGeometry(self.left, self.top, self.width, self.height)
                self.setFixedSize(320,240)
                self.setWindowTitle("Quality Metrics")

                self.pathDisplayBox = QTextEdit(self)
                self.pathDisplayBox.unsetCursor()
                self.pathDisplayBox.setGeometry(10, 10, 250, 30)

                self.getPathButton = QPushButton("Data Path", self)
                self.getPathButton.setGeometry(260, 10, 60, 30)
                self.getPathButton.clicked.connect(self.getDataPath)

                self.computeButton = QPushButton("Compute Metrics", self)
                self.computeButton.setGeometry(100, 50, 100, 30)
                self.computeButton.clicked.connect(self.computeMetrics)

                self.label = QLabel(self)
                self.label.setFixedWidth(40)
                self.label.move(10, 70)
                self.label.setText("Output")
            
                self.outputDisplayBox = QTextEdit(self)
                self.outputDisplayBox.unsetCursor()
                self.outputDisplayBox.setGeometry(10, 100, 300, 120)

        def getDataPath(self):
                dlg = QFileDialog()
                self.path = dlg.getExistingDirectory()
                self.pathDisplayBox.setText(self.path)
                self.displayOutput(self.path)
        
        def computeMetrics(self):
                self.outputDisplayBox.setText("Computing Parameters. Please Wait..")
                self.myworker = ComputeMetrics(self.path)
                self.myworker.completeMsg.connect(self.displayOutput)
                self.myworker.start()
        
        @Slot(str)
        def displayOutput(self, msg):
                self.outputDisplayBox.setText(msg)

if __name__ == "__main__":
        multiprocessing.freeze_support()
        app = QApplication()
        window = MainWindow()
        window.show()
        app.exec_()
        sys.exit(app.exec_())
