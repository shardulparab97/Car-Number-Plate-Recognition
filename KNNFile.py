import cv2
import numpy as np

#The file is used to run the KNN training
KNN = cv2.ml.KNearest_create()

def loadKNNData():
    #For clustering - mapping accordingly here
    Labels = np.loadtxt("Labels.txt", np.float32) #loading labels
    ImageData = np.loadtxt("imageData.txt", np.float32) #loading image data array
    Labels = Labels.reshape((Labels.size,1)) #reshaping to 1D array in order to run training
    KNN.setDefaultK(1) #setting default K as 1 , can be changed later on
    KNN.train(ImageData, cv2.ml.ROW_SAMPLE, Labels) #training for KNN
    return True
