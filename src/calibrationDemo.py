from cp_hw6 import computeIntrinsic, computeExtrinsic, pixel2ray
import os
import glob
import numpy as np

#Example for running intrinsic and extrinsic calibration on provided structured light examples

#Input data locations
baseDir = '../data' #data directory
objName = 'remote2' #object name (should correspond to a dir in data)
# objName = 'cow2' #object name (should correspond to a dir in data)
# objName = 'remote' #object name (should correspond to a dir in data)
# objName = 'cow' #object name (should correspond to a dir in data)
# objName = 'frog' #object name (should correspond to a dir in data)
# seqName = 'v1' #sequence name (subdirectory of object)
seqName = '' #sequence name (subdirectory of object)
# calName = 'calib' #calibration sourse (also a dir in data)
# calName = 'mycalib/' #calibration sourse (also a dir in data)
calName = 'mycalib2/' #calibration sourse (also a dir in data)
image_ext = 'jpg' #file extension for images
useLowRes = False #enable lowres for debugging

#Extrinsic calibration parameters
dW1 = (8, 8) #window size for finding checkerboard corners
checkerboard = (6, 8) #number of internal corners on checkerboard

#Intrinsic calibration parameters
# dX = 558.8 #calibration plane length in x direction
# dY = 303.2125 #calibration plane length in y direction
dX = 190.5 #calibration plane length in x direction
dY = 127.0 #calibration plane length in y direction
dW2 = (8, 8) #window size finding ground plane corners

if useLowRes:
    calName += '-lr'
    seqName += '-lr'

#Part 1: Intrinsic Calibration
images = glob.glob(os.path.join(baseDir, calName, "*"+image_ext))
print(os.path.join(baseDir, calName, "*"+image_ext))
mtx, dist = computeIntrinsic(images, checkerboard, dW1)
#write out intrinsic calibration parameters
np.savez(os.path.join(baseDir, calName, "intrinsic_calib.npz"), mtx=mtx, dist=dist)

#Part 2: Extrinsic Calibration
#load intrinsic parameters
with np.load(os.path.join(baseDir, calName, "intrinsic_calib.npz")) as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]


#obtain extrinsic calibration from reference plane
firstFrame = os.path.join(baseDir, objName, seqName, '000001.jpg')
print(firstFrame)
print("Perform horizontal extrinsic calibration")
tvec_h, rmat_h = computeExtrinsic(firstFrame, mtx, dist, dX, dY)

print("Perform vertical extrinsic calibration")
tvec_v, rmat_v = computeExtrinsic(firstFrame, mtx, dist, dX, dY)

ext_out = {"tvec_h":tvec_h, "rmat_h":rmat_h, "tvec_v":tvec_v, "rmat_v":rmat_v}
np.savez(os.path.join(baseDir, objName, seqName, "extrinsic_calib.npz"), **ext_out)
