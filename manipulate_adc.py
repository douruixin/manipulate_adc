# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:26:56 2017

@author: Ray
"""

import pydicom  # pip install pydicom
import pylab
import os
import numpy as np
from matplotlib import pyplot as plt

# PathDicom = 'D:\\DicomImages\\DICOM'
# lstFilesDCM = []
# for dirName, subdirList,fileList in os.walk(PathDicom):
#    for filename in fileList:
#        lstFilesDCM.append(os.path.join(dirName,filename))

dwi1 = pydicom.read_file('/home/ray/CloudStation/MyWork/Python/radiology/data/00060054.dcm')  # GE b0
#dwi1 = pydicom.read_file('/home/ray/CloudStation/MyWork/Python/radiology/data/05010031.dcm')  # Philips b0
manufacturer = dwi1.Manufacturer.upper()
index = manufacturer.find("PHILIPS")
print(dwi1.dir("pat"))
print(dwi1.PatientName)

# 原始二进制文件
pixel_bytes = dwi1.PixelData
# 像素矩阵  
pix = dwi1.pixel_array

# 读取显示图片  
# pylab.imshow(dwi1.pixel_array, cmap=pylab.cm.gray)
# pylab.show()

dwi2 = pydicom.read_file('/home/ray/CloudStation/MyWork/Python/radiology/data/00060034.dcm')  # GE b1000
# dwi2 = pydicom.read_file('/home/ray/CloudStation/MyWork/Python/radiology/data/05010032.dcm')  # Philips b1000
print(dwi2.dir("pat"))
print(dwi2.PatientName)

# 原始二进制文件
pixel_bytes = dwi2.PixelData

from math import log
from math import e
dwi1_pixel_array = dwi1.pixel_array + 1  # b0
dwi2_pixel_array = dwi2.pixel_array + 1  # b1000

ADC = np.zeros(shape=(int(dwi2.Rows), int(dwi2.Columns)))
EADC = np.zeros(shape=(int(dwi2.Rows), int(dwi2.Columns)))
iADC = np.zeros(shape=(int(dwi2.Rows), int(dwi2.Columns)))

ConstPixelDims = (int(dwi2.Rows), int(dwi2.Columns), 1)
for i in range(ConstPixelDims[0]):
    for j in range(ConstPixelDims[1]):
        ADC[i, j] = log(dwi2_pixel_array[i, j]/dwi1_pixel_array[i, j])/-1000

for i in range(ConstPixelDims[0]):
    for j in range(ConstPixelDims[1]):
        EADC[i, j] = dwi2_pixel_array[i, j]/dwi1_pixel_array[i, j]

for i in range(ConstPixelDims[0]):
    for j in range(ConstPixelDims[1]):
        iADC[i, j] = 50 - ADC[i, j]

from pylab import *
plt.figure()
pylab.imshow(iADC, cmap=pylab.cm.gray)
pylab.show()

# ADC = (lnS1/S2)/b2-b1

# 阈值分割
import cv2
# ret, thresh = cv2.threshold(dwi2.pixel_array, 800, 1500, cv2.THRESH_BINARY) - GE
manufacturer = dwi1.Manufacturer.upper()
index = manufacturer.find("PHILIPS")
if index != -1:
    ret, thresh = cv2.threshold(dwi2.pixel_array, 130, 500, cv2.THRESH_BINARY)  # Philips
else:
    ret, thresh = cv2.threshold(dwi2.pixel_array, 800, 1500, cv2.THRESH_BINARY)  # GE
img = np.uint8(thresh)

# 降噪处理
kernel = np.ones((3, 3), np.uint8)
opening_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 标记轮廓
# 将阈值以上的像素位置赋值为1
mask = opening_thresh == 800
opening_thresh[mask] = 1
image = dwi1.pixel_array
img8 = cv2.convertScaleAbs(opening_thresh)  # 将 uint16 转换为 uint8

binary, contours, hierarchy = cv2.findContours(img8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

from PIL import Image
from pylab import *

x=[]
y=[]
for i in range(len(contours[0])):
    x.append(contours[0][i-1][0, 0])
    y.append(contours[0][i-1][0, 1])

plt.figure()
plt.subplot(1, 6, 1)
plt.title('b0')
pylab.imshow(dwi1.pixel_array, cmap=pylab.cm.gray)
plt.subplot(1, 6, 2)
plt.title('b1000')
pylab.imshow(dwi2.pixel_array, cmap=pylab.cm.gray)
plt.subplot(1, 6, 3)
plt.title('ADC')
pylab.imshow(ADC, vmin=-0.0001, vmax=0.004, cmap=pylab.cm.gray)
plt.subplot(1, 6, 4)
plt.title('EADC')
pylab.imshow(EADC, vmin=-0.02, vmax=0.5, cmap=pylab.cm.gray)
plt.subplot(1, 6, 5)
plt.title('Segment CI')
pylab.imshow(img, cmap=pylab.cm.gray)
plt.subplot(1, 6, 6)
plt.title('Identify CI')
pylab.imshow(ADC, vmin=-0.0001, vmax=0.004, cmap=pylab.cm.gray)
# plt.plot(x[:len(contours[0])], y[:len(contours[0])], '-', color='red')
plt.plot(x[:len(contours[0])], y[:len(contours[0])], '-', color='red')
pylab.show()
