import rasterio
import rasterio.plot
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


source = "./test_data/"
data_nameMS = ['MS/'+file for file in os.listdir(source+'MS/')]
data_namePAN = ['PAN/'+file for file in os.listdir(source+'PAN/')]
data_namePSMS = ['PS-MS/'+file for file in os.listdir(source+'PS-MS/')]
data_namePSRGB = ['PS-RGB/'+file for file in os.listdir(source+'PS-RGB/')]


def show_MS(filename):
    counter = 1
    fig1, ax1 = plt.subplots(4, 2)
    for i in range(4):
        for j in range(2):
            tiff = rasterio.open(f'{filename}')
            pic = tiff.read(counter)
            ax1[i, j].imshow(pic)
            ax1[i, j].set_axis_off()
            ax1[i, j].set_title(f'{filename.split("/")[-1]} band {counter}', fontsize=5)
            counter += 1
    plt.show()

def show_PAN(filenamePAN):
    fig2, ax2 = plt.subplots()
    tiff = rasterio.open(f'{filenamePAN}')
    pic = tiff.read(1)
    ax2.imshow(pic)
    ax2.set_axis_off()
    ax2.set_title(f'{filenamePAN.split("/")[-1]}', fontsize=5)
    plt.show()

def show_PSMS(filenamePAN):
    fig3, ax3 = plt.subplots()
    tiff = rasterio.open(f'{filenamePAN}')
    pic = tiff.read(1)
    ax3.imshow(pic)
    ax3.set_axis_off()
    ax3.set_title(f'{filenamePAN.split("/")[-1]}', fontsize=5)
    plt.show()

def show_PSRGB(filenamePAN):
    img = cv2.imread(filenamePAN, cv2.IMREAD_UNCHANGED)
    print(f'dtype: {img.dtype}, shape: {img.shape}, min: {np.min(img)}, max: {np.max(img)}')
    cv2.imshow('tiff', img)
    cv2.waitKey(0)
    """fig4, ax4 = plt.subplots()
    tiff = rasterio.open(f'{filenamePAN}')
    print((tiff.read(1), tiff.read(2), tiff.read(3)))
    pic = np.dstack((tiff.read(1), tiff.read(2), tiff.read(3)))
    print(pic.dtype)
    ax4.imshow(pic)
    ax4.set_axis_off()
    ax4.set_title(f'{filenamePAN.split("/")[-1]}', fontsize=5)
    plt.show()"""

#show_MS(f'{source}{data_nameMS[0]}')
#show_PAN(f'{source}{data_namePAN[0]}')
#show_PSMS(f'{source}{data_namePSMS[0]}')
show_PSRGB(f'{source}{data_namePSRGB[0]}')


#rasterio.plot.show(tiff, title = "fe in the Medierranean Sea")
"""from osgeo import gdal
import matplotlib.pyplot as plt

ds = gdal.Open('/test_data/SN3_roads_train_AOI_3_Paris_MS_img6.tif', gdal.GA_ReadOnly)
rb = ds.GetRasterBand(1)
img_array = rb.ReadAsArray()\

fig1, ax1 = plt.subplots(1, 1)
plt.subplots_adjust(wspace=0, hspace=0.1)

ax1.imshow(img_array)
ax1.set_axis_off()"""