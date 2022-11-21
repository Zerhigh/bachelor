import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time


def get_all_images(path, data_type):
    image_names = [img_path for img_path in os.listdir(path) if f'.{data_type}' in img_path]
    return image_names


def square_image(image):
    squared_image = image[:min(image.shape[:2]), :min(image.shape[:2]), :]
    return squared_image


def plotting_result_images(image_list):
    sizer_res1 = int(np.sqrt(len(image_list)))
    index = 0
    fig1, ax1 = plt.subplots(sizer_res1, sizer_res1)
    plt.subplots_adjust(wspace=0.000001, hspace=0.001)
    for i in range(sizer_res1):
        for j in range(sizer_res1):
            ax1[i, j].imshow(image_list[index])
            ax1[i, j].set_axis_off()
            index += 1
    return 0


def tile_image(image, tile_size, pixel_size, overlap):
    """
    image: uploadable image, should be square, starting in the lower left corner it will be squared
    tile_size: requiered size in m
    pixel_size: given size of a pixel
    overlap: overlap of two adjacent pictures in m
    """
    # square image if necessary
    if image.shape[0] != image.shape[1]:
        image = square_image(image)
    increment = tile_size * pixel_size
    no_images = int(np.floor(image.shape[0]/increment))
    ret_list = list()
    for i in range(no_images-1):
        for j in range(no_images-1):
            ret_list.append(image[i*increment : (i+1)*increment, j*increment : (j+1)*increment, :])

    return ret_list

start = time.time()
source = 'C:/Users/Samuel/Desktop/TU/BachelorArbeit/bing_exp_beijing_2.tif'
img = cv2.imread(source)
#cv2.imshow('image', img)
#cv2.waitKey(0)
res1 = tile_image(img, 100, 1, 0)

stop = time.time()
print(f'time of script running [s]: {round(stop-start, 5)}')

plotting_result_images(res1)
plt.show()


