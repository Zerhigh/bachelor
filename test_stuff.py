import matplotlib.pyplot as plt
import os
import time
import numpy as np
import datetime
import random
import pickle
# change cv2s limitation on image size
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2



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
    plt.subplots_adjust(wspace=0, hspace=0.1)
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


def tile_image_w_overlap(image, parameter_dict, save_name):
    s = time.time()
    """
    image: uploadable image, should be square, starting in the lower left corner it will be squared
    tile_size: requiered size in m
    pixel_size: given size of a pixel
    overlap: overlap of two adjacent pictures in m
    """
    # define redundancy
    tile_size = parameter_dict['tile_size']
    pixel_size = parameter_dict['pixel_size']
    overlap = parameter_dict['overlap']
    save = parameter_dict['save']
    save_path = parameter_dict['save_path']
    tiling_mask = parameter_dict['tiling_mask']
    overlap_indices = parameter_dict['overlap_indices']

    # square image if necessary
    if image.shape[0] != image.shape[1]:
        image = square_image(image)

    #increment = image_size # tile_size * pixel_size
    no_images = int(len(overlap_indices)) #int(np.floor(image.shape[0]/(increment-overlap)))
    len_id = len(str(no_images))
    ret_dict = dict()

    print(no_images)

    for i in range(no_images):
        # create first index
        first_ind = list(str(i))
        while len(first_ind) <= len_id:
            first_ind.insert(0, '0')
        fi = ''.join(first_ind)
        for j in range(no_images):
            # create second index
            second_ind = list(str(j))
            while len(second_ind) <= len_id:
                second_ind.insert(0, '0')
            si = ''.join(second_ind)
            # print(f'{(i*(increment - overlap))}:{(i+1)*increment - i*overlap}, {j*(increment - overlap)}:{(j+1)*increment - j*overlap}')
            # add_image = image[(i*(increment - overlap)) : (i+1)*increment - i*overlap, j*(increment - overlap) : (j+1)*increment - j*overlap, :]
            add_image = image[overlap_indices[i][0]:overlap_indices[i][1], overlap_indices[j][0]:overlap_indices[j][1], :]
            if save:
                if (not tiling_mask and len(np.unique(add_image)) > 1) or (tiling_mask and f'{save_name.split(".")[0]}_{fi}_{si}.tif' in tiled_images):
                    cv2.imwrite(f'{save_path}{save_name.split(".")[0]}_{fi}_{si}.tif', add_image)
                    print(f'saved {save_path}{save_name.split(".")[0]}_{fi}_{si}.tif')
            # check and dont include monocolor areas
            if len(np.unique(add_image)) > 5:
                ret_dict[f'img_{fi}_{si}'] = add_image
    e = time.time()
    #print(f'tiling {image.shape[0]}x{image.shape[0]} image with tile size {tile_size} [m] and overlap of {overlap} [m] took {e-s} [s]')
    return ret_dict


def save_images(path, folder_name, image_dict, params):
    s = time.time()
    os.mkdir(f'{path}tiled_images/{folder_name}')
    for key, image in image_dict.items():
        cv2.imwrite(f'{path}tiled_images/{folder_name}/{key}.tif', image)
    with open(f'{path}tiled_images/{folder_name}/README.txt', 'w') as file:
        file.write(f'tiling image {folder_name} on {datetime.datetime.now().isoformat()} \n')
        for key, value in params.items():
            file.write('%s:%s\n' % (key, value))

    e = time.time()
    print(f'saving {len(image_dict)} images took {e - s} [s]')


def determine_overlap(img_size, wish_size):
    num_pics = int(np.ceil(img_size/wish_size))
    applied_step = int((num_pics * wish_size - img_size) / (num_pics - 1))
    overlap_indices = [(i*(wish_size-applied_step), (i+1)*wish_size - i*applied_step) for i in range(num_pics)]
    print(overlap_indices)
    return overlap_indices

start = time.time()
print(os.getcwd())

base_source = 'C:/Users/shollend/bachelor/test_data/train/'

params_masks = {'tile_size': 1300,
          'pixel_size': 0.3,
          'overlap': 0,
          'save': True,
          'save_path': f'{base_source}/tiled512_overlap/masks2m/',
          'tiling_mask': True,
          'overlap_indices': determine_overlap(1300, 512)}

params_images = {'tile_size': 1300,
          'pixel_size': 0.3,
          'overlap': 0,
          'save': True,
          'save_path': f'{base_source}/tiled512_overlap/images/',
          'tiling_mask': False,
          'overlap_indices': determine_overlap(1300, 512)}

tiled_images = []

### tile RGB images ###

for img_name in os.listdir(f'{base_source}/images'):
    img = cv2.imread(f'{base_source}/images/{img_name}')
    res1 = tile_image_w_overlap(img, params_images, img_name)

tiled_images = os.listdir(f'{base_source}/tiled512_overlap/images/')

### tile masks ###

for img_name in os.listdir(f'{base_source}/masks2m'):
    img = cv2.imread(f'{base_source}/masks2m/{img_name}')
    res1 = tile_image_w_overlap(img, params_masks, img_name)


"""
base_source = 'C:/Users/Samuel/Desktop/TU/BachelorArbeit/maxar/'

source = '10300100D94F1700-visual'
params = {'tile_size': 1500,
          'pixel_size': 1,
          'overlap': 0}

folder_name = f'{source}_t{params["tile_size"]}_p{params["pixel_size"]}_o{params["overlap"]}'
if folder_name not in f'{os.listdir(base_source+"/tiled_images")}':
    print('image files are not yet tiled and saved, this may take a while..')
    start2 = time.time()
    img = cv2.imread(f'{base_source+source}.tif')
    stop2 = time.time()
    print(f'time of image reading [s]: {round(stop2 - start2, 5)}')
    res1 = tile_image_w_overlap(img, params)
    save_images(base_source, folder_name, res1, params)
else:
    print(f'{source} is already tiled with these parameters')
"""
stop = time.time()
print(f'time of script running [s]: {round(stop-start, 5)}')

#plotting_result_images(res1)
#plt.show()
# cv2.imshow('image', img)
# cv2.waitKey(0)


