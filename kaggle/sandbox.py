import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from osgeo import gdal
import kaggle.example as ex

print('sandbox')

pathh = 'C:/Users/shollend/bachelor/test_data/train/rehashed/test_data_img137.tif'
pathh = 'C:/Users/shollend/bachelor/kaggle/output/model_2class_1912_10e/000.png'
pathh = [x for x in os.listdir('C:/Users/shollend/bachelor/test_data/train/images/')][0]


fig, axs = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)

label_image_semantic_IMG = Image.open(pathh)
label_image_semantic = np.asarray(label_image_semantic_IMG)[:, :]
print(label_image_semantic.shape)
print(np.unique(label_image_semantic))

axs.imshow(label_image_semantic)
axs.grid(False)
plt.show()