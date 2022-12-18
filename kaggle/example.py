from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from keras_segmentation.models.unet import vgg_unet
import keras_segmentation
from keras.models import *
import time
import tensorflow as tf

from types import MethodType


original_image = "./input/semantic_drone_dataset/training_set/images/001.jpg"
label_image_semantic_RAW = "./input/semantic_drone_dataset/training_set/gt/semantic/label_images/001.png"
label_image_semantic_name = "./input/semantic_drone_dataset/training_set/gt/semantic/label_images/"
label_image_semantic = "./input/semantic_drone_dataset/training_set/gt/semantic/label_images/"

### Plotting ###
"""
for i in range(8):
    fig, axs = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)

    #axs[0].imshow(Image.open(original_image))
    #axs[0].grid(False)

    label_image_semantic_IMG = Image.open(label_image_semantic_name+f"00{i}.png")
    label_image_semantic = np.asarray(label_image_semantic_IMG)[:, :, 0]
    axs.imshow(label_image_semantic)
    axs.grid(False)
    plt.show()
"""

### Determine all unique pixel values from all images ###
"""
unique_values = set()
for arr in os.listdir(label_image_semantic):
    uniques = np.unique(np.asarray(Image.open(label_image_semantic + arr))[:, :, 0])
    for unique in uniques:
        if unique not in unique_values:
            unique_values.add(unique)
    print(f"did image {arr}")

print(uniques)
print(len(uniques))
"""

### Reassign all images ###

# reassign_hash = {0: 0, 1: 128, 2: 2, 3: 130, 4: 102, 5: 70, 6: 9, 7: 107, 8: 112, 9: 48, 10: 51, 11: 119, 12: 254, 13: 153, 14: 28, 15: 190, 16: 255}
# reassign_hash = {0: 0, 1: 128, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}
reassign_hash = {0: 0, 1: 128, 2: 2, 3: 130, 4: 102, 5: 70, 6: 9, 7: 107, 8: 112, 9: 48, 10: 51, 11: 119, 12: 254, 13: 153, 14: 28, 15: 190, 16: 255}

for arr in os.listdir(label_image_semantic):
    blab = np.asarray(Image.open(label_image_semantic + arr))[:, :, 0]
    image = blab.copy()
    image[image == 128] = 1
    image[image != 128] = 0
    #for k, v in reassign_hash.items():
    #    image[image == v] = k
    im = Image.fromarray(image)
    im.save(f"./input/semantic_drone_dataset/training_set/gt/semantic/rehashed/{arr}")
    print(f"saved {arr}")


### Train Model ###

# change epoch numbers
kaggle_commit = False
epochs = 20
if kaggle_commit:
    epochs = 5
epochs = 20

# Aerial Semantic Segmentation Drone Dataset tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, fence-pole, window, door, obstacle

n_classes = 2 #22
model = vgg_unet(n_classes=n_classes ,  input_height=416, input_width=608)

# train model vgg_unet (?)
model.train(
    train_images = "./input/semantic_drone_dataset/training_set/images/",
    train_annotations = "./input/semantic_drone_dataset/training_set/gt/semantic/rehashed/",
    checkpoints_path = "vgg_unet", epochs=epochs)

model_name = 'model_2class_1812_20e'
model.save(model_name)


### Load Model ###
"""
n_classes = 22
reconstructed_model = load_model("model_1812_example")
reconstructed_model.predict_segmentation = MethodType(keras_segmentation.predict.predict, reconstructed_model)
reconstructed_model.input_width = 608
reconstructed_model.input_height= 416
reconstructed_model.output_width = 208
reconstructed_model.output_height = 304
reconstructed_model.n_classes=n_classes
"""

### Predict result ###

start = time.time()

for arr in os.listdir(label_image_semantic)[:15]:
    input_image = f"./input/semantic_drone_dataset/training_set/images/{arr}"
    out = model.predict_segmentation(
        inp=input_image,
        out_fname=f"{arr.split('.')[0]}.png"
    )

fig, axs = plt.subplots(1, 2, figsize=(8, 8), constrained_layout=True)

img_orig = Image.open(input_image)
axs[0].imshow(img_orig)
axs[0].set_title('original image-001.jpg')
axs[0].grid(False)

axs[1].imshow(out)
axs[1].set_title('prediction image-out.png')
axs[1].grid(False)

validation_image = "./input/semantic_drone_dataset/training_set/gt/semantic/rehashed/001.png"
axs[2].imshow( Image.open(validation_image))
axs[2].set_title('true label image-001.png')
axs[2].grid(False)

plt.show()


### Use Pretrained model ###
"""from .models.all_models import model_from_name
def model_from_checkpoint_path(model_config, latest_weights):

    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    model.load_weights(latest_weights)
    return model

def pspnet_101_voc12():

    model_config = {
        "input_height": 473,
        "input_width": 473,
        "n_classes": 21,
        "model_class": "pspnet_101",
    }

    model_url = "https://www.dropbox.com/s/" \
                "uvqj2cjo4b9c5wg/pspnet101_voc2012.h5?dl=1"
    latest_weights = tf.keras.utils.get_file("pspnet101_voc2012.h5", model_url)

    return model_from_checkpoint_path(model_config, latest_weights)

model = pspnet_101_voc12()
"""