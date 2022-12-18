from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from keras_segmentation.models.unet import vgg_unet
import time

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
"""
reassign_hash = {0: 0, 1: 128, 2: 2, 3: 130, 4: 102, 5: 70, 6: 9, 7: 107, 8: 112, 9: 48, 10: 51, 11: 119, 12: 254, 13: 153, 14: 28, 15: 190, 16: 255}

for arr in os.listdir(label_image_semantic):
    blab = np.asarray(Image.open(label_image_semantic + arr))[:, :, 0]
    image = blab.copy()
    for k, v in reassign_hash.items():
        image[image == v] = k
    im = Image.fromarray(image)
    im.save(f"./input/semantic_drone_dataset/training_set/gt/semantic/rehashed/{arr}")
    print(f"saved {arr}")
"""

### Train Model and display one result ###
# change epoch numbers
kaggle_commit = True
epochs = 20
if kaggle_commit:
    epochs = 5

# Aerial Semantic Segmentation Drone Dataset tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, fence-pole, window, door, obstacle

n_classes = 22
model = vgg_unet(n_classes=n_classes ,  input_height=416, input_width=608)

# train model vgg_unet (?)
model.train(
    train_images = "./input/semantic_drone_dataset/training_set/images/",
    train_annotations = "./input/semantic_drone_dataset/training_set/gt/semantic/rehashed/",
    checkpoints_path = "vgg_unet", epochs=epochs)

model.save('try1812_example')


### Predict result ###
"""
start = time.time()

input_image = "./input/semantic_drone_dataset/training_set/images/001.jpg"
out = model.predict_segmentation(
    inp=input_image,
    out_fname="out.png"
)

fig, axs = plt.subplots(1, 3, figsize=(20, 20), constrained_layout=True)

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

done = time.time()
elapsed = done - start
"""