from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from kaggle.models.unet import vgg_unet
from kaggle.models.unet import vgg_unet, resnet50_unet
from kaggle.predict import predict
from keras.models import load_model
import tensorflow as tf

from types import MethodType

# old images
"""
original_image = "./input/semantic_drone_dataset/training_set/images/001.jpg"
label_image_semantic_rehashed = "./input/semantic_drone_dataset/training_set/gt/semantic/rehashed/"
label_image_semantic_name = "./input/semantic_drone_dataset/training_set/gt/semantic/label_images/"
label_image_semantic = "./input/semantic_drone_dataset/training_set/gt/semantic/label_images/"
"""
# Paris images
label_image_semantic_rehashed = 'C:/Users/shollend/bachelor/test_data/train/tiled512/rehashed/'
label_image_semantic_masks = 'C:/Users/shollend/bachelor/test_data/train/tiled512/masks2m/'
label_image_semantic = 'C:/Users/shollend/bachelor/test_data/train/tiled512/images/'
images_original = 'C:/Users/shollend/bachelor/test_data/train/images/'


### Plotting ###
def plotting(path):
    for i in range(1):
        fig, axs = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)

        #axs[0].imshow(Image.open(original_image))
        #axs[0].grid(False)

        # label_image_semantic_IMG = Image.open(path+f"00{i}.png")
        print(path)
        label_image_semantic_IMG = Image.open(label_image_semantic_rehashed+path)
        label_image_semantic = np.asarray(label_image_semantic_IMG)[:, :]
        axs.imshow(label_image_semantic)
        axs.grid(False)
        plt.show()

# plotting(os.listdir(label_image_semantic_rehashed)[0])

### Determine all unique pixel values from all images ###
def unique_pixel_vals(path):
    unique_values = set()
    for arr in os.listdir(path):
        uniques = np.unique(np.asarray(Image.open(path + arr))[:, :, 0])
        for unique in uniques:
            if unique not in unique_values:
                unique_values.add(unique)
        print(f"did image {arr}")

    print(uniques)
    print(len(uniques))
    return uniques

#unique_pixels = unique_pixel_vals(label_image_semantic)

### Reassign all images ###

# reassign_hash = {0: 0, 1: 128, 2: 2, 3: 130, 4: 102, 5: 70, 6: 9, 7: 107, 8: 112, 9: 48, 10: 51, 11: 119, 12: 254, 13: 153, 14: 28, 15: 190, 16: 255}
# reassign_hash = {0: 0, 1: 128, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}
reassign_hash = {0: 0, 1: 128, 2: 2, 3: 130, 4: 102, 5: 70, 6: 9, 7: 107, 8: 112, 9: 48, 10: 51, 11: 119, 12: 254,
                 13: 153, 14: 28, 15: 190, 16: 255}
reassign_hash_Paris = {255: 1}

def rehash_images(rehash, inpath, outpath):
    for arr in os.listdir(f'{inpath}'):
        print(arr)
        # old images blab = np.asarray(Image.open(f"{inpath}masks2m/{arr}"))[:, :, 0]
        blab = np.asarray(Image.open(f"{inpath}{arr}"))[:, :]
        image = blab.copy()
        #image[image != 128] = 0
        image[image == 255] = 1
        #for k, v in rehash.items():
        #    image[image == v] = k
        im = Image.fromarray(image)
        im.save(f"{outpath}{arr}")
        print(f"saved {arr}")

# rehash_images(reassign_hash_Paris, label_image_semantic_masks, label_image_semantic_rehashed)

### Train Model ###

def train_model(model_name_assign, epochs, n_classes, hw, train_images, train_annotations):
    # Aerial Semantic Segmentation Drone Dataset tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, fence-pole, window, door, obstacle

    n_classes = n_classes# 2 #2 #22
    model = vgg_unet(n_classes=n_classes,  input_height=hw[0], input_width=hw[1])
    # model = resnet50_unet(n_classes=n_classes, input_height=hw[0], input_width=hw[1])

    # train model vgg_unet (?)
    model.train(
        train_images = train_images,
        train_annotations = train_annotations,
        checkpoints_path = "vgg_unet", epochs=epochs, optimizer_name='adam')
    # train resnet
    """model.train(
        train_images=train_images,
        train_annotations=train_annotations,
        checkpoints_path="resnet50_unet", epochs=epochs, optimizer_name='adam') #adam"""

    model.save(model_name_assign)
    return model

model_name = 'paris_model_2class_0401_30e_512x512_resnet_binarycrossentropy_valsplit0_1_adam'
# model = train_model(model_name, 15, 2, (512, 512), label_image_semantic, label_image_semantic_rehashed)

### Load Model ###
def load_model_wparams(model_name_get, inp_hw, out_hw, n_classes):
    reconstructed_model = load_model(model_name_get)
    reconstructed_model.predict_segmentation = MethodType(predict, reconstructed_model)
    reconstructed_model.input_width = inp_hw[0] #512 #256
    reconstructed_model.input_height = inp_hw[1]#512 #256
    reconstructed_model.output_width = out_hw[0] #256 #128#256#704# 208
    reconstructed_model.output_height = out_hw[1] #256 #988# 304
    reconstructed_model.n_classes = n_classes
    return reconstructed_model

rec_model = load_model_wparams(model_name, (512, 512), (256, 256), 2)

### Predict result ###
def predict_results(model, inp_path, out_path):
    for arr in os.listdir(inp_path):
        # input_image = f"{inp_path}{arr.split('.')[0]}.jpg"
        # print(arr)
        # print(f"{inp_path}{arr}")
        # print(f"{out_path}{arr.split('.')[0]}.jpg")
        input_image = f"{inp_path}{arr}"
        out = model.predict_segmentation(
            inp=input_image,
            out_fname=f"{out_path}{arr}"
        )

# predict_results(model, label_image_semantic, f"./output/{model_name}/")
predict_results(rec_model, images_original, f"./output/{model_name}_full_size/")

### Plot result ###
"""
model_name = "model_2class_1912_10e"

for img in os.listdir(f"./output/{model_name}/")[:5]:
    fig, axs = plt.subplots(1, 3, figsize=(8, 8), constrained_layout=True)
    img_orig = Image.open(f"./input/semantic_drone_dataset/training_set/images/{img.split('.')[0]}.jpg")
    axs[0].imshow(img_orig)
    axs[0].set_title('original image')
    axs[0].grid(False)

    img_new = Image.open(f"./output/{model_name}/{img}")
    axs[1].imshow(img_new)
    axs[1].set_title('prediction image-out.png')
    axs[1].grid(False)

    validation_image = f"./input/semantic_drone_dataset/training_set/gt/semantic/rehashed/{img.split('.')[0]}.png"
    axs[2].imshow( Image.open(validation_image))
    axs[2].set_title('true label image-001.png')
    axs[2].grid(False)

    plt.show()

"""

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

### Get Image from Geojson ###
# from apls import apls