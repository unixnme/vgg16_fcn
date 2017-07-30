# test vgg16 network without upsampling layer
import vgg16_fcn
import utils
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
import cv2
import keras
import os.path

model = vgg16_fcn.get_trained_model()
model = utils.convert_to_FCN(model)
model = utils.decapitate(model)

cmap = utils.color_map()
index = {}
for n in range(21):
    index[tuple(cmap[n])] = n
index[tuple(cmap[255])] = 255

image_dir = '/Users/ykang7/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages/'
ground_truth_dir = '/Users/ykang7/.keras/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass/'
filename = '/Users/ykang7/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'

with open(filename, 'r') as file:
    context = file.readlines()


img_size = (160, 160)
raw_imgs = []
true_imgs = []
for name in context:
    raw_imgs.append(os.path.join(image_dir, name[:-1] + '.jpg'))
    true_imgs.append(os.path.join(ground_truth_dir, name[:-1] + '.png'))

batch_size = 10
gen = utils.get_batches(raw_imgs, true_imgs, batch_size, (160, 160), (5, 5))
model.compile(keras.optimizers.Adadelta(), loss=keras.losses.sparse_categorical_crossentropy, metrics='accuracy')
model.fit_generator(generator=gen, steps_per_epoch=int(np.floor(len(raw_imgs)/float(batch_size))), epochs=1, verbose=1)


