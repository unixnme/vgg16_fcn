from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('image')
args = parser.parse_args()

nb_classes = 21
# Bilinear interpolation (reference: https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/upsampling.py)
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

def fcn_32s_base():
    inputs = Input(shape=(None, None, 3))
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    x = Conv2D(filters=nb_classes,
               kernel_size=(1, 1))(vgg16.output)
    model = Model(inputs=inputs, outputs=x)
    return model

def fcn_32s_top():
    inputs = Input(shape=(None, None, 21))
    x = Conv2DTranspose(filters=nb_classes,
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)))(inputs)
    model = Model(inputs=inputs, outputs=x)
    return model

base = fcn_32s_base()
top = fcn_32s_top()
model = Model(inputs=base.input, outputs=top(base.output))

img_org = Image.open(args.image)
w_org, h_org = img_org.size
img = img_org.resize(((w_org//32)*32, (h_org//32)*32))
w, h = img.size
x = np.asarray(img, dtype=np.float32)
x = np.expand_dims(x, axis=0)

pred = base.predict(preprocess_input(x))
pred = pred[0].argmax(axis=-1).astype(np.uint8)
img = Image.fromarray(pred, mode='P')
palette_im = Image.open('palette.png')
img.palette = copy.copy(palette_im.palette)
img.save('predict_base.png')
img = img.resize((w_org, h_org), Image.BILINEAR)
img.save('predict_base_expanded.png')

pred = model.predict(preprocess_input(x))
pred = pred[0].argmax(axis=-1).astype(np.uint8)
img = Image.fromarray(pred, mode='P')
img = img.resize((w_org, h_org))
palette_im = Image.open('palette.png')
img.palette = copy.copy(palette_im.palette)
img.save('predict.png')