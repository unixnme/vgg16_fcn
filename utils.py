import keras
from keras.models import Model
from keras.layers import Flatten, Input, Conv2D, InputLayer, MaxPooling2D, Dropout, Dense, Conv2DTranspose
from mnist_cnn import mnist_cnn, train_mnist
from vgg16_cnn import get_trained_model
from vgg16_fcn import test_model, preprocess_input
from mnist_fcn import test, train
from keras.preprocessing import image
import numpy as np
from colormap import color_map
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import cv2

def convert_to_FCN(model):
    if not isinstance(model, Model):
        raise Exception("model must be a valid Keras model")

    # currently this function only works for sequential model
    input_shape = (None, None, model.layers[0].input_shape[-1])
    img_input = Input(shape=input_shape)
    x = img_input
    counter = 1
    mapping = {}
    for layer in model.layers:
        if isinstance(layer, (InputLayer, Flatten)):
            output_shape = layer.input_shape
            continue

        elif isinstance(layer, Conv2D):
            x = Conv2D(layer.filters, layer.kernel_size, strides=layer.strides, padding=layer.padding,
                       data_format=layer.data_format, dilation_rate=layer.dilation_rate, activation=layer.activation,
                       use_bias=layer.use_bias, kernel_initializer=layer.kernel_initializer,
                       bias_initializer=layer.bias_initializer, kernel_regularizer=layer.kernel_regularizer,
                       bias_regularizer=layer.bias_regularizer,
                       activity_regularizer=layer.activity_regularizer, kernel_constraint=layer.kernel_constraint,
                       bias_constraint=layer.bias_constraint)(x)

        elif isinstance(layer, MaxPooling2D):
            x = MaxPooling2D(layer.pool_size, strides=layer.strides, padding=layer.padding,
                             data_format=layer.data_format)(x)

        elif isinstance(layer, Dropout):
            x = Dropout(layer.rate, noise_shape=layer.noise_shape, seed=layer.seed)(x)

        elif isinstance(layer, Dense):
            x = Conv2D(layer.output_shape[1], output_shape[1:3], activation=layer.activation, padding='same')(x)
            output_shape = (None, 1, 1, None)

        else:
            raise Exception("not supported layer:", layer.name)

        mapping[layer.name] = counter
        counter += 1

    new_model = Model(img_input, x)

    # transfer weights
    for layer in model.layers:
        if not mapping.has_key(layer.name):
            continue

        weights = layer.get_weights()
        if not weights:
            continue

        idx = mapping[layer.name]
        weights[0] = weights[0].reshape(new_model.layers[idx].get_weights()[0].shape)
        new_model.layers[idx].set_weights(weights)

    if model.__dict__.has_key('optimizer'):
        new_model.compile(model.optimizer, model.loss, metrics=model.metrics, loss_weights=model.loss_weights,
                          sample_weight_mode=model.sample_weight_mode)

    model = new_model
    return model

def decapitate(model):
    img_input = model.layers[0].input
    # decapitate the final layer (softmax activation)
    x = model.layers[-2].output
    model = Model(img_input, x)
    return model

def upsample(model):
    img_input = model.layers[0].input
    x = model.layers[-1].output
    # add 21 classes
    x = Conv2D(21, (1, 1), activation='softmax')(x)
    # upsampling
    x = Conv2DTranspose(21, (1, 1), strides=32, padding='same')(x)
    # create new model
    model = Model(img_input, x)
    return model


def convert_to_segmentation(img, cmap):
    result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            result[row,col] = cmap[np.argmax(img[row,col])]
    return result

def convert_to_groundtruth(img, index):
    result = np.zeros((img.shape[0], img.shape[1], 21))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            x = tuple(img[row,col])
            if index.has_key(x):
                result[row,col,index[x]] = 1

    return result

def test_upsampling(model):
    # just upsample as is
    img_input = model.layers[0].input
    x = model.layers[-1].output
    x = Conv2DTranspose(21, (64, 64), strides=32, padding='same')(x)
    model = Model(img_input, x)

    # draw elephant probability map (label idx = 386)
    img_path = 'elephant.jpg'
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    preds = np.squeeze(preds, axis=0)
    preds = preds[:,:,0]
    plt.imshow(preds, cmap='jet')
    plt.show()

if __name__ == '__main__':
    model = get_trained_model()
    model = convert_to_FCN(model)
    test_upsampling(model)
    #test_model(model)
    model = decapitate(model)
    model = upsample(model)

    cmap = color_map()
    index = {}
    for n in range(21):
        index[tuple(cmap[n])] = n

    image_dir = '/Users/ykang7/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages/'
    ground_truth_dir = '/Users/ykang7/.keras/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass/'
    filename = '/Users/ykang7/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    file = open(filename, 'r')
    context = file.readlines()
    for _name in context:
        name = image_dir + _name[:-1] + '.jpg'
        img = image.load_img(name)
        fig = plt.figure()
        a = fig.add_subplot(1,2,1)
        plt.imshow(img)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        f = fig.add_subplot(1,2,2)
        img = model.predict(np.expand_dims(img, axis=0))[0]
        gimg = cv2.imread(ground_truth_dir + _name[:-1] + '.png')
        gimg = cv2.cvtColor(gimg, cv2.COLOR_BGR2RGB)
        gimg = convert_to_groundtruth(gimg, index)
        img = convert_to_segmentation(img, cmap)
        plt.imshow(img)
        plt.show()

    # model = mnist_cnn()
    # model = train_mnist(model)
    # model = convert_to_FCN(model)
    # test(model)
