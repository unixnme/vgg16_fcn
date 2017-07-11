import vgg16_cnn
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from numpy.random import randint


def create_model():
    # ** reference: https://github.com/aurora95/Keras-FCN/blob/master/utils/transfer_FCN.py
    input_shape = (None, None, vgg16_cnn.num_channels)
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Dense --> Convolutional
    x = Conv2D(4096, (7, 7), activation='relu', name='fc1')(x)
    x = Conv2D(4096, (1, 1), activation='relu', name='fc2')(x)
    x = Conv2D(vgg16_cnn.num_classes, (1, 1), activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='vgg16')
    return model


def transfer_weights(model):
    model_cnn = vgg16_cnn.get_trained_model()

    index = {}
    for layer in model.layers:
        if layer.name:
            index[layer.name] = layer

    for layer in model_cnn.layers:
        weights = layer.get_weights()
        if layer.name == 'fc1' or layer.name == 'fc2' or layer.name == 'predictions':
            weights[0] = np.reshape(weights[0], index[layer.name].get_weights()[0].shape)
        if index.has_key(layer.name):
            index[layer.name].set_weights(weights)

    return model


def get_trained_model():
    return transfer_weights(create_model())


def test_model(model):
    # ** reference: https://github.com/fchollet/deep-learning-models
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(vgg16_cnn.img_rows, vgg16_cnn.img_cols))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.squeeze(preds).reshape(1, -1)
    print('Predicted:', decode_predictions(preds))

    # test variable size input
    img = image.load_img(img_path, target_size=(448, 448))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.squeeze(preds, axis=0)
    preds = np.argmax(preds, axis=2)
    print preds


if __name__ == '__main__':
    model = get_trained_model()
    test_model(model)