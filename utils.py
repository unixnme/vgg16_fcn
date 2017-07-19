import keras
from keras.models import Model
from keras.layers import Flatten, Input, Conv2D, InputLayer, MaxPooling2D, Dropout, Dense
from mnist_cnn import mnist_cnn, train_mnist
from vgg16_cnn import get_trained_model
from vgg16_fcn import test_model
from mnist_fcn import test, train


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
            x = Conv2D(layer.output_shape[1], output_shape[1:3], activation=layer.activation)(x)
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


if __name__ == '__main__':
    # model = get_trained_model()
    # model = convert_to_FCN(model)
    # test_model(model)

    model = mnist_cnn()
    model = train_mnist(model)
    model = convert_to_FCN(model)
    test(model)
