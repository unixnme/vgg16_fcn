from reference import upsample_filt, bilinear_upsample_weights
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, AveragePooling2D
from keras.models import Model
from keras.initializers import Constant
import matplotlib.pyplot as plt
import numpy as np

# get random input
np.random.seed(0)
nx = 1024; ny = 1024
u = np.linspace(0, 1, nx)
v = np.linspace(0, 1, ny)
xx, yy = np.meshgrid(u, v, sparse=False)
x_train = xx**2 + yy**2
x_train = np.expand_dims(x_train, axis=0)
x_train = np.stack([x_train, np.zeros(x_train.shape)], axis=-1)


img_input = Input(shape=(None, None, x_train.shape[-1]))
x = Activation(activation='softmax')(img_input)

# as is
model1 = Model(inputs=img_input, outputs=x)
y = model1.predict(x_train)
plt.figure()
plt.imshow(y[0,:,:,0], cmap='jet', vmin=0, vmax=1)
plt.colorbar()
plt.show()

# downsample
x = model1.output
x = AveragePooling2D((2,2))(x)
model2 = Model(inputs=model1.input, outputs=x)
y = model2.predict(x_train)
plt.figure()
plt.imshow(y[0,:,:,0], cmap='jet', vmin=0, vmax=1)
plt.colorbar()
plt.show()

# upsample
x = model2.output
x = Conv2DTranspose(2,(4,4),strides=(2,2),kernel_initializer=Constant(bilinear_upsample_weights(2, 2)), padding='same')(x)
model3 = Model(inputs=model2.input, output=x)
y = model3.predict(x_train)
plt.figure()
plt.imshow(y[0,:,:,0], cmap='jet', vmin=0, vmax=1)
plt.colorbar()
plt.show()