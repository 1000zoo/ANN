from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import time
model = VGG16(
    weights = 'imagenet',
    include_top = False
)
model.summary()
conv_model = model.layers[0]
conv_model.summary()

for layer in conv_model.layers:
    if layer.name.startswith('block5'):
        layer.trainable = True
conv_model.summary()