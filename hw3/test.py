from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import time

model = models.load_model("C:/Users/cjswl/python__/ann/hw3/backup-model/chest_x_ray_pretrained_model_01.h5")
model.summary()
conv_model = model.layers[0]
conv_model.summary()

for layer in conv_model.layers:
    if layer.name.startswith('block5'):
        layer.trainable = True
conv_model.summary()