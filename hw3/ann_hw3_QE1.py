## import
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import time

## constants & assign
TRAIN_DIR = "C:/Users/cjswl/python__/ann-data/chest_xray/train"
VAL_DIR = "C:/Users/cjswl/python__/ann-data/chest_xray/val"
TEST_DIR = "C:/Users/cjswl/python__/ann-data/chest_xray/test"

EPOCHS1 = 100
EPOCHS2 = 100
STEPS_PER_EPOCH = 100

starttime = time.time()
train_data = ImageDataGenerator(rescale=1./255)
val_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    TRAIN_DIR,
    batch_size = 10,
    class_mode = 'binary'
)
val_generator = val_data.flow_from_directory(
    VAL_DIR,
    batch_size = 10,
    class_mode = 'binary'
)
test_generator = test_data.flow_from_directory(
    TEST_DIR,
    batch_size = 10,
    class_mode = 'binary'
)

## model
input_shape = (None, None, 3)

model = models.Sequential()
conv_base = VGG16(
    weights = 'imagenet',
    include_top = False,
    input_shape = input_shape
)
conv_base.trainable = False
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    loss='binary_crossentropy', metrics=['accuracy']
)

history = model.fit(
    train_generator, epochs = EPOCHS1, steps_per_epoch = STEPS_PER_EPOCH,
    validation_data = val_generator
)

## save
model.save("chest_x_ray_pretrained_model_QE1.h5")

## visualization result
train_loss = history.history["loss"]
train_acc = history.history["accuracy"]
test_loss, test_acc = model.evaluate(test_generator)
print("train_loss : ", train_loss[-1], "train_acc : ", train_acc[-1])
print("test_loss : ", test_loss, "test_acc : ", test_acc)
print("time : ", time.time() - starttime)

## plot function
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training'], loc=0)

def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training'], loc=0)

plot_loss(history)
plt.savefig('QE1-loss.png')
plt.clf()
plot_acc(history)
plt.savefig('QE1-accuracy.png')
plt.clf()

## 2step start
print("="*30)
print("="*30)
print("="*30)
print("==========2step part==========")
print("="*30)
print("="*30)
print("="*30)
conv_base = model.layers[0]

## unfrozen block5
for layer in conv_base.layers:
    if layer.name.startswith("block5"):
        layer.trainable = True

model.compile(
    optimizer = optimizers.RMSprop(learning_rate=1e-5),
    loss = "binary_crossentropy", metrics = ['accuracy']
)
starttime = time.time()

history = model.fit_generator(
    train_generator, epochs = EPOCHS2, steps_per_epoch = STEPS_PER_EPOCH,
    validation_data = val_generator
)

## visualization result
train_loss = history.history["loss"]
train_acc = history.history["accuracy"]
test_loss, test_acc = model.evaluate(test_generator)
print("train_loss : ", train_loss[-1], "train_acc : ", train_acc[-1])
print("test_loss : ", test_loss, "test_acc : ", test_acc)
print("time : ", time.time() - starttime)

plot_loss(history)
plt.savefig('QE1_afterloss.png')
plt.clf()
plot_acc(history)
plt.savefig('QE1_afteraccuracy.png')

## save fine-tuning model
model.save("chest_x_ray_2step_QE1.h5")