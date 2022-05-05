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

EPOCHS = 100
VAL_STEPS = 20

starttime = time.time()
train_data = ImageDataGenerator(rescale=1./255)
val_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    TRAIN_DIR,
    target_size = (128, 128),
    batch_size = 20,
    class_mode = 'binary'
)
val_generator = val_data.flow_from_directory(
    VAL_DIR,
    target_size = (128, 128),
    batch_size = 20,
    class_mode = 'binary'
)
test_generator = test_data.flow_from_directory(
    TEST_DIR,
    target_size = (128, 128),
    batch_size = 20,
    class_mode = 'binary'
)

## model
input_shape = (128, 128, 3)

model = models.Sequential()
conv_base = VGG16(
    weights = 'imagenet',
    include_top = False,
    input_shape = input_shape
)
conv_base.trainable = False
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    loss='binary_crossentropy', metrics=['accuracy']
)

history = model.fit(
    train_generator, epochs = EPOCHS,
    validation_data = val_generator, validation_steps = VAL_STEPS
)

## save
model.save("chest_x_ray_pretrained_model_01.h5")

## visualization result
train_loss = history.history["loss"]
train_acc = history.history["accuracy"]
test_loss, test_acc = model.evaluate(test_generator)
print("train_loss : ", train_loss[-1], "train_acc : ", train_acc[-1])
print("test_loss : ", test_loss, "test_acc : ", test_acc)
print("time : ", time.time() - starttime)

def plot_acc(h, title="accuracy"):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history ['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

plot_loss(history)
plt.savefig('loss.png')
plt.clf()
plot_acc(history)
plt.savefig('accuracy.png')