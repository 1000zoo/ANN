TRAIN_PATH = "/Users/1000zoo/Documents/prog/ANN/data_files/train"
VAL_PATH = "/Users/1000zoo/Documents/prog/ANN/data_files/validation"
TEST_PATH = "/Users/1000zoo/Documents/prog/ANN/data_files/test"

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers


(x_train, y_train), (x_test, y_test) = mnist.load_data()
l, w, h = x_train.shape
input_shape = [w, h, 1]
x_train = x_train.reshape(-1, w, h, 1)
x_train = x_train.astype('float')/255
x_test = x_test.reshape(-1, w, h, 1)
x_test = x_test.astype('float')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

"""
CONV2D(32)
CONV2D(64)
MaxPooling
Flatten
Dense(128)
Dense(softmax)
"""

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                    input_shape = input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy', metrics=['accuracy'])
    return model

import time
start = time.time()
num_epochs = 30
model = build_model()
history = model.fit(
    x_train, y_train,
    epochs = num_epochs,
    validation_split = 0.2,
    batch_size = 100,
    verbose = 1 
)

# model.save("cats_and_dogs_small_1.h5")

train_loss, train_acc = model.evaluate(x_train, y_train)
train_loss, test_acc = model.evaluate(x_test, y_test)
print("train_acc : ", train_acc)
print("test_acc : ", test_acc)
print("time (in sec) : ", time.time() - start)

import matplotlib.pyplot as plt

plt.title("Loss")
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])
plt.savefig("result/ex1_loss.png")
plt.clf()

plt.title("Accuracy")
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["acc", "val_acc"])
plt.savefig("result/ex1_acc.png")