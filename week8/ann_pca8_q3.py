from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# data loading
(X_train, Y_train), (X_test, Y_test) =  cifar10.load_data()

# data preprocessing
X_train = X_train / 255.0
X_test = X_test / 255.0
num_classes=10
Y_train_ = to_categorical(Y_train, num_classes)
Y_test_ = to_categorical(Y_test, num_classes)

# model definition
L,W,H,C = X_train.shape
input_shape = [W, H, C] # as a shape of image
def build_model():
    model = models.Sequential()
    # first convolutional layer
    model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu',
                           input_shape=input_shape))
    # second convolutional layer
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    # max-pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))                 # add dropout
    # Fully connected MLP
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))                  # add dropout
    model.add(layers.Dense(num_classes, activation='softmax'))
    # compile   # add learning rate
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# main loop without cross-validation
import time
starttime=time.time()
num_epochs = 50
model = build_model()
# set batch_size 100 to 32
history = model.fit(X_train, Y_train_, validation_split=0.2,
                    epochs=num_epochs, batch_size=32, verbose=1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=1)])
train_loss, train_acc = model.evaluate(X_train, Y_train_)
test_loss, test_acc = model.evaluate(X_test, Y_test_)
print('train_acc:', train_acc)
print('test_acc:', test_acc)
print("elapsed time (in sec): ", time.time()-starttime)

# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history ['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

def plot_loss(h, title="loss"):
    plt.plot(h.history ['loss'])
    plt.plot(h.history ['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

plot_loss(history)
plt.savefig('result/chapter5-1_cifar10.loss.png')
plt.clf()
plot_acc(history)
plt.savefig('result/chapter5-1_cifar10.accuracy.png')

