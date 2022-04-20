TRAIN_PATH = "/Users/1000zoo/Documents/prog/ANN/data_files/train"
VAL_PATH = "/Users/1000zoo/Documents/prog/ANN/data_files/validation"
TEST_PATH = "/Users/1000zoo/Documents/prog/ANN/data_files/test"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size = (150, 150),
    batch_size = 20,
    class_mode = 'binary'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = train_datagen.flow_from_directory(
    VAL_PATH,
    target_size = (150, 150),
    batch_size = 20,
    class_mode = 'binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size = (150, 150),
    batch_size = 20,
    class_mode = 'binary'
)

from tensorflow.keras import models, layers, optimizers

input_shape = [150, 150, 3]

"""
CONV2D(128)
MaxPooling
Flatten
Dense(128)
Dense(softmax)
"""
def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                    input_shape = input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4),
                    loss='binary_crossentropy', metrics=['accuracy'])
    return model

import time
start = time.time()
num_epochs = 30
model = build_model()
history = model.fit_generator(
    train_generator,
    epochs = num_epochs,
    steps_per_epoch = 100,
    validation_data = validation_generator,
    validation_steps = 50 
)

model.save("cats_and_dogs_small_1.h5")

train_loss, train_acc = model.evaluate_generator(train_generator)
train_loss, test_acc = model.evaluate_generator(test_generator)
print("train_acc : ", train_acc)
print("test_acc : ", test_acc)
print("time (in sec) : ", time.time() - start)

import matplotlib.pyplot as plt

plt.title("Loss")
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])
plt.savefig("ex2loss.png")
plt.clf()

plt.title("Accuracy")
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["acc", "val_acc"])
plt.savefig("ex2_acc.png")
