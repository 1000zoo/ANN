from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3
    
# set image generators
train_dir='/Users/1000zoo/Documents/prog/ANN/data_files/dogs_and_cats/train'
test_dir='/Users/1000zoo/Documents/prog/ANN/data_files/dogs_and_cats/test'
validation_dir='/Users/1000zoo/Documents/prog/ANN/data_files/dogs_and_cats/validation'

train_datagen = ImageDataGenerator(rescale=1./255,
                    rotation_range=20, shear_range=0.1,
                    width_shift_range=0.1, height_shift_range=0.1,
                    zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# model definition
input_shape = [150, 150, 3] # as a shape of image

def build_model():
    model=models.Sequential()
    conv_base = InceptionV3(weights='imagenet',
                      include_top=False,
                      input_shape=input_shape)
    conv_base.trainable=False
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# main loop without cross-validation
import time
starttime=time.time()
num_epochs = 30
model = build_model()
history = model.fit_generator(train_generator,
                    epochs=num_epochs, steps_per_epoch=100,
                    validation_data=validation_generator, validation_steps=50)

# saving the model
model.save('cats_and_dogs_small_pretrained.h5')

# evaluation
train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate(test_generator)
print('train_acc:', train_acc)
print('test_acc:', test_acc)
print("elapsed time (in sec): ", time.time()-starttime)

# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history ['val_accuracy'])
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
plt.savefig('Q2.loss.png')
plt.clf()
plot_acc(history)
plt.savefig('Q2.accuracy.png')

