## runall
#  Q1 ~ Q3 + QE1

## import
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
import matplotlib.pyplot as plt
import time

## common
TRAIN_DIR = "C:/Users/cjswl/python__/ann-data/chest_xray/chest_xray/train"
VAL_DIR = "C:/Users/cjswl/python__/ann-data/chest_xray/chest_xray/val"
TEST_DIR = "C:/Users/cjswl/python__/ann-data/chest_xray/chest_xray/test"
FIG_DIR = "figures_QE3/"
TXT_DIR = "txtfiles_QE3/"

def run(Q):
    ## Q1
    if Q == 1:
        input_shape = (128, 128, 3)
        batch_size = 20
        epochs1 = 100
        epochs2 = 50
    ## Q2
    elif Q == 2:
        input_shape = (128, 128, 3)
        batch_size = 20
        epochs1 = 100
        epochs2 = 100
    ## Q3 - 256
    elif Q == 3:
        input_shape = (256, 256, 3)
        batch_size = 10
        epochs1 = 100
        epochs2 = 100
    ## Q3 - 512
    elif Q == 4:
        input_shape = (512, 512, 3)
        batch_size = 10
        epochs1 = 100
        epochs2 = 100
    ## QE1
    elif Q == 5:
        input_shape = (None, None, 3)
        batch_size = 10
        epochs1 = 100
        epochs2 = 100
    else:
        print("ERROR")
        exit()
    
    starttime = time.time()
    train, val, test = generator(input_shape[:2], batch_size)
    model = get_model(Q, input_shape)
    history_before = model.fit_generator(
        train, epochs = epochs1,
        steps_per_epoch = 100,
        validation_data = val
    )
    model_name = "new_models_QE3/chest_x_ray_Q" + str(Q) + "_QE3.h5"
    model.save(model_name)

    train_loss = history_before.history["loss"][-1]
    train_acc = history_before.history["accuracy"][-1]
    test_loss, test_acc = model.evaluate_generator(test)
    plot_result(history_before, Q, "accuracy", "before")
    plot_result(history_before, Q, "loss", "before")
    text_name = TXT_DIR + "Q" + str(Q) + "before_QE3.txt"
    with open(text_name, 'w') as f:
        f.write("train_loss : " + str(train_loss) + "\n")
        f.write("train_acc : " + str(train_acc) + "\n")
        f.write("test_loss : " + str(test_loss) + "\n")
        f.write("test_acc : " + str(test_acc) + "\n")
        f.write("time : " + str(time.time() - starttime) + "\n")
    
    model_2step = load_model(model_name)
    conv_base = model_2step.layers[0]

    for layer in conv_base.layers:
        if layer.name.startswith('block5'):
            layer.trainable = True
    model_2step.compile(
        optimizer = optimizers.RMSprop(learning_rate=1e-5),
        loss = "binary_crossentropy", metrics = ["accuracy"]
    )
    train_2step, val_2step, test_2step = generator(input_shape[:2], batch_size)

    history_after = model_2step.fit_generator(
        train_2step, epochs = epochs2,
        steps_per_epoch = 100,
        validation_data = val_2step
    )

    model_2step.save("new_models_QE3/chest_x_ray_Q" + str(Q) + "_after_QE3.h5")

    train_loss = history_after.history["loss"][-1]
    train_acc = history_after.history["accuracy"][-1]
    test_loss, test_acc = model_2step.evaluate_generator(test_2step)
    plot_result(history_after, Q, "accuracy", "after")
    plot_result(history_after, Q, "loss", "after")
    text_name = TXT_DIR + "Q" + str(Q) + "after_QE3.txt"
    with open(text_name, 'w') as f:
        f.write("train_loss : " + str(train_loss) + "\n")
        f.write("train_acc : " + str(train_acc) + "\n")
        f.write("test_loss : " + str(test_loss) + "\n")
        f.write("test_acc : " + str(test_acc) + "\n")
        f.write("time : " + str(time.time() - starttime) + "\n")


def get_model(Q, input_shape):
    model = models.Sequential()
    conv_base = InceptionV3(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape
    )
    conv_base.trainable = False
    model.add(conv_base)
    if Q == 1:
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dense(128, activation='relu'))
    else:
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
        optimizer=optimizers.RMSprop(learning_rate=1e-3),
        loss='binary_crossentropy', metrics=['accuracy']
    )
    return model

def data_generator(dir, input_shape, batch_size, data):
    if input_shape[0] == None:
        data_gen = data.flow_from_directory(
        dir,
        batch_size = batch_size,
        class_mode = 'binary'
        )
    else:
        data_gen = data.flow_from_directory(
            dir,
            target_size = input_shape,
            batch_size = batch_size,
            class_mode = 'binary'
        )
    return data_gen

def generator(input_shape, batch_size):
    train_data = ImageDataGenerator(rescale=1./255)
    val_data = ImageDataGenerator(rescale=1./255)
    test_data = ImageDataGenerator(rescale=1./255)
    train_generator = data_generator(TRAIN_DIR, input_shape, batch_size, train_data)
    val_generator = data_generator(VAL_DIR, input_shape, batch_size, val_data)
    test_generator = data_generator(TEST_DIR, input_shape, batch_size, test_data)
    return train_generator, val_generator, test_generator

def plot_result(h, Q, title, T):
    if title == "accuracy":
        plt.plot(h.history['accuracy'])
        plt.plot(h.history['val_accuracy'])
    elif title == "loss":
        plt.plot(h.history['loss'])
        plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)
    plt.savefig(FIG_DIR + 'Q' + str(Q) + title + T + '_QE3.png')
    plt.clf()

import datetime

if __name__ == "__main__":
    print(datetime.datetime.now())
    for q in range(1,6):
        print(q)
        run(q)
    print(datetime.datetime.now())