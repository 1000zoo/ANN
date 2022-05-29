## main model
"""
hyper-parameter
- units = 52
- Activation = sigmoid
- Optimizer = Adam(lr = 0.001)
- Loss function = mse
- Batch size = 20
- Epoch = 1000
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import TimeSeriesSplit

PATH = "/Users/1000zoo/Documents/prog/data_files/hw4_covid/practice_data.txt"
EPOCHS = 1000

def generator(data, time_steps=10, batch_size=20):
    batch = len(data) - time_steps
    b = 0
    while(True):
        input = []
        target = []
        if b + batch_size >= batch:
            b = 0

        for i in range(batch_size):
            try:
                input.append(data[i+b:time_steps+i+b])
                target.append(data[time_steps+i+b])
            except IndexError:
                print("err")
                exit()
        b += batch_size
        input = np.array(input)
        target = np.array(target)
        yield input, target

def normalize(data):
    norm_data = copy.deepcopy(data)
    for i, d in enumerate(data):
        max_value = np.max(d)
        if max_value <= 0:
            continue
        norm_data[i] /= max_value
    return norm_data

def build_model():
    model = models.Sequential()
    model.add(layers.SimpleRNN(52, activation="sigmoid"))
    model.add(layers.Dense(52))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    # model.summary()
    return model

def tscv_fit(model, train_data, batch_size=20, epochs=EPOCHS):
    tscv = TimeSeriesSplit(n_splits=4)
    cnt_fold = 0
    loss_history = []
    for x_index, y_index in tscv.split(train_data):
        cnt_fold += 1
        print("="*30)
        print("="*30)
        print(cnt_fold, "-fold")
        print("="*30)
        print("="*30)
        history = model.fit_generator(
            generator(train_data[x_index]), steps_per_epoch = len(x_index) // batch_size,
            validation_data = generator(train_data[y_index]), validation_steps = len(y_index) // batch_size,
            epochs = epochs
        )
        loss_history.append(history.history)
    
    final_history = model.fit_generator(
        generator(train_data), steps_per_epoch = len(train_data) // batch_size,
        epochs = epochs
    )
    loss_history.append(final_history.history)
    return loss_history

def main():
    with open(PATH, "rb") as f:
        odata = pickle.load(f)
    data = normalize(odata)
    train_data = data[:350]
    test_data = data[350:]

    model = build_model()
    loss_history = tscv_fit(model, train_data)

    model.save("covid_rnn_test.h5")
    for fold, history in enumerate(loss_history):
        if fold == len(loss_history) - 1:
            fold_name = "all training set"
        else:
            fold_name = str(fold) + " training set"
        loss_figure(history)
        plt.title(fold_name + "-loss graph")
        plt.savefig("figures/loss" + fold_name + ".jpg")
        plt.clf()
        try:
            train_mean = np.mean(history["loss"])
            val_mean = np.mean(history["val_loss"])
            print(fold_name, "- fold train mean :", train_mean)
            print(fold_name, "- fold val mean :", val_mean)
            print(fold_name, "- optimal epochs :", np.argmin(history["val_loss"]))
        except KeyError:
            train_mean = np.mean(history["loss"])
            print(fold_name, "- fold train mean :", train_mean)

def loss_figure(h):
    plt.plot(h["loss"])
    try:
        plt.plot(h["val_loss"])
    except KeyError as err:
        pass
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"], loc=0)


if __name__ == "__main__":
    main()
    