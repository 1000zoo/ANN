import os
import numpy as np

PATH = "/Users/1000zoo/Documents/prog/data_files/jena_climate/jena_climate_2009_2016.csv"

class Data():
    def __init__(self, fname, ratio):
        f = open(fname)
        data = f.read()
        f.close()

        lines = data.split("\n")
        header = lines[0].split(".")
        lines = lines[1:]
        values = [line.split(",")[1:] for line in lines]
        self.float_data = np.array(values).astype("float32")
        self.data_length = self.float_data.shape[-1]

        self.ratio = ratio
        self.train_set_length = int(self.float_data.shape[0] * ratio[0])
        self.val_set_length = int(self.float_data.shape[0] * ratio[1])

        Data.normalize(self)

    def normalize(self):
        import copy
        self.data = copy.deepcopy(self.float_data)
        mean = self.data[:self.train_set_length].mean(axis=0)
        self.data -= mean
        std = self.data[:self.train_set_length].std(axis=0)
        self.data /= std
        self.mean = mean
        self.std = std

    def get_generators(self, lookback, delay, batch_size = 128, step = 6):
        self.train_gen = Data.generator(
            self, lookback=lookback, delay=delay,
            min_index=0, max_index=self.train_set_length,
            shuffle=True, step=step, batch_size=batch_size
            )
        self.val_gen = Data.generator(
            self, lookback=lookback, delay=delay,
            min_index=self.train_set_length, max_index=self.train_set_length+self.val_set_length,
            shuffle=False, step=step, batch_size=batch_size
            )
        self.test_gen = Data.generator(
            self, lookback=lookback, delay=delay,
            min_index=self.train_set_length+self.val_set_length, max_index=None,
            shuffle=True, step=step, batch_size=batch_size
            )
        self.train_steps = (self.train_set_length - lookback) // batch_size
        self.val_steps = (self.val_set_length - lookback) // batch_size
        self.test_steps = (len(self.data) - self.val_set_length - self.train_set_length - lookback) // batch_size
        self.lookback = lookback
        self.batch_size = batch_size
        self.step = step

    def generator(self, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
        if max_index is None:
            max_index = len(self.data) - delay - 1
        i = min_index + lookback

        while True:
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size
                )
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros((len(rows), lookback // step, self.data_length))
            targets = np.zeros((len(rows),))

            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = self.data[indices]
                targets[j] = self.data[rows[j] + delay - 1][1]
            print(samples.shape, targets.shape)
            exit()
            yield samples, targets



def main_ANN(ratio = [0.5, 0.25, 0.25], lookback = 2880, step = 6, delay = 144, batch_size = 128, epochs = 20):
    fname = "/Users/1000zoo/Documents/prog/ANN/data_files/jena_climate/jena_climate_2009_2016.csv"
    dataset = Data(fname, ratio)
    dataset.get_generators(lookback=lookback, delay=delay, batch_size=batch_size, step=step)

    from tensorflow.keras import layers, models

    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(lookback//step, dataset.data.shape[-1])))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="RMSprop", loss="mae")

    print("="*30)
    print("="*30)
    print("="*30)
    print("ANN")
    model.summary()

    history = model.fit_generator(
        dataset.train_gen, steps_per_epoch=dataset.train_steps, epochs=epochs,
        validation_data=dataset.val_gen, validation_steps=dataset.val_steps
    )
    train_loss = model.evaluate_generator(dataset.train_gen, steps=dataset.val_steps)
    test_loss = model.evaluate(dataset.test_gen, steps=dataset.test_steps)
    print("train_loss:", train_loss)
    print("test_loss:", test_loss)
    return history

def main_RNN(ratio = [0.5, 0.25, 0.25], lookback = 2880, step = 6, delay = 144, batch_size = 128, epochs = 20):
    fname = PATH
    dataset = Data(fname, ratio)
    dataset.get_generators(lookback=lookback, delay=delay, batch_size=batch_size, step=step)

    from tensorflow.keras import layers, models

    model = models.Sequential()
    model.add(layers.SimpleRNN(32, activation="relu", input_shape=(lookback//step, dataset.data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer="RMSprop", loss="mae")

    print("="*30)
    print("="*30)
    print("="*30)
    print("RNN")
    model.summary()

    train = dataset.train_gen

    history = model.fit_generator(
        dataset.train_gen, steps_per_epoch=dataset.train_steps, epochs=epochs,
        validation_data=dataset.val_gen, validation_steps=dataset.val_steps
    )
    train_loss = model.evaluate_generator(dataset.train_gen, steps=dataset.val_steps)
    test_loss = model.evaluate_generator(dataset.test_gen, steps=dataset.val_steps)
    print("train_loss:", train_loss)
    print("test_loss:", test_loss)
    return history


def main_LSTM(ratio = [0.5, 0.25, 0.25], lookback = 2880, step = 6, delay = 144, batch_size = 128, epochs = 20):
    fname = "/Users/1000zoo/Documents/prog/ANN/data_files/jena_climate/jena_climate_2009_2016.csv"
    dataset = Data(fname, ratio)
    dataset.get_generators(lookback=lookback, delay=delay, batch_size=batch_size, step=step)

    from tensorflow.keras import layers, models

    model = models.Sequential()
    model.add(layers.LSTM(32,input_shape=(lookback//step, dataset.data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer="RMSprop", loss="mae")

    print("="*30)
    print("="*30)
    print("="*30)
    print("LSTM")
    model.summary()

    # history = model.fit_generator(
    #     dataset.train_gen, steps_per_epoch=500, epochs=epochs,
    #     validation_data=dataset.val_gen, validation_steps=dataset.val_steps
    # )
    # train_loss = model.evaluate_generator(dataset.train_gen, steps=dataset.val_steps)
    # test_loss = model.evaluate_generator(dataset.test_gen, steps=dataset.val_steps)
    # print("train_loss:", train_loss)
    # print("test_loss:", test_loss)
    # return history, model


if __name__ == "__main__":
    # main_ANN()
    main_RNN()
    # main_LSTM()

    import matplotlib.pyplot as plt

    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title("ANN LOSS")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.legend(["Training", "Validation"], loc=0)
    plt.show()

    m.summary()