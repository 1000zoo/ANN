from tensorflow.keras import layers, models

class AE(models.Model):
    def __init__(self, x_nodes=784, z_dim=36):
        x_shape = (x_nodes, )
        x = layers.Input(shape=x_shape)
        z = layers.Dense(z_dim, activation='relu')(x)
        y = layers.Dense(x_nodes, activation='sigmoid')(z)

        super().__init__(x, y)
        self.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        self.x = x
        self.z = z
        self.z_dim = z_dim

    def Encoder(self):
        return models.Model(self.x, self.z)

    def Decoder(self):
        z_shape = (self.z_dim, )
        z = layers.Input(shape=z_shape)
        y_layer = self.layers[-1]
        y = y_layer(z)
        return models.Model(z, y)

from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

def plot(history, plot_type, q):
    h = history.history
    path = "./result/"
    val_type = "val_" + plot_type
    plt.plot(h[plot_type])
    plt.plot(h[val_type])
    plt.title(plot_type)
    plt.ylabel(plot_type)
    plt.xlabel("Epoch")
    plt.legend(['Training', 'Validation'], loc=0)
    plt.savefig(path + plot_type + '_' + q + '.jpg')
    plt.clf()

def data_load():
    (x_train, _), (x_test, _) = cifar10.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return (x_train, x_test)

def show_ae(autoencoder, x_test, qnum):
    path = "./result/show_ae_" + qnum + ".jpg"

    encoder = autoencoder.Encoder()
    decoder = autoencoder.Decoder()
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10
    plt.figure(figsize=(20, 6))

    for i in range(n):
        ax = plt.subplot(3, n, i+1)
        plt.imshow(x_test[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i+1+n)
        plt.stem(encoded_imgs[i].reshape(-1), use_line_collection=True)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(path)
    plt.clf()

def main():
    """
    q1. Auto-Encoder
        data_load() : mnist => cifar10          V
        main() : x_nodes = 32 * 32 * 3          V
        AE.__init__() : lossfunction = "mse"    V
        show_ae() : (28, 28) => (32, 32, 3)     V

        a. show_ae() : hidden neurons = 36
        b. show_ae() : hidden neurons = 360
        c. show_ae() : hidden neurons = 1080
    """

    alp = ["Q1a", "Q1b", "Q1c"]
    alpnum = {
        "Q1a" : 36,
        "Q1b" : 360,
        "Q1c" : 1080
    }

    x_nodes = 32 * 32 * 3

    for q in alp:
        qnum = q
        z_dim = alpnum[q]

        (x_train, x_test) = data_load()
        autoencoder = AE(x_nodes, z_dim)

        history = autoencoder.fit(
            x_train, x_train,
            epochs = 20,
            batch_size = 256,
            shuffle = True,
            validation_data = (x_test, x_test)
        )

        show_ae(autoencoder, x_test, qnum)
        plot(history, "loss", qnum)
        plot(history, "accuracy", qnum)

if __name__ == "__main__":
    main()