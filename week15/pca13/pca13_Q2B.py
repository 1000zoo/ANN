from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def Conv2D(filters, kernel_size, padding="same", activation="relu"):
    return layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)

class SCAE(models.Model):
    def __init__(self, org_shape=(3,32,32)):
        original = layers.Input(shape=org_shape)
        x = Conv2D(82, (3,3))(original)
        x = layers.MaxPooling2D((2,2), padding="same")(x)
        x = Conv2D(64, (3,3))(x)
        x = layers.MaxPooling2D((2,2), padding="same")(x)
        z = Conv2D(16, (7,7))(x)

        y = Conv2D(16, (3,3))(z)
        y = layers.UpSampling2D((2,2))(y)
        y = Conv2D(64, (3,3))(y)
        y = layers.UpSampling2D((2,2))(y)
        y = Conv2D(82, (3,3))(y)
        decoded = Conv2D(3, (3,3), activation="sigmoid")(y)

        super().__init__(original, decoded)
        self.compile(optimizer='adam', loss="mse", metrics=['accuracy'])

        self.x = x
        self.z = z
        self.original = original
        self.summary()

    def Encoder(self):
        return models.Model(self.original, self.z)

    def Decoder(self):
        z_shape = (8,8,16)
        z = layers.Input(shape=z_shape)
        h = self.layers[-6](z)
        h = self.layers[-5](h)
        h = self.layers[-4](h)
        h = self.layers[-3](h)
        h = self.layers[-2](h)
        h = self.layers[-1](h)
        return models.Model(z, h)

from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

def plot(history, plot_type, q="2B"):
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
    [_, W, H, _] = x_train.shape
    # x_train = x_train.reshape((-1, W, H, 1))
    # x_test = x_test.reshape((-1, W, H, 1))
    return (x_train, x_test)

def show_ae(autoencoder, x_test, qnum="2B"):
    path = "./result/show_ae_" + qnum + ".jpg"

    encoder = autoencoder.Encoder()
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    n = 10
    plt.figure(figsize=(20, 6))

    for i in range(n):
        ax = plt.subplot(4, n, i+1)
        plt.imshow(x_test[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i+1+n)
        plt.stem(encoded_imgs[i].reshape(-1), use_line_collection=True)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i+1+n*2)
        plt.imshow(encoded_imgs[i].reshape(8, 8))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i+1+n*3)
        plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(path)
    plt.clf()

def main(epochs=20, batch_size=128):
    input_shape=[32, 32, 3]
    (x_train, x_test) = data_load()
    autoencoder = SCAE(input_shape)

    history = autoencoder.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, x_test)
    )

    plot(history, "loss")
    plot(history, "accuracy")
    show_ae(autoencoder, x_test)



if __name__ == "__main__":
    main()