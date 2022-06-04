from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

import matplotlib.pyplot as plt

class Unet(models.Model):
    def conv(x, n_f, mp_flag=True):
        if mp_flag:
            x = layers.MaxPooling2D((2,2), padding="same")(x)
        x = layers.Conv2D(n_f, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("tanh")(x)
        x = layers.Dropout(0.05)(x)
        x = layers.Conv2D(n_f, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("tanh")(x)
        return x

    def deconv_unet(x, e, n_f):
        x = layers.UpSampling2D((2,2))(x)
        x = layers.Concatenate(axis=3)([x, e]) ##*****
        x = layers.Conv2D(n_f, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("tanh")(x)
        x = layers.Conv2D(n_f, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("tanh")(x)
        return x

    def __init__(self, org_shape):
        original = layers.Input(shape=org_shape)

        c1 = Unet.conv(original, 16, mp_flag=False)
        c2 = Unet.conv(c1, 32)

        encoded = Unet.conv(c2, 64)

        x = Unet.deconv_unet(encoded, c2, 32)
        y = Unet.deconv_unet(x, c1, 16)

        decoded = layers.Conv2D(3, (3,3), activation="sigmoid", padding="same")(y)

        super().__init__(original, decoded)
        self.compile(optimizer="adadelta", loss="mse")

class Data():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32")/255
        x_test = x_test.astype("float32")/255

        self.x_train_in = x_train
        self.x_test_in = x_test
        self.x_train_out = x_train
        self.x_test_out = x_test

        img_rows, img_cols, n_ch = self.x_train_in.shape[1:]
        self.input_shape = (img_rows, img_cols, n_ch)

def show_images(data, unet):
    x_test_in = data.x_test_in
    x_test_out = data.x_test_out
    decoded_imgs = unet.predict(x_test_in)

    n = 10
    plt.figure(figsize=(20, 6))

    for i in range(n):
        ax = plt.subplot(3, n, i+1)
        plt.imshow(x_test_in[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i+1+n)
        plt.imshow(decoded_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i+1+n*2)
        plt.imshow(x_test_out[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("show_images_Q3.jpg")
    plt.clf()

def plot_loss(history):
    h = history.history
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["loss", "val_loss"])
    plt.title("Loss")
    plt.savefig("loss_graph_Q3.jpg")
    plt.clf()


def main(epochs=10, batch_size=512, fig=True):
    data = Data()
    unet = Unet(data.input_shape)
    unet.summary()

    history = unet.fit(
        data.x_train_in, data.x_train_out,
        epohcs = epochs, batch_size = batch_size,
        shuffle = True,
        validation_data = (data.x_test_in, data.x_test_out)
    )

    if fig:
        plot_loss(history)
        show_images(data, unet)

if __name__ == "__main__":
    # import argparse
    # from distutils import util

    # parser = argparse.ArgumentParser(description="Unet for cifar-10")
    # parser.add_argument(
    #     "--epochs", type = int, default = 100,
    #     help = "training epohcs (default: 100)"
    # )
    # parser.add_argument(
    #     "--batch_size", type = int, default = 128,
    #     help = "batch_size (default: 128)"
    # )
    # parser.add_argument(
    #     "--fig", type = lambda x: bool(util.strtobool(x)),
    #     default = True, help = "flag to show figures (default: True)"
    #     )

    # args = parser.parse_args()
    # print("Aargs:", args)
    main()


