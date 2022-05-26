from tensorflow.keras import layers, models
class AE(models.Model):
    def __init__(self, x_nodes=784, z_dim=36):
        x_shape = (x_nodes,)
        x = layers.Input(shape=x_shape)
        z = layers.Dense(z_dim, activation='relu')(x)
        y = layers.Dense(x_nodes, activation='sigmoid')(z)
    # Essential parts:
        super().__init__(x, y)
        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Optional Parts: they are for Encoder and Decoder
        self.x = x
        self.z = z
        self.z_dim = z_dim
# These Encoder & Decoder are inside the AE class!
def Encoder(self):
    return models.Model(self.x, self.z)
def Decoder(self):
    z_shape = (self.z_dim,)
    z = layers.Input(shape=z_shape)
    y_layer = self.layers[-1]
    y = y_layer(z)
    return models.Model(z, y) 

from tensorflow.keras.datasets import mnist
import numpy as np
# see ANN(3) – Step 4: load and preprocess data
# A function for data loading
# input: same as ANN’s input in the ANN(3) slides.
# output: same as the inputs
def data_load():
    (X_train, _), (X_test, _) = mnist.load_data() # under-bar for ignoring output arguments
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    return (X_train, X_test)

import matplotlib.pyplot as plt
def show_ae(autoencoder, X_test):
    encoder = autoencoder.Encoder()
    decoder = autoencoder.Decoder()
    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(3, n, i + 1 + n)
        plt.stem(encoded_imgs[i].reshape(-1), use_line_collection=True)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
def main():
    x_nodes = 784
    z_dim = 36
    (X_train, X_test)=data_load()
    autoencoder = AE(x_nodes, z_dim)
    history = autoencoder.fit(X_train, X_train,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test))
    plot_loss(history) # see the slide 27 of ANN(3)
    plt.savefig('ae_mnist.loss.png')
    plt.clf()
    plot_acc(history) # see the slide 27 of ANN(3)
    plt.savefig('ae_mnist.acc.png')
    show_ae(autoencoder, X_test)
    plt.savefig('ae_mnist.predicted.png')
    plt.show()
# when there is no code outside the class or functions.
# Running the main function as a default.

def plot_loss(h):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
def plot_acc(h):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])

if __name__ == '__main__':
    main()

