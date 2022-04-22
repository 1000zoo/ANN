import tensorflow.keras as keras
import numpy as np

x = np.array([0,1,2,3,4])
y = x*2 + 1

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile('SGD', 'mse')

print('Before Learning: ')
print("Targets: ", y[2:])
print("predictions: ", model.predict(x[2:]).flatten())

x1 = np.array([x[2]])
y1 = np.array([y[2]])
model.fit(x1, y1, epochs=2000, verbose=0)

print("After Learning: ")
print("Targets: ", y[2:])
print("Predictions: ", model.predict(x[2:]).flatten())
print(model.get_weights())