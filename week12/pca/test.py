from tensorflow.keras import models, layers, optimizers

model = models.Sequential()
model.add(layers.Dense(32, activation="relu", input_shape=(32,32,)))
model.add(layers.Dense(1, activation="sigmoid"))
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()