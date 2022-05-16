import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

sample = "hi hello"
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

dic_size = len(char2idx)
hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
sequence_length = len(sample) - 1
lr = 0.01

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

inputs = to_categorical(x_data, num_classes)
outputs = to_categorical(y_data, num_classes)

def build_model():
    model = Sequential()
    # fill here
    model.summary()
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def print_string(predictions):
    for i, prediction in enumerate(predictions):
        result_str = [idx2char[c] for c in np.argmax(prediction, axis=1)]
        print("\tstr:", "".join(result_str))

model = build_model()
for i in range(30):
    model.fit(inputs, outputs, epochs=5)
    predictions = model.predict(inputs)
    print_string(predictions)

