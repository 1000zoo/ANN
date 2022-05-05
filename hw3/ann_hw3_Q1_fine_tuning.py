## import
from tensorflow.keras.models import load_model
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import time


## tuned model 불러오기 & summary 로 구조확인
tuned_model = load_model("chest_x_ray_pretrained_model_01.h5")
conv_base = tuned_model.layers[0]

tuned_model.summary()
# conv_base.summary()

## 데이터 불러오기 & 5번 블록
from ann_hw3_pretrain import train_generator, val_generator, test_generator, \
                                plot_acc, plot_loss

conv_base.summary()

for layer in conv_base.layers:
    if layer.name.startswith("block5"):
        layer.trainable = True
conv_base.summary()

tuned_model.compile(
    optimizer = optimizers.RMSprop(learning_rate=1e-5),
    loss = "binary_crossentropy", metrics = ['accuracy']
)
starttime = time.time()
epochs = 50
val_steps = 50
steps_per_epoch = 100

history = tuned_model.fit_generator(
    train_generator, epochs = epochs, steps_per_epoch = steps_per_epoch,
    validation_data = val_generator, validation_steps = val_steps
)

train_loss = history.history["loss"]
train_acc = history.history["accuracy"]
test_loss, test_acc = tuned_model.evaluate(test_generator)
print("train_loss : ", train_loss[-1], "train_acc : ", train_acc[-1])
print("test_loss : ", test_loss, "test_acc : ", test_acc)
print("time : ", time.time() - starttime)

plot_loss(history)
plt.savefig('q1_afterloss.png')
plt.clf()
plot_acc(history)
plt.savefig('q1_afteraccuracy.png')

tuned_model.save("chest_x_ray_2step_01.h5")