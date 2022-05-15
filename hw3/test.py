from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import time
import sklearn
import sklearn.metrics
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

TEST_DIR = "C:/Users/cjswl/python__/ann-data/chest_xray/chest_xray/test"
MODEL_DIR = "new_models/"
q1_model_before = MODEL_DIR + "chest_x_ray_Q1.h5"

model = models.load_model(q1_model_before)
data = ImageDataGenerator(rescale=1./255)

test_generator = data.flow_from_directory(
            TEST_DIR,
            target_size = (128, 128, 3),
            batch_size = 20,
            class_mode = 'binary'
)

y_test = test_generator.classes
print(type(y_test))
print(type(test_generator))
# test_generator.reset()
# y_pred = model.predict_generator(test_generator)
# print(y_pred)

# LR = linear_model.LinearRegression().fit(y_pred)