## import
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn import metrics
import matplotlib.pyplot as plt
import time

TEST_DIR = "C:/Users/cjswl/python__/ann-data/chest_xray/chest_xray/test"
MODEL_DIR = "new_models/"

q1_model_before = MODEL_DIR + "chest_x_ray_Q1.h5"
q1_model_after = MODEL_DIR + "chest_x_ray_Q1_after.h5"
q2_model_before = MODEL_DIR + "chest_x_ray_Q2.h5"
q2_model_after = MODEL_DIR + "chest_x_ray_Q2_after.h5"
q3_256_model_before = MODEL_DIR + "chest_x_ray_Q3.h5"
q3_256_model_after = MODEL_DIR + "chest_x_ray_Q3_after.h5"
q3_512_model_before = MODEL_DIR + "chest_x_ray_Q4.h5"
q3_512_model_after = MODEL_DIR + "chest_x_ray_Q4_after.h5"
qe1_model_before = MODEL_DIR + "chest_x_ray_Q5.h5"
qe1_model_after = MODEL_DIR + "chest_x_ray_Q5_after.h5"

model_list = [
    q1_model_before, q1_model_after,
    q2_model_before, q2_model_after,
    q3_256_model_before, q3_256_model_after,
    q3_512_model_before, q3_512_model_after,
    qe1_model_before, qe1_model_after,
]

batch_list = [
    20, 20,
    20, 20,
    15, 15,
    10, 10,
    20, 20
]

input_shape_list = [
    (128, 128, 3), (128, 128, 3),
    (128, 128, 3), (128, 128, 3),
    (256, 256, 3), (256, 256, 3),
    (512, 512, 3), (512, 512, 3),
    (None, None, 3), (None, None, 3)
]


y_test_gen = ImageDataGenerator(rescale=1./255)
for model_str, input_shape, batch_size in zip(model_list, input_shape_list, batch_list):
    model = load_model(model_str)
    data = ImageDataGenerator(rescale=1./255)
    if input_shape[0] == None:
        test_generator = data.flow_from_directory(
        TEST_DIR,
        batch_size = batch_size,
        class_mode = 'binary'
        )
    else:
        test_generator = data.flow_from_directory(
            TEST_DIR,
            target_size = input_shape,
            batch_size = batch_size,
            class_mode = 'binary'
        )
    y_test = test_generator.classes

    y_pred = model.predict_generator(test_generator)
    auc = metrics.roc_auc_score(y_test, y_pred)
    print(auc)

    matrix = metrics.confusion_matrix(y_test, y_pred>0.5)
    print(matrix)