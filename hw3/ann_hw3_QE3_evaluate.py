## import
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn import metrics
import matplotlib.pyplot as plt
import time

TEST_DIR = "C:/Users/cjswl/python__/ann-data/chest_xray/chest_xray/test"
MODEL_DIR = "new_models_QE3/"

q1_model_before = MODEL_DIR + "chest_x_ray_Q1_QE3.h5"
q1_model_after = MODEL_DIR + "chest_x_ray_Q1_after_QE3.h5"
q2_model_before = MODEL_DIR + "chest_x_ray_Q2_QE3.h5"
q2_model_after = MODEL_DIR + "chest_x_ray_Q2_after_QE3.h5"
q3_256_model_before = MODEL_DIR + "chest_x_ray_Q3_QE3.h5"
q3_256_model_after = MODEL_DIR + "chest_x_ray_Q3_after_QE3.h5"
q3_512_model_before = MODEL_DIR + "chest_x_ray_Q4_QE3.h5"
q3_512_model_after = MODEL_DIR + "chest_x_ray_Q4_after_QE3.h5"
qe1_model_before = MODEL_DIR + "chest_x_ray_Q5_QE3.h5"
qe1_model_after = MODEL_DIR + "chest_x_ray_Q5_after_QE3.h5"

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
    (128, 128), (128, 128),
    (128, 128), (128, 128),
    (256, 256), (256, 256),
    (512, 512), (512, 512),
    (None, None), (None, None)
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
#Precision, Recall (sensitivity), Specificity, F1 score, AUC

    matrix = metrics.confusion_matrix(y_test, y_pred>0.5)
    print(matrix)
    tp = matrix[0][0]
    fn = matrix[0][1]
    fp = matrix[1][0]
    tn = matrix[1][1]
    prec = tp / (tp + fp)
    spec = tn / (tn + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (prec * recall) / (prec + recall)

