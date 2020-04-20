import tensorflow as tf
from tensorflow.keras import backend as K

import matplotlib as mpl
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import os

# Init NuScenes. Requires the dataset to be stored on disk.
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

matplotlib.rcParams['figure.figsize'] = (24, 18)
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams.update({'font.size': 20})

TRAIN_SIZE = 9800
TRAIN_TIME = 6
BATCH_SIZE = 32
BUFFER_SIZE = 500

total_ped_matrix = np.load("details/new_ped_matrix.npy")

with open("details/ped_dataset.pkl", "rb") as f:
    ped_dataset = pickle.load(f)

# defining the custom rmse loss function
def rmse_loss(gt_path, pred_path):
    '''
    calculates custom rmse loss between every time point
    '''
    gt_path = tf.reshape(gt_path, [-1, 10, 2])
    pred_path = tf.reshape(pred_path, [-1, 10, 2])
    
    return K.mean(K.sqrt(K.sum(K.square(gt_path-pred_path), axis=1)))

# loading the model
fc_model = tf.keras.models.load_model("checkpoints/mlp_best.hdf5", compile=False)

fc_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), 
                 loss=rmse_loss, 
                 metrics=["accuracy"])

# undo normalization for plotting
def move_from_origin(l, origin):
    x0, y0 = origin
    return [[x + x0, y + y0] for x, y in l]

def rotate_from_y(l, angle):
    theta = -angle
    return [(x*np.cos(theta) - y*np.sin(theta), 
                    x*np.sin(theta) + y*np.cos(theta)) for x, y in l]

# loss calculation for test prediction
def rmse_error(l1, l2):
    loss = []
    
    if len(np.array(l1).shape) < 2:
        return ((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)**0.5
    for p1, p2 in zip(l1, l2):
        loss.append(((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5)
    
    loss = np.array(loss)
    return np.mean(loss)

rmse_values = []
fde_values = []
inference_times = []

for test_idx in range(9800, len(ped_dataset)):
    test_data = np.reshape(total_ped_matrix[test_idx,:6,:]
                           , (1, 42))

    start_time = time.time()
    predictions = fc_model.predict(test_data).reshape(-1, 2)
    end_time = time.time()

    predictions = move_from_origin(rotate_from_y(predictions, ped_dataset[test_idx]["angle"]),
                                   ped_dataset[test_idx]["origin"])
    loss = rmse_error(predictions, np.array(ped_dataset[test_idx]["translation"])[6:,:2])
    
    final_loss = rmse_error(predictions[-1], 
                            np.array(ped_dataset[test_idx]["translation"])[-1,:2])
    
    rmse_values.append(loss)
    fde_values.append(final_loss)
    inference_times.append(end_time-start_time)

print("Rmse_loss: ", np.mean(np.array(rmse_values)))
print("FDE_loss: ", np.mean(np.array(fde_values)))
print("Inference_time: ", np.mean(np.array(inference_times)))
