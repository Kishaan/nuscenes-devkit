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
    
with open('details/scene_info.pkl', 'rb') as handle:
    scene_info = pickle.load(handle)

# train_test split
x_train = total_ped_matrix[:TRAIN_SIZE, :TRAIN_TIME, :]
y_train = total_ped_matrix[:TRAIN_SIZE, TRAIN_TIME:, :2]
y_train = y_train.reshape(TRAIN_SIZE, 20)

x_test = total_ped_matrix[TRAIN_SIZE:, :TRAIN_TIME, :]
y_test = total_ped_matrix[TRAIN_SIZE:, TRAIN_TIME:, :2]

# predict based on previous position and velocity
def ca_model(idx):
    ped_trajectory = np.array(ped_dataset[idx]['translation'])[:6, :2]
    pos_t0 = ped_trajectory[-1]
    delt_t0 = np.diff(ped_trajectory, axis=0)[-1]
    
    dt = 0.5
    vel_t0 = delt_t0 / dt

    vel = np.diff(ped_trajectory, axis=0) / dt
    acc_t0 = np.diff(vel, axis=0)[-1]

    ca_predictions = []

    cur_pos = pos_t0 

    for _ in range(len(ped_dataset[idx]['translation']) - TRAIN_TIME):
        new_pos = cur_pos + dt*vel_t0 + 0.5*acc_t0*(dt**2)
        ca_predictions.append(new_pos)
        cur_pos = new_pos
        
    return ca_predictions

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

for test_idx in range(TRAIN_SIZE, len(ped_dataset)):
    test_data = np.reshape(total_ped_matrix[test_idx,:6,:]
                           , (1, 42))

    start_time = time.time()
    predictions = ca_model(test_idx)
    end_time = time.time()
#     n_scene = ped_dataset[test_idx]["scene_no"]
#     ego_poses = map_files[scene_info[str(n_scene)]["map_name"]].render_egoposes_on_fancy_map(
#                     nusc, scene_tokens=[nusc.scene[n_scene]['token']], verbose=False,
#                     render_egoposes=True, render_egoposes_range=False, 
#                     render_legend=False)

#     plt.scatter(*zip(*np.array(ped_dataset[test_idx]["translation"])[:6,:2]), c='k', s=5, zorder=2)
#     plt.scatter(*zip(*np.array(ped_dataset[test_idx]["translation"])[6:,:2]), c='b', s=5, zorder=3)
#     plt.scatter(*zip(*predictions), c='r', s=5, zorder=4)
#     plt.show()
    
    loss = rmse_error(predictions, 
                              np.array(ped_dataset[test_idx]["translation"])[6:,:2])
        
    final_loss = rmse_error(predictions[-1], 
                            np.array(ped_dataset[test_idx]["translation"])[-1,:2])
    
    rmse_values.append(loss)
    fde_values.append(final_loss)
    inference_times.append(end_time-start_time)

print("Rmse_loss: ", np.mean(np.array(rmse_values)))
print("FDE_loss: ", np.mean(np.array(fde_values)))
print("Inference_time: ", np.mean(np.array(inference_times)))
