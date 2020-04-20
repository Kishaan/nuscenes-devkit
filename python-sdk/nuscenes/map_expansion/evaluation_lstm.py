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

nusc = NuScenes(version='v1.0-trainval', \
                dataroot='../../../../data/', \
                verbose=False)

with open('details/scene_info.pkl', 'rb') as handle:
    scene_info = pickle.load(handle)

# so_map = NuScenesMap(dataroot='../../../../data/', \
#                        map_name='singapore-onenorth')
# bs_map = NuScenesMap(dataroot='../../../../data/', \
#                        map_name='boston-seaport')
# sh_map = NuScenesMap(dataroot='../../../../data/', \
#                        map_name='singapore-hollandvillage')
# sq_map = NuScenesMap(dataroot='../../../../data/', \
#                        map_name='singapore-queenstown')

# # dict mapping map name to map file
# map_files = {'singapore-onenorth': so_map,
#              'boston-seaport': bs_map,
#              'singapore-hollandvillage': sh_map,
#              'singapore-queenstown': sq_map}

# defining the custom rmse loss function
def model_loss(gt, pred):
    '''
    calculates custom rmse loss between every time point
    '''
    l2_x = K.square(gt[:,:,0] - pred[:,:,0])
    l2_y = K.square(gt[:,:,1] - pred[:,:,1])
    
    # log(sigma)
    logs_x = pred[:,:,2] * 0.5 
    logs_y = pred[:,:,3] * 0.5
    
    # sigma^2
    s_x = K.exp(pred[:,:,2])
    s_y = K.exp(pred[:,:,3])

    # weight for aleatoric loss
    w = 0.5
    
    r = (l2_x/(2*w*s_x)) + (l2_y/(2*w*s_y))
    return K.mean(r) + w*logs_x + w*logs_y

# defining the custom rmse loss function
def rmse_loss(gt_path, pred_path):
    '''
    calculates custom rmse loss between every time point
    '''
    return K.mean(K.sqrt(K.sum(K.square(gt_path-pred_path), axis=1))) 

def euc_dist(gt, pred):
    # custom metric to monitor rmse
    gt_path = gt
    pred_path = pred[:,:,:2]
    
    gt_x = gt_path[:,:,0]
    gt_y = gt_path[:,:,1]
    
    pred_x = pred_path[:,:,0]
    pred_y = pred_path[:,:,1]
    
    rmse = K.mean(K.sqrt(K.sum(K.square(gt_path - pred_path), axis=1)))
    return rmse

# loading the model
lstm_model = tf.keras.models.load_model("checkpoints/uncertain_lstm_best.hdf5", compile=False)

lstm_model.compile(optimizer='adam', 
                 loss=model_loss, 
                 metrics=[euc_dist])

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

rmse3_values = []
rmse4_values = []

fde3_values = []
fde4_values = []

inference_times = []

for test_idx in range(9800, len(ped_dataset)):
    test_data = total_ped_matrix[test_idx:test_idx+1,:6,:]

    start_time = time.time()
    predictions = lstm_model.predict(test_data)[:,:,:2].reshape(-1, 2)
    end_time = time.time()

    predictions = move_from_origin(rotate_from_y(predictions, ped_dataset[test_idx]["angle"]),
                                   ped_dataset[test_idx]["origin"])
    # =========== plotting ==============
    # alphas = np.linspace(1, 0.5, 10)

    # # for red the first column needs to be one
    # red_colors = np.zeros((10,4))
    # red_colors[:,0] = 1.0
    # # the fourth column needs to be your alphas
    # red_colors[:, 3] = alphas

    # # for red the first column needs to be one
    # blue_colors = np.zeros((10,4))
    # blue_colors[:,2] = 1.0
    # # the fourth column needs to be your alphas
    # blue_colors[:, 3] = alphas

    # n_scene = ped_dataset[test_idx]["scene_no"]

    # layers_gt = map_files[scene_info[str(n_scene)]["map_name"]].layers_on_point(
    #     np.array(ped_dataset[test_idx]["translation"])[-1,0], 
    #     np.array(ped_dataset[test_idx]["translation"])[-1,1]
    # )

    # layers_pred = map_files[scene_info[str(n_scene)]["map_name"]].layers_on_point(
    #     np.array(predictions)[-1,0], 
    #     np.array(predictions)[-1,1]
    # )

    # create image only if last position has different layers
    # if layers_gt["walkway"]:
    #     if not layers_pred["walkway"]:
            # loss = rmse_error(predictions, np.array(ped_dataset[test_idx]["translation"])[6:,:2])
    
            # final_loss = rmse_error(predictions[-1], 
            #                         np.array(ped_dataset[test_idx]["translation"])[-1,:2])
            
            # rmse_values.append(loss)
            # fde_values.append(final_loss)
            # ego_poses = map_files[scene_info[str(n_scene)]["map_name"]].render_pedposes_on_fancy_map(
            #                 nusc, scene_tokens=[nusc.scene[n_scene]['token']], 
            #                 ped_path = np.array(ped_dataset[test_idx]["translation"])[:,:2], verbose=False,
            #                 render_egoposes=True, render_egoposes_range=False, 
            #                 render_legend=False)

            # plt.scatter(*zip(*np.array(ped_dataset[test_idx]["translation"])[:6,:2]), c='k', s=5, zorder=2)
            # plt.scatter(*zip(*np.array(ped_dataset[test_idx]["translation"])[6:,:2]), color=blue_colors, s=5, zorder=3)
            # plt.scatter(*zip(*predictions), color=red_colors, s=5, zorder=4)
            # plt.savefig(f"images/situation_ulstm/{test_idx}.png", bbox_inches='tight', pad_inches=0)
            # only for saving in the folder
            # plt.close()
            # plt.show()

    # if layers_pred["walkway"]:
    #     if not layers_gt["walkway"]:
            # loss = rmse_error(predictions, np.array(ped_dataset[test_idx]["translation"])[6:,:2])
    
            # final_loss = rmse_error(predictions[-1], 
            #                         np.array(ped_dataset[test_idx]["translation"])[-1,:2])
            
            # rmse_values.append(loss)
            # fde_values.append(final_loss)
            # ego_poses = map_files[scene_info[str(n_scene)]["map_name"]].render_pedposes_on_fancy_map(
            #                 nusc, scene_tokens=[nusc.scene[n_scene]['token']], 
            #                 ped_path = np.array(ped_dataset[test_idx]["translation"])[:,:2], verbose=False,
            #                 render_egoposes=True, render_egoposes_range=False, 
            #                 render_legend=False)

            # plt.scatter(*zip(*np.array(ped_dataset[test_idx]["translation"])[:6,:2]), c='k', s=5, zorder=2)
            # plt.scatter(*zip(*np.array(ped_dataset[test_idx]["translation"])[6:,:2]), color=blue_colors, s=5, zorder=3)
            # plt.scatter(*zip(*predictions), color=red_colors, s=5, zorder=4)
            # plt.savefig(f"images/situation_ulstm/{test_idx}.png", bbox_inches='tight', pad_inches=0)
            # only for saving in the folder
            # plt.close()
            # plt.show()
    # =============== end of plotting ================

    loss = rmse_error(predictions, np.array(ped_dataset[test_idx]["translation"])[6:,:2])
    
    final_loss = rmse_error(predictions[-1], 
                            np.array(ped_dataset[test_idx]["translation"])[-1,:2])

    rmse3_loss = rmse_error(predictions[:6], np.array(ped_dataset[test_idx]["translation"])[6:12,:2])
    fde3_loss = rmse_error(predictions[5], 
                            np.array(ped_dataset[test_idx]["translation"])[11,:2])

    rmse4_loss = rmse_error(predictions[:8], np.array(ped_dataset[test_idx]["translation"])[6:14,:2])
    fde4_loss = rmse_error(predictions[7], 
                            np.array(ped_dataset[test_idx]["translation"])[13,:2])
    
    rmse_values.append(loss)
    fde_values.append(final_loss)

    rmse3_values.append(rmse3_loss)
    fde3_values.append(fde3_loss)

    rmse4_values.append(rmse4_loss)
    fde4_values.append(fde4_loss)

    inference_times.append(end_time-start_time)

# plt.hist(rmse_values)
# plt.title("Average displacement errors for test samples")
# plt.savefig(f"images/hist_ulstm/situation_ade.png", bbox_inches='tight', pad_inches=1)
# plt.close()

# plt.hist(fde_values)
# plt.title("Final displacement errors for test samples")
# plt.savefig(f"images/hist_ulstm/situation_fde.png", bbox_inches='tight', pad_inches=1)
# plt.close()

print("Rmse3_loss: ", np.mean(np.array(rmse3_values)))
print("FDE3_loss: ", np.mean(np.array(fde3_values)))

print("Rmse4_loss: ", np.mean(np.array(rmse4_values)))
print("FDE4_loss: ", np.mean(np.array(fde4_values)))

print("Rmse_loss: ", np.mean(np.array(rmse_values)))
print("FDE_loss: ", np.mean(np.array(fde_values)))

print("Inference_time: ", np.mean(np.array(inference_times)))
