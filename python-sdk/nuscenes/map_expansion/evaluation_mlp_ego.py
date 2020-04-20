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

total_ped_matrix = np.load("details/ego_ped_matrix.npy")

with open("details/ped_dataset.pkl", "rb") as f:
    ped_dataset = pickle.load(f)

with open('details/scene_info.pkl', 'rb') as handle:
    scene_info = pickle.load(handle)

nusc = NuScenes(version='v1.0-trainval', \
                dataroot='../../../../data/', \
                verbose=False)

so_map = NuScenesMap(dataroot='../../../../data/', \
                       map_name='singapore-onenorth')
bs_map = NuScenesMap(dataroot='../../../../data/', \
                       map_name='boston-seaport')
sh_map = NuScenesMap(dataroot='../../../../data/', \
                       map_name='singapore-hollandvillage')
sq_map = NuScenesMap(dataroot='../../../../data/', \
                       map_name='singapore-queenstown')

# dict mapping map name to map file
map_files = {'singapore-onenorth': so_map,
             'boston-seaport': bs_map,
             'singapore-hollandvillage': sh_map,
             'singapore-queenstown': sq_map}

# defining the custom rmse loss function
def ttc_loss(pred_path, ego_path):
    tot_loss = 0.0
    for i in range(pred_path.shape[0]):
        del_time = 0.0
        del_dist = np.inf
        for j in range(pred_path.shape[1]):
            for k in range(ego_path.shape[1]):
                pred_pos = pred_path[i,j,:]
                ego_pos = ego_path[i,k,:]
                
                dist = np.sqrt(np.sum((pred_pos - ego_pos)**2))
                if dist < del_dist:
                    del_dist = dist
                    del_time = abs(j-k)
                    
        tot_loss += del_time
    return np.float32(tot_loss/float(len(pred_path)))  

def model_loss(gt, pred):
    '''
    calculates custom rmse loss between every time point
    '''
    pred_path = tf.reshape(pred, [-1,10,2])
    gt_path = gt[:,:20]
    gt_path = tf.reshape(gt_path, [-1,10,2])
    
    rmse_error = K.mean(K.sqrt(K.sum(K.square(gt_path-pred_path), axis=1)))
    
    ego_path = gt[:,20:]
    ego_path = tf.reshape(ego_path, [-1,10,2])
    
    ttc_error = tf.numpy_function(ttc_loss, [pred_path, ego_path], tf.float32)
    return rmse_error

def euc_dist(gt, pred):
    # custom metric to monitor rmse
    gt = gt[:,:20]
    gt_path = tf.reshape(gt, [-1,10,2])
    pred_path = tf.reshape(pred, [-1,10,2])
    rmse = K.mean(K.sqrt(K.sum(K.square(gt_path - pred_path), axis=1)))
    return rmse

# loading the model
fc_ego_model = tf.keras.models.load_model("checkpoints/mlp_ttc_ego.hdf5", compile=False)

fc_ego_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), 
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
inference_times = []

situation_idxs = [9804, 9821, 9867, 9868, 9869, 9871, 9877, 9878, 9891, 9941, 10034, 10073, 10099, 
                  10106, 10119, 10129, 10142, 10143, 10196, 10245, 10256, 10302, 10330, 10365, 
                  10369, 10373, 10392, 10397, 10434, 10441, 10492, 10502, 10559, 10561, 10605, 
                  10615, 10628, 10693, 10792, 10797, 10798, 10805, 10810, 10828, 10906, 10921,
                  10939, 10945, 10980, 11004, 11016, 11034, 11179, 11186, 11231, 11232, 11295, 
                  11334, 11335, 11340, 11341, 11366, 11367, 11397, 11417, 11449, 11487, 11541,
                  11629, 11644, 11662, 11679, 11682, 11697, 11698, 11708, 11713, 11745, 11751, 
                  11804, 11827, 11828, 11830, 11838, 11971, 11980, 11984, 11989, 12041, 12077,
                  12173, 12200, 12201, 12205, 12286, 12309]

for test_idx in range(9800, len(ped_dataset)):
    test_data = np.reshape(total_ped_matrix[test_idx,:,:6,:]
                           , (1, 84))

    start_time = time.time()
    predictions = fc_ego_model.predict(test_data).reshape(-1, 2)
    end_time = time.time()

    predictions = move_from_origin(rotate_from_y(predictions, ped_dataset[test_idx]["angle"]),
                                   ped_dataset[test_idx]["origin"])

    #=========== plotting ==============
    alphas = np.linspace(1, 0.5, 10)

    # for red the first column needs to be one
    red_colors = np.zeros((10,4))
    red_colors[:,0] = 1.0
    # the fourth column needs to be your alphas
    red_colors[:, 3] = alphas

    # for red the first column needs to be one
    blue_colors = np.zeros((10,4))
    blue_colors[:,2] = 1.0
    # the fourth column needs to be your alphas
    blue_colors[:, 3] = alphas

    n_scene = ped_dataset[test_idx]["scene_no"]

    if test_idx in situation_idxs:
        ego_poses = map_files[scene_info[str(n_scene)]["map_name"]].render_pedposes_on_fancy_map(
                        nusc, scene_tokens=[nusc.scene[n_scene]['token']], 
                        ped_path = np.array(ped_dataset[test_idx]["translation"])[:,:2], verbose=False,
                        render_egoposes=True, render_egoposes_range=False, 
                        render_legend=False)

        plt.scatter(*zip(*np.array(ped_dataset[test_idx]["translation"])[:6,:2]), c='k', s=5, zorder=2)
        plt.scatter(*zip(*np.array(ped_dataset[test_idx]["translation"])[6:,:2]), color=blue_colors, s=5, zorder=3)
        plt.scatter(*zip(*predictions), color=red_colors, s=5, zorder=4)
        plt.savefig(f"images/ego_ttc/with_ttc/{test_idx}_mlp.png", bbox_inches='tight', pad_inches=0)
        #only for saving in the folder
        plt.close()
        #plt.show()
    loss = rmse_error(predictions, np.array(ped_dataset[test_idx]["translation"])[6:,:2])
    
    final_loss = rmse_error(predictions[-1], 
                            np.array(ped_dataset[test_idx]["translation"])[-1,:2])
    
    rmse_values.append(loss)
    fde_values.append(final_loss)
    inference_times.append(end_time-start_time)

print("Rmse_loss: ", np.mean(np.array(rmse_values)))
print("FDE_loss: ", np.mean(np.array(fde_values)))
print("Inference_time: ", np.mean(np.array(inference_times)))
