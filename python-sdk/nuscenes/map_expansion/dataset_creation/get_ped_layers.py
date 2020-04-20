import matplotlib.pyplot as plt
import numpy as np
import pickle

# from collections import defaultdict
# from shapely.geometry import Point, Polygon

# Init NuScenes. Requires the dataset to be stored on disk.
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# LOADING the necessary files
with open('../details/scene_info.pkl', 'rb') as handle:
    scene_info = pickle.load(handle)
    
with open('../details/pedestrian_details.pkl', 'rb') as handle:
    pedestrian_details = pickle.load(handle)
    
with open('../details/new_ped_details.pkl', 'rb') as handle:
    new_ped_details = pickle.load(handle)

with open('../details/ped_dataset.pkl', 'rb') as handle:
    ped_dataset = pickle.load(handle)

so_map = NuScenesMap(dataroot='../../../../../data/', \
                       map_name='singapore-onenorth')
bs_map = NuScenesMap(dataroot='../../../../../data/', \
                       map_name='boston-seaport')
sh_map = NuScenesMap(dataroot='../../../../../data/', \
                       map_name='singapore-hollandvillage')
sq_map = NuScenesMap(dataroot='../../../../../data/', \
                       map_name='singapore-queenstown')

# dict mapping map name to map file
map_files = {'singapore-onenorth': so_map,
             'boston-seaport': bs_map,
             'singapore-hollandvillage': sh_map,
             'singapore-queenstown': sq_map}

for idx in range(len(ped_dataset)):
    print(idx)
    ped_dataset[idx]["ped_layers"] = [] 
    for j in range(len(ped_dataset[idx]["translation"])):
        n_scene = ped_dataset[idx]["scene_no"]
        cur_layers = map_files[scene_info[str(n_scene)]["map_name"]].layers_on_point(
            np.array(ped_dataset[idx]["translation"])[j,0], 
            np.array(ped_dataset[idx]["translation"])[j,1]
        )
        layers_list = [l for l in cur_layers.keys() if cur_layers[l]]
        ped_dataset[idx]["ped_layers"].append(layers_list)

with open('../details/ped_dataset.pkl', 'wb') as handle:
    pickle.dump(ped_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)