import matplotlib.pyplot as plt
import numpy as np
import pickle

from collections import defaultdict

# Init NuScenes. Requires the dataset to be stored on disk.
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

nusc = NuScenes(version='v1.0-trainval', \
                dataroot='../../../../data/', \
                verbose=True)

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

with open('details/scene_info.pkl', 'rb') as handle:
    scene_info = pickle.load(handle)

for n_scene in range(849,850):
    map_poses = map_files[scene_info[str(n_scene)]["map_name"]].render_egoposes_on_fancy_map(nusc, \
                    scene_tokens=[nusc.scene[n_scene]['token']], verbose=True,
                    render_egoposes=False, render_egoposes_range=False,
                    render_legend=False)

    plt.scatter(map_poses[:, 0], map_poses[:, 1], s=20, c='k', alpha=1.0, zorder=2)
    plt.scatter(scene_info[str(n_scene)]["trajectories_x"],
                scene_info[str(n_scene)]["trajectories_y"], s=20, c='r', alpha=1.0, zorder=2)

    out_path = "images/new_trajectories.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=1)
    plt.close()
