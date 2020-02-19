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

# dict with person token as key and other features as values
pedestrian_details = dict()

# dict with scene number as key and trajectories and map name as values
scene_info = dict()

# initializing a dict for layer names and number of points in each layer
layer_list = so_map.layer_names
layer_list.append("white_area")
layer_dict = dict.fromkeys(layer_list, 0)

# defining the sensor to extract ego_pose from sample_data,
# we need a sensor to get sample_data
sensor = "LIDAR_TOP"

for n_scene in range(850):
    # initialize the scene
    my_scene = nusc.scene[n_scene]
    # getting the map name
    cur_map = nusc.get('log', my_scene["log_token"])["location"]
    # entering the scene number and map name
    scene_info[str(n_scene)] = {"trajectories_x": [], "trajectories_y": [],\
                          "map_name": cur_map}

    # per scene person token database
    seen_person_tokens = []

    # first sample
    first_sample_token = my_scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    while True:
        for ann in sample['anns']:
            group_name = nusc.get('sample_annotation', ann)['category_name']
            if "human.pedestrian" in group_name and \
               nusc.get('sample_annotation', ann)['instance_token'] not in seen_person_tokens:

                cur_person_token = nusc.get('sample_annotation', ann)['instance_token']
                cur_person_instance = nusc.get("instance", cur_person_token)
                nbr_samples = cur_person_instance['nbr_annotations']

                # initializing the dict with the new person token
                pedestrian_details[cur_person_token] = {"translation":[],
                                                        "rotation":[],
                                                        "velocity":[],
                                                        "ego_translation":[],
                                                        "ego_rotation":[],
                                                        "ego_time": []}


                first_token = cur_person_instance['first_annotation_token']
                current_token = first_token

                moving_counter = 0
                for i in range(nbr_samples):
                    current_ann = nusc.get('sample_annotation', current_token)

                    # getting the sample corresponding to this annotation to retrieve ego details
                    annotation_sample = nusc.get('sample', current_ann['sample_token'])

                    if current_ann["attribute_tokens"]:
                        current_attr = nusc.get('attribute', current_ann['attribute_tokens'][0])['name']
                        if current_attr.split(".")[1] != "sitting_lying_down":
                            # updating pedestrian details dict
                            pedestrian_details[cur_person_token]["group"] = group_name.split(".")[-1]
                            pedestrian_details[cur_person_token]["translation"].append(
                                        current_ann["translation"])
                            pedestrian_details[cur_person_token]["rotation"].append(
                                        current_ann["rotation"])
                            pedestrian_details[cur_person_token]["scene_no"] = n_scene
                            pedestrian_details[cur_person_token]["map_name"] = cur_map
                            # only takes velocity at a particular time step
                            pedestrian_details[cur_person_token]["velocity"].append(
                                list(nusc.box_velocity(current_token)))


                            # updating ego details
                            lidar_data = nusc.get('sample_data',
                                                  annotation_sample['data'][sensor])
                            ego_token = lidar_data['ego_pose_token']
                            ego_pose = nusc.get('ego_pose', ego_token)
                            pedestrian_details[cur_person_token]["ego_translation"].append(
                                ego_pose["translation"])
                            pedestrian_details[cur_person_token]["ego_rotation"].append(
                                ego_pose["rotation"])
                            pedestrian_details[cur_person_token]["ego_time"].append(
                                ego_pose["timestamp"])

                    current_token = current_ann["next"]


                seen_person_tokens.append(cur_person_token)

        if sample['next'] != '':
            sample = nusc.get('sample', sample['next'])
        else:
            #last sample of the scene
            break

# saving all the dict files in details folder

with open('details/pedestrian_details.pkl', 'wb') as handle:
    pickle.dump(pedestrian_details, handle, protocol=pickle.HIGHEST_PROTOCOL)
