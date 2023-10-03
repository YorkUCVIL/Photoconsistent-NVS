import os
import sys
from os.path import join as pjoin
import argparse

# add parent dir to path
script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(src_path)
os.chdir(src_path)

import colmap_utils
import numpy as np
import json
from colmap_utils import *

import argparse
argParser = argparse.ArgumentParser(description="")
argParser.add_argument("-o",dest="output_dir",action="store",default='gt',type=str)
cli_args = argParser.parse_args()

scene_root = os.path.join('../instance-data',cli_args.output_dir)
gen_runs = os.listdir(pjoin( scene_root,'samples' ))
gen_runs.sort()
newest_gen_run = gen_runs[-1]
colmap_root = pjoin(scene_root,'colmap',newest_gen_run)

# create evaluation dir
metric_path = os.path.join(scene_root,'evaluation',newest_gen_run,'sed')
os.makedirs(metric_path,exist_ok=True)

# read colmap database
db_path = os.path.join(colmap_root,'colmap.db')
reader = colmap_utils.DB_reader(db_path)
keypoint_data = colmap_utils.extract_keypoints(reader.get_table_data('keypoints'))
match_data = colmap_utils.extract_matches(reader.get_table_data('matches'))

# read gt pose spec
spec_path = pjoin(scene_root,'sampling-spec.json')
with open(spec_path) as f:
    spec = json.load(f)
    poses = spec['poses']
    poses = [np.array(x) for x in poses]
    focal_y = spec['focal_y']

# use latest sampling idx
sampling_instances = os.listdir(os.path.join(scene_root,'samples'))
sampling_instances.sort()
im_root = os.path.join(scene_root,'samples',sampling_instances[-1],'images')
available_images = os.listdir(im_root)
available_images.sort()


def get_matches_for_pair(id1,id2):
    for matches in match_data:
        if matches[0][0] == id1 and matches[0][1] == id2:
            return matches[1], False
        elif matches[0][0] == id2 and matches[0][1] == id1:
            return matches[1], True
    return None, None

def get_essential_matrix(cam_world_1,cam_world_2,intrinsics):
    '''
    Formulation is funny, put P_L on the left, for the matrix, P_R on the right
    The matrix provided uses blender coordinates: y up, z backward
    '''
    # convert poses to world relative to 1
    cam_world_1_inv = np.linalg.inv(cam_world_1)
    cam_world_rel_2 = np.matmul(cam_world_1_inv,cam_world_2)

    # get relative offset
    T = cam_world_rel_2[:3,3:4]

    # compute fundamental matrix
    R = np.linalg.inv(cam_world_rel_2[:3,:3]) # rotation to get relative world points to camera 2 coordinate
    Tx = np.array([ # skew symmetric for crossproduct
        [0,-T[2,0],T[1,0]],
        [T[2,0],0,-T[0,0]],
        [-T[1,0],T[0,0],0],
    ])
    F = np.matmul(R,Tx).T

    # compute essential_matrix
    intrinsics_inv = np.linalg.inv(intrinsics)
    E = np.matmul(F,intrinsics_inv)
    E = np.matmul(intrinsics_inv.T,E)

    return E

def get_min_dist(p_1,E,kp):
    epipolar_norm_2 = np.matmul(p_1,E)[0,:]
    norm_2d = epipolar_norm_2*1 # clone
    norm_2d[2] = 0
    norm_2d /= np.linalg.norm(norm_2d)
    kp_h = np.array(kp[:2].tolist()+[1])

    s = -np.dot(epipolar_norm_2,kp_h) / np.dot(epipolar_norm_2,norm_2d)
    return norm_2d, s

# compute sed for neighboring images
median_seds = []
for idx_1 in range(len(available_images)-1):
    idx_2 = idx_1 + 1

    # find im_ids
    vis_pair = [available_images[idx_1],available_images[idx_2]]
    id1 = reader.get_image_id_from_fn(vis_pair[0])
    id2 = reader.get_image_id_from_fn(vis_pair[1])

    # load match info
    cur_matches, flip = get_matches_for_pair(id1,id2)
    if cur_matches is None:
        cur_matches = []
    else:
        if flip: cur_matches = np.flip(cur_matches,axis=1)

    # load pose/camera data
    pose_1, pose_2 = poses[idx_1],poses[idx_2]
    intrinsics_realestate = np.array([
        [focal_y*256,0,128],
        [0,focal_y*256,128],
        [0,0,1],
    ])
    blender_conversion = np.array([
        [1,0,0],
        [0,-1,0],
        [0,0,-1],
    ])
    intrinsics = np.matmul(intrinsics_realestate,blender_conversion)

    E_12 = get_essential_matrix(pose_1,pose_2,intrinsics)
    E_21 = get_essential_matrix(pose_2,pose_1,intrinsics)

    seds = []
    for match in cur_matches:
        # load match info
        cur_matches, flip = get_matches_for_pair(id1,id2)
        if flip: cur_matches = np.flip(cur_matches,axis=1)
        if cur_matches is None: cur_matches = []

        kp1 = keypoint_data[id1][match[0]]
        kp2 = keypoint_data[id2][match[1]]

        p1 = np.array([ kp1[:2].tolist()+[1] ])
        p2 = np.array([ kp2[:2].tolist()+[1] ])

        # project 3d normal to get normal of 2d line
        _,ed_2 = get_min_dist(p1,E_12,kp2)
        _,ed_1 = get_min_dist(p2,E_21,kp1)

        sed = 0.5*(abs(ed_2) + abs(ed_1))
        seds.append(sed)
    n_matches = len(seds)
    median = 99999999999 if n_matches == 0 else np.median(seds)
    median_seds.append({'pair':vis_pair,'median':median,'n_matches':n_matches})

with open(pjoin(metric_path,'neighbors.json'),'w') as f:
    json.dump(median_seds,f)

