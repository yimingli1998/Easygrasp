import numpy as np
import trimesh
import os
import json
import pickle
from tqdm import tqdm
from utils import common_utils, scene_utils,pc_utils,grasp_utils
import torch
import copy
import time
from scipy import spatial
import yaml
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../')

with open(os.path.join(BASE_DIR,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

def match_point_grasp(img_id,save_path):
    scene = trimesh.Scene()
    grasp_path = os.path.join(BASE_DIR,'scene_grasps_table')
    scene_id = img_id//cfg['num_images_per_scene']
    point,rgb = scene_utils.load_scene_pointcloud(img_id, use_base_coordinate=cfg['use_base_coordinate'],add_noise = False)
    grasp = np.load(
        os.path.join(grasp_path, f'scene_grasp_{str(scene_id).zfill(4)}.npy'),
        allow_pickle = True).item()

    point_grasp_dict = {}
    kdtree = spatial.KDTree(point)
    bad_point = grasp['bad_point']
    # pc_bad = trimesh.PointCloud(bad_point,colors = [0,255,0,255])
    # scene.add_geometry(pc_bad)
    good_point = grasp['good_point'][:,:3]
    good_point_score = grasp['good_point'][:,-1].round(1)

    pc_good = trimesh.PointCloud(good_point,colors = [255,0,0,255])
    # scene.add_geometry(pc_good)
    # scene.show()
    points_query = kdtree.query_ball_point(bad_point, 0.002)
    points_query = [item for sublist in points_query for item in sublist]
    points_query = list(set(points_query))
    for index in points_query:
        point_grasp_dict[index] = -1

    for s in np.unique(good_point_score)[::-1]:
        good_point_with_score = good_point[good_point_score==s]
        if len(good_point_with_score) >0:
            # print(len(good_point_with_score))
            points_query = kdtree.query_ball_point(good_point_with_score, 0.002)
            for i,pq in enumerate(points_query):
                if pq != []:
                    for index in pq:
                        point_grasp_dict[index] = i
    np.save(f'{save_path}/scene_{str(img_id).zfill(6)}_point.npy',point)
    np.save(f'{save_path}/scene_{str(img_id).zfill(6)}_label.npy',point_grasp_dict)

    print(f'scene_{str(img_id).zfill(6)} finished!')

def parallel_match_point_grasp(proc,save_path):
    from multiprocessing import Pool
    p = Pool(processes = proc)
    res_list = []
    for i in range(0,cfg['num_images']):
        res_list.append(p.apply_async(match_point_grasp, (i,save_path,)))
    p.close()
    p.join()
    for res in tqdm(res_list):
        res.get()

if  __name__ =='__main__':
    save_path = 'point_grasp_data'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for i in range(1000,32000,32):
        match_point_grasp(i,save_path)
    # parallel_match_point_grasp(6,save_path)

