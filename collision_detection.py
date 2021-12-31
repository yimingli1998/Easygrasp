import numpy as np
import trimesh
import os
import json
import pickle
from tqdm import tqdm

from utils import common_utils,scene_utils,grasp_utils
import torch
import copy
import yaml

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../')
with open(os.path.join(BASE_DIR,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

def collision_checker(index):
    print(f"scene {str(index//cfg['num_images_per_scene'])} begin!\n")
    scene = trimesh.Scene()
    scene_mesh,gt_objs,transform_list = scene_utils.load_scene(index,use_base_coordinate = cfg['use_base_coordinate'])
    scene.add_geometry(scene_mesh)
    # scene.show()
    collision_manager,_ = trimesh.collision.scene_to_collision(scene)
    scene_grasp = {'bad_point':[],
                   'good_point':[]}
    scene_grasp_path = os.path.join('scene_grasps')
    if os.path.exists(scene_grasp_path) is False:
        os.makedirs(scene_grasp_path)
    for obj,transform in tqdm(zip(gt_objs,transform_list),total=len(gt_objs),desc= f"collision detection for scene {str(index//cfg['num_images_per_scene'])}"):
        obj_name = str(obj['obj_id'])
        # print(obj_name)
        # if obj_name!= '26':
        #     continue
        grasps = np.load(os.path.join(BASE_DIR,f'filtered_grasps/{obj_name.zfill(6)}.npy'),allow_pickle=True).item()
        # print('***')
        # print('grasps')
        for idx in grasps.keys():
            pg = grasps[idx]
            # print(grasps[idx].keys())
            scene_point = common_utils.transform_points(pg['point'][:3][np.newaxis,:],transform)[0]
            if 'R' not in grasps[idx].keys():
                scene_grasp['bad_point'].append(scene_point)
            else:
                R,s,d,w = pg['R'],pg['s'],pg['d'],pg['w']
                if len(R) > cfg['num_grasps_per_point']:
                    choice = np.random.choice(len(R),cfg['num_grasps_per_point'])
                    R,s,d,w = R[choice],s[choice],d[choice],w[choice]
                for i in range(len(R)):
                    trans_R = np.dot(transform[:3,:3],R[i])
                    approach = trans_R[:3,0]
                    if approach[2] > 0:
                        continue
                    binormal = trans_R[:3,1]
                    if binormal[2] > 0.5 or binormal[2] < -0.5:
                        continue
                    gripper = grasp_utils.plot_gripper_pro_max(scene_point,trans_R,w[i],d[i],1.1-s[i])
                    collision  = collision_manager.in_collision_single(gripper)
                    if not collision:
                        grasp_label = np.concatenate([scene_point,trans_R.reshape(-1),[w[i],d[i],s[i]]])
                        scene_grasp['good_point'].append(grasp_label)
                        # scene.add_geometry(gripper)
                        # scene.show()
                        break
                else:
                    # scene.add_geometry(gripper)
                    # scene.show()
                    scene_grasp['bad_point'].append(scene_point)

    scene_grasp['bad_point'] = np.asarray(scene_grasp['bad_point'])
    scene_grasp['good_point'] = np.asarray(scene_grasp['good_point'])
    print(f"scene {str(index//cfg['num_images_per_scene'])} finished!\n bad point:{len(scene_grasp['bad_point'])}\n good point:{len(scene_grasp['good_point'])}")
    np.save(os.path.join(scene_grasp_path,f"scene_grasp_{str(index//cfg['num_images_per_scene']).zfill(4)}.npy"),scene_grasp)

def parallel_collision_checker(proc):
    from multiprocessing import Pool
    p = Pool(processes = proc)
    res_list = []
    for i in range(0,cfg['num_images']):
        if i % cfg['num_images_per_scene'] == 0:
            res_list.append(p.apply_async(collision_checker, (i,)))
    p.close()
    p.join()
    for res in tqdm(res_list):
        res.get()

if  __name__ =='__main__':

    # parallel_collision_checker(proc = 32)
    collision_checker(10500)
