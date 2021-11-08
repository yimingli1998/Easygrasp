import torch
import torch.utils.data
import os
import numpy as np
import copy
import glob
import yaml
import trimesh
from utils import scene_utils,pc_utils,grasp_utils
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../')

with open(os.path.join(BASE_DIR,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

def vis_scene():
    for i in range(cfg['num_scenes']):
        img_id = i*cfg['num_images_per_scene']
        scene_mesh,gt_objs,transform_list = scene_utils.load_scene(img_id)
        scene_mesh.show()



def vis_scene_pointcloud():
    for i in range(cfg['num_scenes']):
        img_id = i*cfg['num_images_per_scene']
        print(img_id)
        # if img_id == 11136:
        #     continue
        xyz,rgb = scene_utils.load_scene_pointcloud(img_id+10)
        xyz,rgb,_ = pc_utils.crop_point(xyz,rgb)

        pc = trimesh.PointCloud(xyz,rgb)
        pc.show()
        # print(points.shape)

def vis_scene_grasp():
    for i in range(0,cfg['num_images'],cfg['num_images_per_scene']):
        # i = 10500
        scene_id = i//cfg['num_images_per_scene']
        scene = trimesh.Scene()
        # scene_mesh,gt_objs,transform_list = scene_utils.load_scene(i)
        # scene.add_geometry(scene_mesh)
        xyz,rgb = scene_utils.load_scene_pointcloud(i,add_noise=False)
        xyz,rgb,_ = pc_utils.crop_point(xyz,rgb)
        pc = trimesh.PointCloud(xyz,rgb)
        scene.add_geometry(pc)
        grasp = np.load(os.path.join(BASE_DIR,f'scene_grasps_table/scene_grasp_{str(scene_id).zfill(4)}.npy'),allow_pickle =True).item()
        good_points = grasp['good_point']
        grippers = grasp_utils.bat_grasp_to_gripper(good_points)
        scene.add_geometry(grippers)
        scene.show()

def vis_object_grasp():
    for obj_id in range(cfg['num_train_objects']):
        obj_id=26
        scene = trimesh.Scene()
        grasp = np.load(os.path.join(BASE_DIR,f'filtered_grasps/{str(obj_id).zfill(6)}.npy'),allow_pickle =True).item()
        model = trimesh.load(os.path.join(BASE_DIR,f'blender_models/train/lm/origin_models/obj_{str(obj_id).zfill(6)}.ply'))
        scene.add_geometry(model)
        for idx in grasp.keys():
            if 'R' not in grasp[idx].keys():
                continue
            else:
                pg = grasp[idx]
                p,R,s,d,w = pg['point'],pg['R'],pg['s'],pg['d'],pg['w']
                print(s.shape)
                grippers = grasp_utils.grasp_to_gripper(p,R,s,d,w)
                scene.add_geometry(grippers)
        scene.show()



if  __name__ =='__main__':
    # path = os.path.join(BASE_DIR,'filtered_grasps')
    # grasp_utils.filter_origin_grasps(path)

    # vis_object_grasp()
    vis_scene_grasp()
