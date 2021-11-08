import numpy as np
import trimesh
import os
import json
import pickle
from tqdm import tqdm
from utils import common_utils,pc_utils,grasp_utils
import torch
import copy
import yaml

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../')
with open(os.path.join(BASE_DIR,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

def load_scene_pointcloud(img_id,add_noise, use_base_coordinate=True):
    file_path = os.path.join(BASE_DIR,'data/bop_data/lm/train_pbr',str(img_id//1000).zfill(6))
    with open(os.path.join(file_path,'../../camera.json')) as f:
        intrinsics = json.load(f)
    depth_file = os.path.join(file_path,f'depth/{str(img_id%1000).zfill(6)}.png')
    rgb_file = os.path.join(file_path,f'rgb/{str(img_id%1000).zfill(6)}.jpg')
    points,rgb = pc_utils.depth_to_pointcloud(depth_file,rgb_file,intrinsics,add_noise)
    if use_base_coordinate:
        # load camera to base pose
        with open(os.path.join(file_path,'scene_camera.json')) as f:
            camera_config = json.load(f)[str(img_id%1000)]
        R_w2c = np.asarray(camera_config['cam_R_w2c']).reshape(3,3)
        t_w2c = np.asarray(camera_config['cam_t_w2c'])*0.001
        c_w = common_utils.inverse_transform_matrix(R_w2c,t_w2c)
        points = common_utils.transform_points(points,c_w)

    return points,rgb

def load_scene(img_id,use_base_coordinate = True,load_grasp = True):
    meshes = []
    file_path = os.path.join(BASE_DIR,'data/bop_data/lm/train_pbr',str(img_id//1000).zfill(6))
    # load obj poses
    with open(os.path.join(file_path,'scene_gt.json')) as f:
        gt_objs = json.load(f)[str(img_id%1000)]
    # load camera to base pose
    with open(os.path.join(file_path,'scene_camera.json')) as f:
        camera_config = json.load(f)[str(img_id%1000)]
    R_w2c = np.asarray(camera_config['cam_R_w2c']).reshape(3,3)
    t_w2c = np.asarray(camera_config['cam_t_w2c'])*0.001
    c_w = common_utils.inverse_transform_matrix(R_w2c,t_w2c)

    # create plannar
    planar = trimesh.creation.box([1,1,0.05])
    planar.visual.face_colors = [255,255,255,255]
    if not use_base_coordinate:
        planar.apply_transform(common_util.rt_to_matrix(R_w2c,t_w2c))
    meshes.append(planar)
    transform_list = []
    for obj in gt_objs:
        mesh = trimesh.load(os.path.join(BASE_DIR,'data/bop_data/lm/models','obj_' + str(obj['obj_id']).zfill(6)+'.ply'))
        T_obj = trimesh.transformations.translation_matrix(np.asarray(obj['cam_t_m2c'])*0.001)
        quat_obj = trimesh.transformations.quaternion_from_matrix(np.asarray(obj['cam_R_m2c']).reshape(3,3))
        R_obj = trimesh.transformations.quaternion_matrix(quat_obj)
        matrix_obj = trimesh.transformations.concatenate_matrices(T_obj,R_obj)
        mesh.apply_transform(matrix_obj)
        transform = matrix_obj
        if use_base_coordinate:
            mesh.apply_transform(c_w)
            transform = np.dot(c_w,transform)
        transform_list.append(transform)
        # mesh.visual.face_colors = [100,0,0,255]
        meshes.append(mesh)
    scene_mesh = np.sum(m for m in meshes)
    return scene_mesh,gt_objs,transform_list

def vis_grasp_dataset(point,rgb,grasp_point_label,grasp_label):
    scene = trimesh.Scene()
    pc = trimesh.PointCloud(point,colors = rgb)
    scene.add_geometry(pc)
    # scene.show()
    bad_point = point[grasp_point_label==-1]
    bad_pc = trimesh.PointCloud(bad_point,colors = cfg['color']['bad_point'])
    scene.add_geometry(bad_pc)
    good_point = point[grasp_point_label==0]
    good_pc = trimesh.PointCloud(good_point,colors = cfg['color']['good_point'])
    scene.add_geometry(good_pc)
    # scene.show()
    good_grasp_label = grasp_label[grasp_point_label>-1]
    good_grasp_label[:,:3] = good_point
    grippers = grasp_utils.bat_grasp_to_gripper(good_grasp_label)
    scene.add_geometry(grippers)
    scene.show()
