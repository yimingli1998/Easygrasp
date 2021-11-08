import numpy as np
import trimesh
import os
import json
import pickle
from tqdm import tqdm
from utils import common_utils, scene_utils,pc_utils
import torch
import copy
import time
from scipy import spatial
import yaml
import glob

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../')
with open(os.path.join(BASE_DIR,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

class SimGraspDataset:
    def __init__(self,data_path, split='train'):
        self.data_path = data_path
        self.num_imgs = cfg['num_images']
        self.num_points = cfg['dataset']['num_points']
        self.vis = True
        self.all_files = sorted(glob.glob(os.path.join(data_path,'*')))
        if split == 'train':
            self.files = self.all_files[:30000]
        else:
            self.files = self.all_files[30000:]
        print('num files:',len(self.files))

        self.add_noise = True
        self.use_bin_loss = cfg['train']['use_bin_loss']

    def __getitem__(self, index):
        # t0 = time.time()
        # index = 91
        point_data = {}
        img_id = int(self.files[index].split('_')[-2])
        while (img_id>27999 and img_id<=28999):
            index = np.random.randint(len(self.files))
            img_id = int(self.files[index].split('_')[-2])

        scene_id = img_id//cfg['num_images_per_scene']

        point_data['img_id'] = img_id
        point_data['scene_id'] = scene_id

        grasp = np.load(os.path.join(BASE_DIR,f'scene_grasps_table/scene_grasp_{str(scene_id).zfill(4)}.npy'),allow_pickle =True).item()
        good_point =grasp['good_point']
        point,rgb = scene_utils.load_scene_pointcloud(img_id,add_noise = self.add_noise)
        point_grasp_index = np.load(f'{self.data_path}/scene_{str(img_id).zfill(6)}_label.npy',allow_pickle=True).item()
        grasp_point_label = -np.ones(len(point))*2

        point_index = list(point_grasp_index.keys())
        grasp_point_label[point_index] = list(point_grasp_index.values())
        grasp_label = np.zeros((len(point),good_point.shape[1]))
        good_point_mask = grasp_point_label>-1
        good_grasp_idx = np.array(grasp_point_label[good_point_mask],dtype = np.int)
        grasp_label[good_point_mask] = good_point[good_grasp_idx]
        grasp_point_label[good_point_mask] = 0
        # if self.vis:
        #     scene_utils.vis_grasp_dataset(point,rgb,grasp_point_label,grasp_label)

        point,rgb,crop_mask = pc_utils.crop_point(point,rgb)
        grasp_point_label,grasp_label = grasp_point_label[crop_mask],grasp_label[crop_mask]

        # if self.vis:
        #     scene_utils.vis_grasp_dataset(point,rgb,grasp_point_label,grasp_label)



        choice = np.random.choice(len(point),self.num_points,replace=True)
        point,rgb,grasp_point_label,grasp_label = point[choice], \
                                                  rgb[choice], \
                                                  grasp_point_label[choice], \
                                                  grasp_label[choice]
        # if self.vis:
        #     scene_utils.vis_grasp_dataset(point,rgb,grasp_point_label,grasp_label)

        point_data['point'] = point
        point_data['color'] = rgb
        center = np.mean(point,axis=0)
        norm_point = point - center
        point_data['norm_point'] = norm_point

        # random select planar points as negative data
        random_choice = np.random.choice(len(point),
                                         int(cfg['dataset']['sample_planar_rate']*len(point)),
                                         replace=False)
        grasp_point_label[random_choice] = -1
        print(len(grasp_point_label[grasp_point_label==-1]))
        point_data['grasp_point_label'] = grasp_point_label + 1

        if self.vis:
            scene_utils.vis_grasp_dataset(point,rgb,grasp_point_label,grasp_label)


        # point_data['grasp_label'] = grasp_label[:,3:]
        R = grasp_label[:,3:12].reshape(-1,3,3)
        # width: [0,0.08],depth:[0.01,0.04],score:[0.1,0.4]
        width = np.around(grasp_label[:,-3],5)
        depth = np.around(grasp_label[:,-2],5)
        score = np.around(grasp_label[:,-1],5)
        if self.use_bin_loss:
            mask = point_data['grasp_point_label'] > 0
            grasp_label = np.zeros((len(point),6))
            approach = R[mask][:,:3,0]
            binormal = R[mask][:,:3,1]
            angle_ = np.arctan2(binormal[:,1], binormal[:,0])
            angle_[angle_<0] = angle_[angle_<0] + np.pi
            grasp_angle = angle_ / np.pi * 180
            azimuth_ = np.arctan2(approach[:,1], approach[:,0])
            azimuth_[azimuth_<0] = azimuth_[azimuth_<0] + np.pi*2.
            azimuth_angle = azimuth_ / np.pi * 180
            elevation_ = np.arctan2(-approach[:,2], np.sqrt(approach[:,0]**2 + approach[:,1]**2)) + np.pi/2.
            elevation_angle = elevation_ / np.pi * 180
            width = width[mask]
            depth = depth[mask]
            score = score[mask]
            print(len(score))
            grasp_label[mask] = np.vstack([azimuth_angle,elevation_angle,grasp_angle,width*100.0,depth*100.0 -1,score*10.0 -1 ]).transpose(1,0)
            point_data['grasp_label'] = grasp_label

        return point_data,img_id

    def __len__(self):
        return len(self.files)
        # return 800

if  __name__ =='__main__':
    data_path = os.path.join(BASE_DIR,'point_grasp_data_2mm_with_nms')
    gd = SimGraspDataset(data_path)
    dataloader = torch.utils.data.DataLoader(gd, batch_size=1, shuffle=True,
                                             num_workers=1)
    t0 = time.time()
    for i, (data,img_id) in enumerate(tqdm(dataloader)):
        pass
    print(f'time cost:{time.time()-t0}s')