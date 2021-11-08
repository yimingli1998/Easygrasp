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
import h5py

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../')
with open(os.path.join(BASE_DIR,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

class GraspNetDataset:
    def __init__(self,data_path, split='train'):
        self.data_path = data_path
        self.num_imgs = cfg['num_images']
        self.num_points = cfg['dataset']['num_points']
        self.vis = False
        self.files = sorted(glob.glob(self.data_path + 'grasp_hdf5_20000_new/' + split + '/' + '*.h5'))
        print('num files:',len(self.files))
        self.use_bin_loss = cfg['train']['use_bin_loss']

    def __getitem__(self, index):
        # t0 = time.time()
        file_path = self.files[index]
        fin = h5py.File(file_path, 'r')
        coords = np.asarray(fin['coord'])
        points = np.asarray(fin['point'])
        graspPoints_labels = np.asarray(fin['grasp_point'])
        grasps = np.asarray(fin['grasp'])
        
        point_data = {}

        scene_id = int(file_path.split('/')[-1].split('_')[0][-4:])
        img_id = int(file_path.split('/')[-1].split('_')[-1].split('.')[0])
  
        point_data['img_id'] = img_id
        point_data['scene_id'] = scene_id
        point_data['point'] = coords
        point_data['color'] = points[:,3:6]*255.0
#         print(point_data['color'])
        center = np.mean(coords,axis=0)
        norm_point = coords - center
        point_data['norm_point'] = norm_point
        point_data['grasp_point_label'] = graspPoints_labels

        grasp_mask = (grasps[:,7]<0.08) & (grasps[:,9] == False) 
        grasps = grasps[grasp_mask]
#         print(grasps.shape)
        if self.use_bin_loss:
            grasp_label = np.zeros((len(coords),6))
            approach,binormal,depth,width,score,collision,p_idx,g_idx = grasps[:,:3],grasps[:,3:6],grasps[:,6],grasps[:,7],1.1-grasps[:,8],grasps[:,9],np.asarray(grasps[:,10],dtype=np.int),grasps[:,11]
            score[score>=0.4] = 0.4
            angle_ = np.arctan2(binormal[:,1], binormal[:,0])
            angle_[angle_<0] = angle_[angle_<0] + np.pi
            grasp_angle = angle_ / np.pi * 180
            azimuth_ = np.arctan2(approach[:,1], approach[:,0])
            azimuth_[azimuth_<0] = azimuth_[azimuth_<0] + np.pi*2.
            azimuth_angle = azimuth_ / np.pi * 180
            elevation_ = np.arctan2(-approach[:,2], np.sqrt(approach[:,0]**2 + approach[:,1]**2)) + np.pi/2.
            elevation_angle = elevation_ / np.pi * 180
#             print('azimuth_angle:', azimuth_angle.max(),'\t',azimuth_angle.min())
#             print('elevation_angle:', elevation_angle.max(),'\t',elevation_angle.min())
#             print('grasp_angle:', grasp_angle.max(),'\t',grasp_angle.min())
#             print('depth:', depth.max(),'\t',depth.min())
#             print('width:', width.max(),'\t',width.min())
#             print('score:', score.max(),'\t',score.min())
            grasp_label[p_idx] = np.vstack([azimuth_angle,elevation_angle,grasp_angle,width*100.0,depth*100.0,score*10.0 -1]).transpose(1,0)
            point_data['grasp_label'] = grasp_label

        return point_data,img_id

    def __len__(self):
        return len(self.files)
        # return 800

if  __name__ =='__main__':
    data_path = os.path.join(BASE_DIR)
    gnd = GraspNetDataset(data_path)
    dataloader = torch.utils.data.DataLoader(gnd, batch_size=2, shuffle=True,
                                             num_workers=1)
    t0 = time.time()
    for i, (data,img_id) in enumerate(tqdm(dataloader)):
        pass
    print(f'time cost:{time.time()-t0}s')