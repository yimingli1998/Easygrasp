import os
import sys
import glob
import numpy as np
import random
import copy
from random import shuffle
import h5py
import torch
from tqdm import tqdm
import time
import argparse
import loss_utils
from graspnet_dataset import GraspNetDataset
from model import backbone_pointnet2
import yaml
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../')
with open(os.path.join(BASE_DIR,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser('Save Grasp and Visualize')
parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--gpu', type=str, default=cfg['train']['gpu'], help='specify gpu device')
parser.add_argument('--theme', type=str, default=cfg['train']['theme'], help='type of train')
parser.add_argument('--model_path', type=str, default=cfg['eval']['model_path'], help='type of train')
parser.add_argument('--epoch_use', type=str, default=cfg['eval']['epoch'], help='type of train')
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
print('Using GPUs ' + os.environ["CUDA_VISIBLE_DEVICES"])


model = backbone_pointnet2().cuda()
model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load(os.path.join(f'experiment/20211001_122916_test/checkpoints/model_244.pth')))

model_gp = model.eval()

def load_graspnet_data(data_path,split='test_seen'):
    dataset_path = '../'
    test_data = GraspNetDataset(data_path,split)
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                   batch_size = FLAGS.batchsize,
                                                   shuffle=True,
                                                   num_workers = FLAGS.workers)
    if os.path.exists('output_graspnet_grasps') is False:
        os.makedirs('output_graspnet_grasps')
    model.eval()
    with torch.no_grad():
        for i, (data,index) in enumerate(tqdm(test_dataloader)):
            scene_id = data['scene_id'].item()
            img_id = data['img_id'].item()
            bat_point = data['point']
            bat_rgb = data['color'] 
            _,bat_pred_grasp = model(data['point'].cuda().float(),torch.cat([data['norm_point'].cuda().float(),data['color'].cuda().float()],dim =-1).transpose(1, 2))
            bat_pred_graspable,_ = model_gp(data['point'].cuda().float(),torch.cat([data['norm_point'].cuda().float(),data['color'].cuda().float()],dim =-1).transpose(1, 2))
            for point,rgb,pred_graspable, pred_grasp in zip(bat_point,bat_rgb,bat_pred_graspable,bat_pred_grasp):
#                 print(rgb)
                gp,R,width,depth,score = loss_utils.decode_pred(point, pred_graspable, pred_grasp)
                grasp = {
                    'point':    point.detach().cpu().numpy(),
                    'rgb':      rgb.detach().cpu().numpy(),
                    'pos':      gp.detach().cpu().numpy(),
                    'R':        R.detach().cpu().numpy(),
                    'depth':    depth.detach().cpu().numpy(),
                    'score':    score.detach().cpu().numpy(),
                    'width':    width.detach().cpu().numpy(),
                }
#                 print('***')
                np.save(f'output_graspnet_grasps/{scene_id}_{img_id}.npy',grasp)
            
def save_graspnet_grasp(test_dataloader,model):
    if os.path.exists('output_graspnet_grasps') is False:
        os.makedirs('output_graspnet_grasps')
    model.eval()
    with torch.no_grad():
        for i, (data,bat_index) in enumerate(tqdm(test_dataloader)):
            for k in data.keys():
                data[k] = data[k].cuda().float()
            bat_point = data['point']
            bat_pred_graspable,bat_pred_grasp = model(data['point'],torch.cat([data['norm_point'],data['color']],dim =-1).transpose(1, 2))
            for img_id, point, pred_graspable, pred_grasp in zip(bat_index,bat_point,bat_pred_graspable,bat_pred_grasp):
                gp,R,width,depth,score = loss_utils.decode_pred(point, pred_graspable, pred_grasp)
                grasp = {
                    'pos':      gp.cpu().numpy(),
                    'R':        R.cpu().numpy(),
                    'depth':    depth.cpu().numpy(),
                    'score':    score.cpu().numpy(),
                    'width':    width.cpu().numpy(),

                }
                np.save(f'output_grasps/{img_id}.npy',grasp)

if __name__ == '__main__':
    load_graspnet_data('../',split = 'test_seen')