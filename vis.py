import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os
from tqdm import tqdm
import numpy as np
import argparse
import time
import copy
from dataset import SimGraspDataset
import yaml
from utils import scene_utils,grasp_utils
import loss_utils
import glob
import trimesh

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

def save_grasp(test_dataloader,model):
    if os.path.exists('output_grasps') is False:
        os.makedirs('output_grasps')
    model.eval()
    with torch.no_grad():
        for i, (data,bat_index) in enumerate(tqdm(test_dataloader)):
            for k in data.keys():
                data[k] = data[k].cuda().float()
            bat_point = data['point']
            bat_pred_graspable,bat_pred_grasp = model(data['point'],torch.cat([data['norm_point'],data['color']],dim =-1).transpose(1, 2))
            for img_id, point, pred_graspable, pred_grasp in zip(bat_index,bat_point,bat_pred_graspable,bat_pred_grasp):
                pos,R,width,depth,score = loss_utils.decode_pred(point, pred_graspable, pred_grasp)
                print(depth.max())
                print(depth.min())
                grasp = {
                    'pos':      pos.cpu().numpy(),
                    'R':        R.cpu().numpy(),
                    'depth':    depth.cpu().numpy(),
                    'score':    score.cpu().numpy(),
                    'width':    width.cpu().numpy(),
                }

                # for vis
                scene = trimesh.Scene()
                scene_mesh, _, _ = scene_utils.load_scene(int(img_id))
                scene.add_geometry(scene_mesh)
                grasp_group = np.concatenate([grasp['pos'],grasp['R'].reshape(-1,9),grasp['width'][:,np.newaxis],grasp['depth'][:,np.newaxis],grasp['score'][:,np.newaxis]],axis = -1)
                grippers = grasp_utils.bat_grasp_to_gripper(grasp_group)
                scene.add_geometry(grippers)
                scene.show()
                # np.save(f'output_grasps/{img_id}.npy',grasp)

def show():
    grasp_files = glob.glob('output_grasps/*.npy')
    for gf in grasp_files:
        img_id = gf.split('/')[-1].split('.')[0]
        scene_id = int(img_id)//cfg['num_images_per_scene']
        scene = trimesh.Scene()
        scene_mesh, _, _ = scene_utils.load_scene(int(img_id))
        scene.add_geometry(scene_mesh)
        grasp = np.load(gf,allow_pickle=True).item()
        pos = grasp['pos']
        R = grasp['R']
        width = grasp['width']
        depth = grasp['depth']
        score = grasp['score']

        grasp_group = np.concatenate([pos,R.reshape(-1,9),width[:,np.newaxis],depth[:,np.newaxis],score[:,np.newaxis]],axis = -1)
        grippers = grasp_utils.bat_grasp_to_gripper(grasp_group)
        scene.add_geometry(grippers)
        scene.show()

if __name__ == '__main__':
    from model import backbone_pointnet2
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    print('Using GPUs ' + os.environ["CUDA_VISIBLE_DEVICES"])
    dataset_path = '../point_grasp_data_2mm/'

    test_data = SimGraspDataset(dataset_path,split='test')
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                               batch_size = FLAGS.batchsize,
                                               shuffle=True,
                                               num_workers = FLAGS.workers)


    model = backbone_pointnet2().cuda()
    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(os.path.join(f'{FLAGS.model_path}/model_{str(FLAGS.epoch_use).zfill(3)}.pth')))
    save_grasp(test_dataloader,model)
    # show()
