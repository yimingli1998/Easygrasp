import sys
import argparse
import os
import time
import pickle
import scipy.spatial
import torch
import math
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import functools
import yaml
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../')
with open(os.path.join(BASE_DIR,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

azimuth_scope = cfg['bin_loss']['azimuth_scope']
azimuth_bin_size = cfg['bin_loss']['azimuth_bin_size']
elevation_scope = cfg['bin_loss']['elevation_scope']
elevation_bin_size = cfg['bin_loss']['elevation_bin_size']
grasp_angle_scope = cfg['bin_loss']['grasp_angle_scope']
grasp_angle_bin_size = cfg['bin_loss']['grasp_angle_bin_size']
depth_scope = cfg['bin_loss']['depth_scope']
depth_bin_size = cfg['bin_loss']['depth_bin_size']
width_scope = cfg['bin_loss']['width_scope']
width_bin_size = cfg['bin_loss']['width_bin_size']
score_scope = cfg['bin_loss']['score_scope']
score_bin_size = cfg['bin_loss']['score_bin_size']

per_azimuth_bin_num = int(azimuth_scope / azimuth_bin_size)
per_elevation_bin_num = int(elevation_scope / elevation_bin_size)
per_grasp_angle_bin_num = int(grasp_angle_scope / grasp_angle_bin_size)
per_depth_bin_num = int(depth_scope / depth_bin_size)
per_width_bin_num = int(width_scope / width_bin_size)
per_score_bin_num = int(score_scope / score_bin_size)

def get_bin_reg_loss(pred_grasp, gt_grasp):
    loss_dict = {}
    loc_loss = 0
    start_offset = 0

    azimuth_angle,elevation_angle,grasp_angle,width,depth,score = gt_grasp[:,0],gt_grasp[:,1],gt_grasp[:,2],gt_grasp[:,3],gt_grasp[:,4],gt_grasp[:,5]

    azimuth_bin_l, azimuth_bin_r = start_offset, start_offset + per_azimuth_bin_num
    azimuth_res_l, azimuth_res_r = azimuth_bin_r, azimuth_bin_r + per_azimuth_bin_num
    start_offset = azimuth_res_r

    azimuth_shift = torch.clamp(azimuth_angle, 0, azimuth_scope - 1e-4)
    azimuth_bin_label = (azimuth_shift / azimuth_bin_size).floor().long()
    #azimuth_res_label = azimuth_shift - (azimuth_bin_label.float() * azimuth_bin_size + azimuth_bin_size / 2)
    azimuth_res_label = azimuth_shift - (azimuth_bin_label.float() * azimuth_bin_size)
    azimuth_res_norm_label = azimuth_res_label / azimuth_bin_size

    azimuth_bin_onehot = torch.cuda.FloatTensor(azimuth_bin_label.size(0), per_azimuth_bin_num).zero_()
    azimuth_bin_onehot.scatter_(1, azimuth_bin_label.view(-1, 1).long(), 1)

    loss_azimuth_bin = F.cross_entropy(pred_grasp[:, azimuth_bin_l: azimuth_bin_r], azimuth_bin_label)
    loss_azimuth_res = F.smooth_l1_loss((torch.sigmoid(pred_grasp[:, azimuth_res_l: azimuth_res_r]) * azimuth_bin_onehot).sum(dim=1), azimuth_res_norm_label)
    loss_dict['azimuth_bin_loss'] = loss_azimuth_bin.item()
    loss_dict['azimuth_res_loss'] = loss_azimuth_res.item()
    loc_loss += loss_azimuth_bin + loss_azimuth_res

    elevation_bin_l, elevation_bin_r = start_offset, start_offset + per_elevation_bin_num
    elevation_res_l, elevation_res_r = elevation_bin_r, elevation_bin_r + per_elevation_bin_num
    start_offset = elevation_res_r

    elevation_shift = torch.clamp(elevation_angle, 0, elevation_scope - 1e-4)
    elevation_bin_label = (elevation_shift / elevation_bin_size).floor().long()
    #elevation_res_label = elevation_shift - (elevation_bin_label.float() * elevation_bin_size + elevation_bin_size / 2)
    elevation_res_label = elevation_shift - (elevation_bin_label.float() * elevation_bin_size)
    elevation_res_norm_label = elevation_res_label / elevation_bin_size

    elevation_bin_onehot = torch.cuda.FloatTensor(elevation_bin_label.size(0), per_elevation_bin_num).zero_()
    elevation_bin_onehot.scatter_(1, elevation_bin_label.view(-1, 1).long(), 1)

    loss_elevation_bin = F.cross_entropy(pred_grasp[:, elevation_bin_l: elevation_bin_r], elevation_bin_label)
    loss_elevation_res = F.smooth_l1_loss((torch.sigmoid(pred_grasp[:, elevation_res_l: elevation_res_r]) * elevation_bin_onehot).sum(dim=1), elevation_res_norm_label)
    loss_dict['elevation_bin_loss'] = loss_elevation_bin.item()
    loss_dict['elevation_res_loss'] = loss_elevation_res.item()

    loc_loss += loss_elevation_bin + loss_elevation_res

    grasp_angle_bin_l, grasp_angle_bin_r = start_offset, start_offset + per_grasp_angle_bin_num
    grasp_angle_res_l, grasp_angle_res_r = grasp_angle_bin_r, grasp_angle_bin_r + per_grasp_angle_bin_num
    start_offset = grasp_angle_res_r
    # print('start_offset',start_offset)
    grasp_angle_shift = torch.clamp(grasp_angle, 0, grasp_angle_scope - 1e-4)
    grasp_angle_bin_label = (grasp_angle_shift / grasp_angle_bin_size).floor().long()
    #grasp_angle_res_label = grasp_angle_shift - (grasp_angle_bin_label.float() * grasp_angle_bin_size + grasp_angle_bin_size / 2)
    grasp_angle_res_label = grasp_angle_shift - (grasp_angle_bin_label.float() * grasp_angle_bin_size)
    grasp_angle_res_norm_label = grasp_angle_res_label / grasp_angle_bin_size

    grasp_angle_bin_onehot = torch.cuda.FloatTensor(grasp_angle_bin_label.size(0), per_grasp_angle_bin_num).zero_()
    grasp_angle_bin_onehot.scatter_(1, grasp_angle_bin_label.view(-1, 1).long(), 1)

    loss_grasp_angle_bin = F.cross_entropy(pred_grasp[:, grasp_angle_bin_l: grasp_angle_bin_r], grasp_angle_bin_label)
    loss_grasp_angle_res = F.smooth_l1_loss((torch.sigmoid(pred_grasp[:, grasp_angle_res_l: grasp_angle_res_r]) * grasp_angle_bin_onehot).sum(dim=1), grasp_angle_res_norm_label)
    loss_dict['grasp_angle_bin_loss'] = loss_grasp_angle_bin.item()
    loss_dict['grasp_angle_res_loss'] = loss_grasp_angle_res.item()
    loc_loss += loss_grasp_angle_bin + loss_grasp_angle_res

    width_bin_l, width_bin_r = start_offset, start_offset + per_width_bin_num
    width_res_l, width_res_r = width_bin_r, width_bin_r + per_width_bin_num
    start_offset = width_res_r

    width_shift = torch.clamp(width, 0, width_scope - 1e-4)
    width_bin_label = (width_shift / width_bin_size).floor().long()
    #width_res_label = width_shift - (width_bin_label.float() * width_bin_size + width_bin_size / 2)
    width_res_label = width_shift - (width_bin_label.float() * width_bin_size)
    width_res_norm_label = width_res_label / width_bin_size

    width_bin_onehot = torch.cuda.FloatTensor(width_bin_label.size(0), per_width_bin_num).zero_()
    width_bin_onehot.scatter_(1, width_bin_label.view(-1, 1).long(), 1)

    loss_width_bin = F.cross_entropy(pred_grasp[:, width_bin_l: width_bin_r], width_bin_label)
    loss_width_res = F.smooth_l1_loss((torch.sigmoid(pred_grasp[:, width_res_l: width_res_r]) * width_bin_onehot).sum(dim=1), width_res_norm_label)
    loss_dict['width_bin_loss'] = loss_width_bin.item()
    loss_dict['width_res_loss'] = loss_width_res.item()
    loc_loss += loss_width_bin + loss_width_res

    depth_bin_l, depth_bin_r = start_offset, start_offset + per_depth_bin_num
    depth_res_l, depth_res_r = depth_bin_r, depth_bin_r + per_depth_bin_num
    start_offset = depth_res_r
    
    depth_shift = torch.clamp(depth, 0, depth_scope - 1e-4)
    depth_bin_label = (depth_shift / depth_bin_size).floor().long()
    #depth_res_label = depth_shift - (depth_bin_label.float() * depth_bin_size + depth_bin_size / 2)
    depth_res_label = depth_shift - (depth_bin_label.float() * depth_bin_size)
    depth_res_norm_label = depth_res_label / depth_bin_size
    
    depth_bin_onehot = torch.cuda.FloatTensor(depth_bin_label.size(0), per_depth_bin_num).zero_()
    depth_bin_onehot.scatter_(1, depth_bin_label.view(-1, 1).long(), 1)
    
    loss_depth_bin = F.cross_entropy(pred_grasp[:, depth_bin_l: depth_bin_r], depth_bin_label)
    loss_depth_res = F.smooth_l1_loss((torch.sigmoid(pred_grasp[:, depth_res_l: depth_res_r]) * depth_bin_onehot).sum(dim=1), depth_res_norm_label)
    loss_dict['depth_bin_loss'] = loss_depth_bin.item()
    loss_dict['depth_res_loss'] = loss_depth_res.item()
#     loc_loss += loss_depth_bin + loss_depth_res
    loc_loss += loss_depth_bin
    
    score_bin_l, score_bin_r = start_offset, start_offset + per_score_bin_num
    score_res_l, score_res_r = score_bin_r, score_bin_r + per_score_bin_num
    start_offset = score_res_r
#     print(pred_grasp[:, -1].shape,score.shape)
    score_loss = F.smooth_l1_loss(pred_grasp[:, -1], score)
#     print(score_loss)
#     score_shift = torch.clamp(score, 0, score_scope - 1e-4)
#     score_bin_label = (score_shift / score_bin_size).floor().long()
#     #score_res_label = score_shift - (score_bin_label.float() * score_bin_size + score_bin_size / 2)
#     score_res_label = score_shift - (score_bin_label.float() * score_bin_size)
#     score_res_norm_label = score_res_label / score_bin_size
#     # print(score_res_norm_label,score_bin_label)
#     score_bin_onehot = torch.cuda.FloatTensor(score_bin_label.size(0), per_score_bin_num).zero_()
#     score_bin_onehot.scatter_(1, score_bin_label.view(-1, 1).long(), 1)
#     loss_score_bin = F.cross_entropy(pred_grasp[:, score_bin_l: score_bin_r], score_bin_label)
#     loss_score_res = F.smooth_l1_loss((torch.sigmoid(pred_grasp[:, score_res_l: score_res_r]) * score_bin_onehot).sum(dim=1), score_res_norm_label)
#     loss_dict['score_bin_loss'] = loss_score_bin.item()
#     loss_dict['score_res_loss'] = loss_score_res.item()
#     loc_loss += loss_score_bin + loss_score_res
    loss_dict['score_loss'] = score_loss
    loc_loss += score_loss 
    loss_dict['loss_loc'] = loc_loss

    return loss_dict,loc_loss

def angle_to_vector(azimuth_angle, elevation_angle):
    device = azimuth_angle.device
    x_ = torch.cos(azimuth_angle/180*math.pi)
    y_ = torch.sin(azimuth_angle/180*math.pi)
    z_ = -torch.tan((elevation_angle)/180 * math.pi) * torch.sqrt(x_**2 + y_**2)
    approach_vector_ = torch.stack((x_, y_, z_), axis=0)
    approach_vector = torch.div(approach_vector_, torch.norm(approach_vector_, dim=0))
    return approach_vector

def grasp_angle_to_vector(grasp_angle):
    device = grasp_angle.device
    x_ = torch.cos(grasp_angle/180*math.pi)
    y_ = torch.sin(grasp_angle/180*math.pi)
    z_ = torch.zeros(grasp_angle.shape[0]).to(device)
    x_, y_, z_ = x_.double(), y_.double(), z_.double()
    closing_vector = torch.stack((x_, y_, z_), axis=0)

    return closing_vector

def rotation_from_vector(approach_vector, closing_vector):
    #TODO: if element in approach_vector[:, 2] = 0 cause error !
    temp = -(approach_vector[:, 0] * closing_vector[:, 0] + approach_vector[:, 1] * closing_vector[:, 1]) / approach_vector[:, 2]
    closing_vector[:, 2] = temp
    closing_vector = torch.div(closing_vector.transpose(0, 1), torch.norm(closing_vector, dim=1)).transpose(0, 1)
    z_axis = torch.cross(approach_vector.float(), closing_vector.float(), dim = 1)
    R = torch.stack((approach_vector, closing_vector.float(), z_axis.float()), dim=-1)
    return R


def decode_pred(point,pred_gp, pred_grasp):
    # print(pred_gp.size())
    out_gp = torch.argmax(pred_gp,dim = 1).bool()
    # print(out_gp)
    # print(torch.sum(out_gp==0))
    # print(torch.sum(out_gp==1))
    if torch.sum(out_gp) <= 0:
        return
    pred_grasp = pred_grasp[out_gp]

    start_offset = 0

    azimuth_bin_l, azimuth_bin_r = start_offset, start_offset + per_azimuth_bin_num
    azimuth_res_l, azimuth_res_r = azimuth_bin_r, azimuth_bin_r + per_azimuth_bin_num
    start_offset = azimuth_res_r
    azimuth_bin = torch.argmax(pred_grasp[:, azimuth_bin_l: azimuth_bin_r], dim=1)
    azimuth_res_norm = torch.gather(pred_grasp[:, azimuth_res_l: azimuth_res_r], dim=1, index=azimuth_bin.unsqueeze(dim=1)).squeeze(dim=1)
    azimuth_res = azimuth_res_norm * azimuth_bin_size
    azimuth_angle = azimuth_bin.float() * azimuth_bin_size + azimuth_bin_size / 2 + azimuth_res

    elevation_bin_l, elevation_bin_r = start_offset, start_offset + per_elevation_bin_num
    elevation_res_l, elevation_res_r = elevation_bin_r, elevation_bin_r + per_elevation_bin_num
    start_offset = elevation_res_r
    elevation_bin = torch.argmax(pred_grasp[:, elevation_bin_l: elevation_bin_r], dim=1)
    elevation_res_norm = torch.gather(pred_grasp[:, elevation_res_l: elevation_res_r], dim=1, index=elevation_bin.unsqueeze(dim=1)).squeeze(dim=1)
    elevation_res = elevation_res_norm * elevation_bin_size
    elevation_angle = elevation_bin.float() * elevation_bin_size + elevation_bin_size / 2 + elevation_res
    elevation_angle = elevation_angle - 90
    approach_vector = angle_to_vector(azimuth_angle, elevation_angle).transpose(0, 1) #vector B*N, 3
    #print(approach_vector[0])

    grasp_angle_bin_l, grasp_angle_bin_r = start_offset, start_offset + per_grasp_angle_bin_num
    grasp_angle_res_l, grasp_angle_res_r = grasp_angle_bin_r, grasp_angle_bin_r + per_grasp_angle_bin_num
    start_offset = grasp_angle_res_r
    grasp_angle_bin = torch.argmax(pred_grasp[:, grasp_angle_bin_l: grasp_angle_bin_r], dim=1)
    grasp_angle_res_norm = torch.gather(pred_grasp[:, grasp_angle_res_l: grasp_angle_res_r], dim=1, index=grasp_angle_bin.unsqueeze(dim=1)).squeeze(dim=1)
    grasp_angle_res = grasp_angle_res_norm * grasp_angle_bin_size
    grasp_angle = grasp_angle_bin.float() * grasp_angle_bin_size + grasp_angle_bin_size / 2 + grasp_angle_res

    closing_vector = grasp_angle_to_vector(grasp_angle).transpose(0, 1)
    R = rotation_from_vector(approach_vector, closing_vector)
    #print(R[0])
    
    width_bin_l, width_bin_r = start_offset, start_offset + per_width_bin_num
    width_res_l, width_res_r = width_bin_r, width_bin_r + per_width_bin_num
    start_offset = width_res_r
    width_bin = torch.argmax(pred_grasp[:, width_bin_l: width_bin_r], dim=1)
    width_res_norm = torch.gather(pred_grasp[:, width_res_l: width_res_r], dim=1, index=width_bin.unsqueeze(dim=1)).squeeze(dim=1)
    width_res = width_res_norm * width_bin_size
    width = width_bin.float() * width_bin_size + width_bin_size / 2 + width_res

    depth_bin_l, depth_bin_r = start_offset, start_offset + per_depth_bin_num
    depth_res_l, depth_res_r = depth_bin_r, depth_bin_r + per_depth_bin_num
    start_offset = depth_res_r
    depth_bin = torch.argmax(pred_grasp[:, depth_bin_l: depth_bin_r], dim=1)
    depth_res_norm = torch.gather(pred_grasp[:, depth_res_l: depth_res_r], dim=1, index=depth_bin.unsqueeze(dim=1)).squeeze(dim=1)
    depth_res = depth_res_norm * depth_bin_size
    # depth = depth_bin.float() * depth_bin_size + depth_bin_size / 2 + depth_res
    depth = depth_bin.float() * depth_bin_size

    # score_bin_l, score_bin_r = start_offset, start_offset + per_score_bin_num
    # score_res_l, score_res_r = score_bin_r, score_bin_r + per_score_bin_num
    # start_offset = score_res_r
    # score_bin = torch.argmax(pred_grasp[:, score_bin_l: score_bin_r], dim=1)
    # score_res_norm = torch.gather(pred_grasp[:, score_res_l: score_res_r], dim=1, index=score_bin.unsqueeze(dim=1)).squeeze(dim=1)
    # score_res = score_res_norm * score_bin_size
    # score = score_bin.float() * score_bin_size + score_bin_size / 2 + score_res
    # score = score_bin.float() * score_bin_size


    approach = R[:,:3,0]
    gp = point[out_gp]
    # print(approach.shape,gp.shape,depth.shape)
    width = width /100.
    depth = (depth + 1)/100.
    # score = (score + 1)/10.
    score = pred_grasp[:,-1]/10.
    return gp,R,width,depth,score



