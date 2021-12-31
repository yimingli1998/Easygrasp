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
from dataset_all import GraspNetDataset
from model import backbone_pointnet2
import yaml
from utils import scene_utils
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../')
with open(os.path.join(BASE_DIR,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser('Train SimGrasp')
parser.add_argument('--batchsize', type=int, default=cfg['train']['batchsize'], help='input batch size')
parser.add_argument('--workers', type=int, default=cfg['train']['workers'], help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=cfg['train']['epoches'], help='number of epochs for training')
parser.add_argument('--gpu', type=str, default=cfg['train']['gpu'], help='specify gpu device')
parser.add_argument('--learning_rate', type=float, default=cfg['train']['learning_rate'], help='learning rate for training')
parser.add_argument('--optimizer', type=str, default=cfg['train']['optimizer'], help='type of optimizer')
parser.add_argument('--theme', type=str, default=cfg['train']['theme'], help='type of train')
parser.add_argument('--model_path', type=str, default=cfg['eval']['model_path'], help='type of train')
FLAGS = parser.parse_args()

time_now = str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
LOG_DIR = os.path.join(BASE_DIR, 'experiment', time_now + '_' + FLAGS.theme)
# LOG_DIR = os.path.join(BASE_DIR, 'experiment', theme)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FOUT_test = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT_test.write(str(FLAGS) + '\n')
LOG_FOUT_train = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT_train.write(str(FLAGS) + '\n')
save_model_dir = os.path.join(LOG_DIR, 'checkpoints')

if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)

def log_string_test(out_str):
    LOG_FOUT_test.write(out_str + '\n')
    LOG_FOUT_test.flush()
    print(out_str)

def log_string_train(out_str):
    LOG_FOUT_train.write(out_str + '\n')
    LOG_FOUT_train.flush()
    print(out_str)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 10 epochs"""
    lr = max(FLAGS.learning_rate / (2 ** (epoch // 10)), 0.00001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(optimizer,num_epoches,train_dataloader,test_dataloader,model):
    model = model.train()
    for epoch in range(num_epoches):
        loss_list, acc_list = [],[]
        for i, (data,index) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            for k in data.keys():
                data[k] = data[k].cuda().float()
            # print(data['point'].shape,data['norm_point'].shape)
            pred = model(data['point'],torch.cat([data['norm_point'],data['color']],dim =-1).transpose(1, 2))
            # print(pred.shape)
            loss_dict,acc_dict = model.module.get_bin_loss(pred,data)
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()
            loss_list.append([v for v in loss_dict.values()])
            acc_list.append([v for v in acc_dict.values()])
            mean_loss = np.mean(loss_list,0)
            mean_acc = np.mean(acc_list,0)
            if (i%cfg['train']['printfreq'] == 0) or (i == len(train_dataloader)-1):
                print('epoch:%i\t iter:%.0f/%.0f\t gpts loss:%.3f\t azimuth loss:%.3f\t'
                      'elevation loss:%.3f\t grasp angle loss:%.3f\t width loss:%.3f\t'
                      'depth loss:%.3f\t score loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t' 
                      'TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t' 
                      'r:%.3f\t F1:%.3f\t '
                      %(epoch,i,len(train_dataloader),mean_loss[0].item(),mean_loss[1],
                       mean_loss[2],mean_loss[3],mean_loss[4],mean_loss[5],
                        mean_loss[6],mean_loss[7].item(),mean_acc[0],mean_acc[1],
                       mean_acc[2],mean_acc[3],mean_acc[4],mean_acc[5],
                       mean_acc[6],mean_acc[7]
                       ))
        log_string_train('epoch:%i\t iter:%.0f/%.0f\t gpts loss:%.3f\t azimuth loss:%.3f\t'
              'elevation loss:%.3f\t grasp angle loss:%.3f\t width loss:%.3f\t'
              'depth loss:%.3f\t score loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t'
              'TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t'
              'r:%.3f\t F1:%.3f\t '
              %(epoch,i,len(train_dataloader),mean_loss[0].item(),mean_loss[1],
                mean_loss[2],mean_loss[3],mean_loss[4],mean_loss[5],
                mean_loss[6],mean_loss[7].item(),mean_acc[0],mean_acc[1],
                mean_acc[2],mean_acc[3],mean_acc[4],mean_acc[5],
                mean_acc[6],mean_acc[7]
                ))
        torch.save(model.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'model', epoch))
        test(test_dataloader,model)

def test(test_dataloader,model):
    model = model.eval()
    with torch.no_grad():
        loss_list, acc_list = [],[]
        for i, (data,index) in enumerate(test_dataloader):
            for k in data.keys():
                data[k] = data[k].cuda().float()
            pred = model(data['point'],torch.cat([data['norm_point'],data['color']],dim =-1).transpose(1, 2))
            loss_dict,acc_dict = model.module.get_bin_loss(pred,data)
            loss_list.append([v for v in loss_dict.values()])
            acc_list.append([v for v in acc_dict.values()])
            mean_loss = np.mean(loss_list,0)
            mean_acc = np.mean(acc_list,0)
        log_string_test('test:\t gpts loss:%.3f\t azimuth loss:%.3f\t'
                         'elevation loss:%.3f\t grasp angle loss:%.3f\t width loss:%.3f\t'
                         'depth loss:%.3f\t score loss:%.3f\t total loss:%.3f\t TP:%.0d\t FP:%.0d\t'
                         'TN:%.0d\t FN:%.0d\t acc:%.3f\t p:%.3f\t'
                         'r:%.3f\t F1:%.3f\t '
                         %(mean_loss[0].item(),mean_loss[1],
                           mean_loss[2],mean_loss[3],mean_loss[4],mean_loss[5],
                           mean_loss[6],mean_loss[7].item(),mean_acc[0],mean_acc[1],
                           mean_acc[2],mean_acc[3],mean_acc[4],mean_acc[5],
                           mean_acc[6],mean_acc[7]
                           ))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    print('Using GPUs ' + os.environ["CUDA_VISIBLE_DEVICES"])
    train_simgrasp = True
    train_graspnet = True
    if train_simgrasp:
        dataset_path = '../point_grasp_data_2mm_sort_score/'
        simgrasp_train_data = SimGraspDataset(dataset_path,split='train')
#         train_dataloader = torch.utils.data.DataLoader(train_data,
#                                                    batch_size = FLAGS.batchsize,
#                                                    shuffle=True,
#                                                    num_workers = FLAGS.workers)
        
        simgrasp_test_data = SimGraspDataset(dataset_path,split='test')
#         test_dataloader = torch.utils.data.DataLoader(test_data,
#                                                    batch_size = FLAGS.batchsize,
#                                                    shuffle=True,
#                                                    num_workers = FLAGS.workers)
    if train_graspnet:
        dataset_path = '../'
        graspnet_train_data1 = GraspNetDataset(dataset_path,split='train')
        graspnet_train_data2 = GraspNetDataset(dataset_path,split='test_seen')
        graspnet_train_data3 = GraspNetDataset(dataset_path,split='test_novel')
#         train_dataloader = torch.utils.data.DataLoader(train_data,
#                                                        batch_size = FLAGS.batchsize,
#                                                        shuffle=True,
#                                                        num_workers = FLAGS.workers)
        graspnet_test_data = GraspNetDataset(dataset_path,split='test_similar')
#         test_dataloader = torch.utils.data.DataLoader(test_data,
#                                                    batch_size = FLAGS.batchsize,
#                                                    shuffle=True,
#                                                    num_workers = FLAGS.workers)
    merge_train_data = torch.utils.data.ConcatDataset([graspnet_train_data1,graspnet_train_data2,graspnet_train_data3])
    merge_test_data = torch.utils.data.ConcatDataset([graspnet_test_data])
    train_dataloader = torch.utils.data.DataLoader(merge_train_data,
                                                   batch_size = FLAGS.batchsize,
                                                   shuffle=True,
                                                   num_workers = FLAGS.workers)
    test_dataloader = torch.utils.data.DataLoader(merge_test_data,
                                                   batch_size = FLAGS.batchsize,
                                                   shuffle=True,
                                                   num_workers = FLAGS.workers)
    model = backbone_pointnet2().cuda()
    model = torch.nn.DataParallel(model)
#     model.load_state_dict(torch.load(os.path.join(f'{FLAGS.model_path}/model_245.pth')))
    optim_params = [
        {'params': model.parameters(),
         'lr': FLAGS.learning_rate,
         'betas': (0.9, 0.999),
         'eps': 1e-08},
    ]
    optimizer = optim.Adam(optim_params)
#     print(os.path.join(f'{FLAGS.model_path}/model_079.pth'))
#     model.load_state_dict(torch.load(os.path.join(f'{FLAGS.model_path}/model_079.pth')))
    train(optimizer,500,train_dataloader,test_dataloader,model)
#     model.load_state_dict(torch.load(os.path.join(f'{FLAGS.model_path}/model_079.pth')))
#     test(test_dataloader,model)
