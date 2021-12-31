import numpy as np
import trimesh
import os
import yaml
from tqdm import tqdm

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../')

with open(os.path.join(BASE_DIR,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)


def filter_origin_grasps(output_path):
    object_ids = np.load(os.path.join(BASE_DIR,'object_idx.npy'),allow_pickle=True).item()
    train_objects = list(object_ids['train'].tolist())
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)
    for obj_id, grasp_obj_id in enumerate(tqdm(train_objects)):
        filtered_grasp = {}
        grasp = np.load(os.path.join(BASE_DIR,f'grasp_label/{str(grasp_obj_id).zfill(3)}_labels.npz'),allow_pickle =True)
        points = grasp['points']
        offsets = grasp['offsets']
        score = grasp['scores']
        collision = grasp['collision']
        angle = offsets[:,:,:,:,0]
        depth = offsets[:,:,:,:,1]
        width = offsets[:,:,:,:,2]

        template_views = generate_views(300)
        views = np.tile(template_views[ :, np.newaxis, np.newaxis, :],[1,12,4,1])
        # Rs = grasp_utils.batch_viewpoint_params_to_matrix(-views,angle)
        for i,p in enumerate(points):
            s,a,d,w,c = score[i],angle[i],depth[i],width[i],collision[i]
            mask = (w < cfg['filter']['width_thresh']) & (s > 0) & (s <cfg['filter']['score_thresh']) & (c==False)
            s,a,d,w,c = s[mask],a[mask],d[mask],w[mask],c[mask]
            v = views[mask]

            if len(v)==0:
                filtered_grasp[i] = {'point':p}
            else:
                R = batch_viewpoint_params_to_matrix(-v,a)
                filtered_grasp[i] = {
                    'point':p,
                    'R':    R,
                    'd':    d,
                    'w':    w,
                    's':    s,
                }
        np.save(os.path.join(output_path,f'{str(obj_id).zfill(6)}.npy'),filtered_grasp)


def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3, dtype=np.float32), R=1):
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X,Y,Z], axis=1)
    views = R * np.array(views) + center
    return views

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box = trimesh.Trimesh(vertices,triangles)
    return box

def plot_gripper_pro_max(center, R, width, depth, score=1):
    '''
        center: target point
        R: rotation matrix
    '''
    x, y, z = center
    height=0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    color_r = score # red for high score
    color_b = 1 - score # blue for low score
    color_g = 0
    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.faces)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.faces) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.faces) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.faces) + 24
    tail_points[:,0] -= tail_length + finger_width + depth_base
    tail_points[:,1] -= finger_width / 2
    tail_points[:,2] -= height/2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)

    gripper = trimesh.Trimesh(vertices,triangles)
    gripper.visual.face_colors = [color_r,color_g,color_b,1]
    return gripper

def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    axis_x = batch_towards
    ones = np.ones(axis_x.shape[0], dtype=axis_x.dtype)
    zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
    axis_y = np.stack([-axis_x[:,1], axis_x[:,0], zeros], axis=-1)
    axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)
    sin = np.sin(batch_angle)
    cos = np.cos(batch_angle)
    R1 = np.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], axis=-1)
    R1 = R1.reshape([-1,3,3])
    R2 = np.stack([axis_x, axis_y, axis_z], axis=-1)
    matrix = np.matmul(R2, R1)
    return matrix.astype(np.float32)

def grasp_to_gripper(p, R, s, d, w, n_grasp = 3):
    choice = np.random.choice(len(R),n_grasp,replace=True)
    R,s,d,w = R[choice],s[choice],d[choice],w[choice]
    grippers = []
    # print(p.shape,R.shape,s.shape,d.shape,w.shape)
    for i in range(len(R)):
        gripper = plot_gripper_pro_max(p,R[i],w[i],d[i],1.1-s[i])
        grippers.append(gripper)
    grippers = np.sum(grippers)

    return grippers

def bat_grasp_to_gripper(grasp, n_grasp = 100):
    choice = np.random.choice(len(grasp),n_grasp,replace=True)
    grasp = grasp[choice]
    grippers = []
    # print(p.shape,R.shape,s.shape,d.shape,w.shape)
    for i in range(len(grasp)):
        gripper = plot_gripper_pro_max(grasp[i,:3],grasp[i,3:12].reshape(3,3),grasp[i,12],grasp[i,13],1.1-grasp[i,14])
        grippers.append(gripper)
    grippers = np.sum(grippers)

    return grippers

