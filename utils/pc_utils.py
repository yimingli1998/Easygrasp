import numpy as np
from PIL import Image
import os
import yaml

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../')
with open(os.path.join(BASE_DIR,'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)


def depth_to_pointcloud(depth_file,rgb_file,intrinsics,add_noise =False):

    depth = np.array(Image.open(depth_file))*0.001*intrinsics['depth_scale']
    rgb = np.array(Image.open(rgb_file))
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    s = 1.0

    xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    if add_noise:
        depth += np.random.normal(0,cfg['noise']['sigma'],size =(depth.shape[0],depth.shape[1]))

    points_z = depth / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = points_z > 0
    points_x = points_x[mask]
    points_y = points_y[mask]
    points_z = points_z[mask]
    rgb = rgb[mask]
    points = np.stack([points_x, points_y, points_z], axis=-1)
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(points)
    # cloud.colors = o3d.utility.Vector3dVector(colors)
    # print(points.shape)
    return points,rgb

def crop_point(point,rgb = None):
    val_x = (point[:, 0] > -0.5) & (point[:, 0] < 0.5)
    val_y = (point[:, 1] > -0.5) & (point[:, 1] < 0.5)
    val_z = (point[:, 2] > -0.5) & (point[:, 2] < 0.5)
    val = val_x * val_y * val_z
    if rgb is not None:
        return point[val],rgb[val],val
    return point[val],val
