import open3d
import cv2

#!/usr/bin/env python3
import os

import numpy as np
import pyrender
import matplotlib.pyplot as plt
import trimesh
import yaml
import cv2
import sys
from scipy.spatial.transform import Rotation as R
import imageio
from pathlib import Path
from skimage.io import imread_collection
import open3d as o3d

import struct

def normal_to_rgb(normals_to_convert, output_dtype='float'):
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) for a numpy image, or a range of (0,1) to represent PIL Image.

    The surface normals' axes are mapped as (x,y,z) -> (R,G,B).

    Args:
        normals_to_convert (numpy.ndarray): Surface normals, dtype float32, range [-1, 1]
        output_dtype (str): format of output, possibel values = ['float', 'uint8']
                            if 'float', range of output (0,1)
                            if 'uint8', range of output (0,255)
    '''
    camera_normal_rgb = (normals_to_convert + 1) / 2
    if output_dtype == 'uint8':
        camera_normal_rgb *= 255
        camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
    elif output_dtype == 'float':
        pass
    else:
        raise NotImplementedError('Possible values for "output_dtype" are only float and uint8. received value {}'.format(output_dtype))
    print(camera_normal_rgb.dtype)
    print(camera_normal_rgb.max())
    return camera_normal_rgb

def generate_normals(depth_imgs, scene_dir, save=False):
    normals = []
    for i, d_im in enumerate(depth_imgs):
        d_im = d_im.copy()

        normal = generate_normal(d_im)

        normals.append(normal)

        if save:
            if not os.path.exists(os.path.join(scene_dir, "normal")):
                os.makedirs(os.path.join(scene_dir, "normal"))
            imageio.imwrite(os.path.join(scene_dir, "normal/{}.png".format(f'{i + 1:05}')), normal)
    return normals

def generate_normal(d_im):

    normalizedImg = np.zeros((1280, 720))
    d_im = cv2.normalize(d_im, normalizedImg, 0, 1, cv2.NORM_MINMAX)
    d_im[d_im == 0] = np.nan
    d_im *= 255*100*10
    zy, zx = np.gradient(d_im)

    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= -n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    normal = normal_to_rgb(normal, output_dtype='uint8')
    print('hello')
    print(normal.dtype)
    print(normal.max())
    return normal

def get_scene_dirs(dataset_path):
    scenes_path = os.path.join(dataset_path, 'scenes')

    #scene_dirs = [os.path.join(scenes_path, file) for file in os.listdir(scenes_path) if
    #               os.path.isdir(os.path.join(scenes_path, file))]

    scene_dirs = next(os.walk(scenes_path))[1]
    scene_dirs = [os.path.join(scenes_path, scene_dir) for scene_dir in scene_dirs]

    return scene_dirs

def get_cam_intrinsics(scene_dir):
    with open(os.path.join(Path(scene_dir).parent.absolute(), "camera_d435.yaml")) as fd:
        intrinsics_dict = yaml.safe_load(fd)
    cam_matrix = np.array(intrinsics_dict['camera_matrix']).reshape((3, 3))
    img_width = intrinsics_dict['image_width']
    img_height = intrinsics_dict['image_height']

    #TODO EXPLAIN
    cam_matrix[0, 2] = cam_matrix[0, 2] #- 8
    cam_matrix[1, 2] = cam_matrix[1, 2] #+ 20
    return cam_matrix, img_width, img_height

def format_rgb(img, width, height):
    #unint8, flipped
    #img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.resize(img, (width, height))
    return img

def format_depth(img, width, height):
    #float64 to float32, flipped, pixel from mm to m
    #img = cv2.rotate(img, cv2.ROTATE_180)
    img = img * 0.001
    img = np.float32(img)
    img[np.isnan(img)] = 0
    img[np.isinf(img)] = 0

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    #TODO check why capping
    img[img < 0.1] = 0.0
    #img[img > 2.5] = 2.5

    #img[img == 0] = 1e-20

    return img

def format_mask(img, width, height):
    #rotate, rezise, create mask
    #img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = (img > 0)
    return img

def format_gt(img, width, height):
    #img = cv2.rotate(img, cv2.ROTATE_180)
    img = img * 0.001
    img = np.float32(img)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img[np.isnan(img)] = 0
    img[np.isinf(img)] = 0
    return img

def generate_pointcloud(scene_dir, depth_dir, rgb_dir, output_dir, width, height, backround_scene, normals_colouring=False):
    print(depth_dir)
    depths = imread_collection(depth_dir)
    if rgb_dir != None:
        rgbs = imread_collection(os.path.join(scene_dir, 'rgb/*.png'))
    rgbs = np.empty_like(depths)
    print(depths)
    camera_matrix, img_width, img_height = get_cam_intrinsics(scene_dir)
    scaling_factor = width/img_width
    fx = (camera_matrix[0][0] * scaling_factor)
    fy = (camera_matrix[1][1] * scaling_factor)
    cx = (camera_matrix[0][2] * scaling_factor)
    cy = (camera_matrix[1][2] * scaling_factor)
    i = 0
    for depth, rgb in zip(depths, rgbs):
        print(i)
        rgb = format_rgb(rgb, width, height)
        depth = format_depth(depth, width, height)

        if rgb_dir==None:
            rgb = generate_normals([depth*100], scene_dir, save=False)[0]
        
        cv2.imshow('img', rgb)
        cv2.waitKey(0)

        print(rgb.dtype)
        print(rgb.max())

        orgb = o3d.geometry.Image(rgb.astype(np.uint8))
        odepth = o3d.geometry.Image(depth.astype(np.float32))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            orgb, odepth, convert_rgb_to_intensity=False, depth_scale=1.0)

        intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr, project_valid_depth_only=False)

        o3d.visualization.draw_geometries([pcd])
        pcd_dir = os.path.join(scene_dir, "pcd/")

        if not os.path.exists(os.path.join(pcd_dir)):
            os.makedirs(pcd_dir)
        o3d.io.write_point_cloud(os.path.join(pcd_dir, "{}.pcd".format(f'{i + 1:05}')), pcd)
        i = i+1


if __name__ == '__main__':
    #get poses
    dataset_path = "/home/dalina/David/Uni/BachelorThesis/Tools/D435 dataset old/"

    #scene_dirs = get_scene_dirs(dataset_path)

    width=1280
    height=740

    #scene_dirs = scene_dirs[1]

    #print(scene_dirs)

    scene_dirs = ['/home/dalina/David/Uni/BachelorThesis/D435 dataset old/scenes/001_both_objects']
    
    for scene_dir in scene_dirs:
        depth_dir = os.path.join(scene_dir, 'gt_backround/*.png')
        
        rgb_dir = os.path.join(scene_dir, 'rgb/*.png')
        generate_pointcloud(scene_dir, depth_dir, None, 'pcd/', width, height, backround_scene=False)
