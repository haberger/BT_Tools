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

    return camera_normal_rgb

def generate_normals(depth_imgs, scene_dir, save=True):
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

    normalizedImg = np.zeros((640, 360))
    d_im = cv2.normalize(d_im, normalizedImg, 0, 1, cv2.NORM_MINMAX)
    d_im[d_im == 0] = np.nan
    d_im *= 255
    zy, zx = np.gradient(d_im)

    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= -n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    normal = normal_to_rgb(normal, output_dtype='uint8')

    return normal

def convert_to_gl_coordinates(pose):
    T = np.eye(4)
    T[1, 1] *= -1
    T[2, 2] *= -1
    #print(T)
    pose = pose@T
    return pose

def flip_axis(pose, axis):
    T = np.eye(4)
    T[axis, axis] *= -1
    #print(T)
    pose = pose@T
    return pose

def plot_rendering(color, depth):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(color)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(depth, cmap=plt.cm.gray_r)
    plt.show()

def visualise_cam_positions(scene, poses):
    canister_trimesh = trimesh.load_mesh('model/Canister.stl')
    mesh = pyrender.Mesh.from_trimesh(canister_trimesh, poses=poses[0,:,:])
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)

def eval_annotation(depth_imgs, scene_dir):
    if depth_imgs==None:
        if not os.path.exists(os.path.join(scene_dir, "annotation")):
            return
        for i in range(0,64):
            mask = imageio.v2.imread(os.path.join(scene_dir, 'annotation/{}.png'.format(f'anno_{i + 1}')))
            rgb = imageio.v2.imread(os.path.join(scene_dir, 'rgb/{}.png'.format(f'{i + 1:05}')))
            mask[mask>0]=255
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            add = rgb.copy()
            add = np.array(add).astype(np.uint8)
            for c in cnts:
                cv2.drawContours(add, [c], -1, (36, 255, 12), thickness=2)        
            #add = cv2.addWeighted(cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB), 0.3, rgb, 0.5, 0.0)
            if not os.path.exists(os.path.join(scene_dir, "annotation_old")):
                os.makedirs(os.path.join(scene_dir, "annotation_old"))
            imageio.imwrite(os.path.join(scene_dir, "annotation_old/{}.png".format(f'{i + 1:05}')), add)        
        return

    for i, depth in enumerate(depth_imgs):
        depth = depth.copy()
        rgb = imageio.v2.imread(os.path.join(scene_dir, 'rgb/{}.png'.format(f'{i + 1:05}')))
        depth = (depth * 255).astype(np.uint8)
        mask = depth
        mask[mask>0]=255
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        add = rgb.copy()
        add = np.array(add).astype(np.uint8)
        for c in cnts:
            cv2.drawContours(add, [c], -1, (36, 255, 12), thickness=2)        
        #add = cv2.addWeighted(cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB), 0.3, rgb, 0.5, 0.0)
        if not os.path.exists(os.path.join(scene_dir, "annotation")):
            os.makedirs(os.path.join(scene_dir, "annotation"))
        imageio.imwrite(os.path.join(scene_dir, "annotation/{}.png".format(f'{i + 1:05}')), add)

def generate_mask(depth_imgs, scene_dir):
    masks = []
    for i, depth in enumerate(depth_imgs):
        depth = depth.copy()
        depth[depth > 0] = 255
        mask = depth.astype(np.uint8)

        if not os.path.exists(os.path.join(scene_dir, "mask")):
            os.makedirs(os.path.join(scene_dir, "mask"))
        imageio.imwrite(os.path.join(scene_dir, "mask/{}.png".format(f'{i + 1:05}')), mask)
        masks.append(mask)
    return masks

def get_scene_dirs(dataset_path):
    scenes_path = os.path.join(dataset_path, 'scenes')

    #scene_dirs = [os.path.join(scenes_path, file) for file in os.listdir(scenes_path) if
    #               os.path.isdir(os.path.join(scenes_path, file))]

    scene_dirs = next(os.walk(scenes_path))[1]
    scene_dirs = [os.path.join(scenes_path, scene_dir) for scene_dir in scene_dirs]

    return scene_dirs

def get_model_poses(file_path):
    with open(file_path) as fd:
        config_yaml = yaml.safe_load(fd)

    poses = []
    ids = []

    for dict in config_yaml:
        poses.append(np.array(dict['pose']).reshape((4, 4)))
        ids.append(dict['id'])
    return np.array(poses), np.array(ids)

def get_camera_pose(file_path):
    print(file_path)
    data = np.loadtxt(file_path)
    poses = np.zeros((data.shape[0], 4, 4))

    for i in range(data.shape[0]):
        quat = data[i][4:8]
        r = R.from_quat(quat)
        r = r.as_matrix()
        pose = np.array([[r[0, 0], r[0, 1], r[0, 2], data[i, 1]],
                         [r[1, 0], r[1, 1], r[1, 2], data[i, 2]],
                         [r[2, 0], r[2, 1], r[2, 2], data[i, 3]],
                         [0, 0, 0, 1]])
        poses[i, :, :] = pose

    poses_gl = convert_to_gl_coordinates(poses)
    return poses, poses_gl

def get_cam_intrinsics(scene_dir):
    with open(os.path.join(Path(scene_dir).parent.absolute(), "camera_d435.yaml")) as fd:
        intrinsics_dict = yaml.safe_load(fd)
    cam_matrix = np.array(intrinsics_dict['camera_matrix']).reshape((3, 3))
    img_width = intrinsics_dict['image_width']
    img_height = intrinsics_dict['image_height']
    return cam_matrix, img_width, img_height

def calculate_cam_to_model(cam_poses, model_poses, scene_dir):

    transformations = []
    for m_pose in model_poses:
        transformations.append(np.linalg.inv(cam_poses) @ m_pose)

    for t in transformations:
        t = t.reshape(t.shape[0], -1)
    transformations = zip(*transformations)

    with open(os.path.join(scene_dir, "cam_to_model_poses.txt"), "w") as file:
        for transformation in transformations:
            for t in transformation:
                np.savetxt(file, t.ravel(), fmt='%.18e', newline=" ")
                file.write("\n")

def get_mesh(id, dataset_path):
    objects_path = os.path.join(dataset_path, "objects")
    with open(os.path.join(objects_path, 'objects_bop.yaml')) as fd:
        config_yaml = yaml.safe_load(fd)
        mesh_info = next(item for item in config_yaml if item["id"] == id)
        mesh_path = os.path.join(objects_path, mesh_info["mesh"])
        mesh = trimesh.load_mesh(mesh_path)
        return mesh
    
def generate_gt_depth(scene_dir, poses='poses.yaml', gt_handeye='groundtruth_handeye.txt'):
    model_poses, ids = get_model_poses(os.path.join(scene_dir, poses))
    cam_poses, cam_poses_gl = get_camera_pose(os.path.join(scene_dir, gt_handeye))
    camera_matrix, img_width, img_height = get_cam_intrinsics(scene_dir)

    calculate_cam_to_model(cam_poses, model_poses, scene_dir)

    scene = pyrender.Scene(bg_color=[0.9, 0.9, 0.9])

    for model_pose, id in zip(model_poses, ids):
        t_mesh = get_mesh(id, dataset_path)
        mesh = pyrender.Mesh.from_trimesh(t_mesh, poses=model_pose)
        scene.add(mesh)

    camera = pyrender.IntrinsicsCamera(fx=camera_matrix[0][0],
                                       fy=camera_matrix[1][1],
                                       cx=camera_matrix[0][2],
                                       cy=camera_matrix[1][2])

    nc = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(nc)

    depth_imgs = []
    r = pyrender.OffscreenRenderer(img_width, img_height)
    for i, cam_pose_gl in enumerate(cam_poses_gl):
        scene.set_pose(nc, pose=cam_pose_gl)
        color, depth = r.render(scene)
        depth_imgs.append(depth)
        if not os.path.exists(os.path.join(scene_dir, "gt")):
            os.makedirs(os.path.join(scene_dir, "gt"))
        imageio.imwrite(os.path.join(scene_dir, "gt/{}.png".format(f'{i + 1:05}')), (depth * 1000).astype(np.uint16))

    return depth_imgs

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

def generate_pointcloud(scene_dir, depth_path, rgb_path, width, height, output_dir, normals_colouring=False):
    depths = imread_collection(depth_path)
    rgbs = imread_collection(rgb_path)
    #masks = imread_collection(os.path.join(scene_dir, 'mask/*.png'))
    #gts = imread_collection(os.path.join(scene_dir, 'gt/*.png'))

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
        #gt = format_gt(gt, width, height)
        #mask = format_mask(mask, width, height)

        # if backround_scene:
        #     depth[mask] = gt[mask]
        # else:
        #     depth = gt

        if normals_colouring:
            rgb = generate_normal(depth*1)

        orgb = o3d.geometry.Image(rgb.astype(np.uint8))
        odepth = o3d.geometry.Image(depth.astype(np.float32))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            orgb, odepth, convert_rgb_to_intensity=False, depth_scale=1.0)

        intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr, project_valid_depth_only=False)

        #TODO transform needed?
        #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        #TODO does it work with ppf?????

        # if backround_scene:
            # if normals_colouring:
            #     pcd_dir = os.path.join(scene_dir, "pcd/pcd_with_background_normals")
            # else:
            #     pcd_dir = os.path.join(scene_dir, "pcd/pcd_with_background")
        # else:
        #     if normals_colouring:
        #         pcd_dir = os.path.join(scene_dir, "pcd/pcd_model_only_normals")
        #     else:
        #         pcd_dir = os.path.join(scene_dir, "pcd/pcd_model_only")

        if not os.path.exists(os.path.join(output_dir)):
            os.makedirs(output_dir)
        o3d.io.write_point_cloud(os.path.join(output_dir, "{}.pcd".format(f'{i + 1:05}')), pcd)
        i = i+1

def generate_gt_with_background(scene_dir, width, height):
    depths = imread_collection(os.path.join(scene_dir, 'depth/*.png'))
    gts = imread_collection(os.path.join(scene_dir, 'gt/*.png'))
    masks = imread_collection(os.path.join(scene_dir, 'mask/*.png'))
    i = 0
    for depth, mask, gt in zip(depths,masks, gts):
        print(i)

        depth = format_depth(depth, width, height)
        gt = format_gt(gt, width, height)
        mask = format_mask(mask, width, height)

        depth[mask] = gt[mask]

        dir = os.path.join(scene_dir, "gt_background")

        if not os.path.exists(os.path.join(dir)):
            os.makedirs(dir)
        imageio.imwrite(os.path.join(dir, "{}.png".format(f'{i + 1:05}')), (depth * 1000).astype(np.uint16))
        i = i+1

if __name__ == '__main__':
    #get poses
    dataset_path = "/home/dalina/David/Uni/BachelorThesis/D435 dataset old"

    scene_dirs = get_scene_dirs(dataset_path)

    width=1280
    height=720

    #scene_dirs = scene_dirs[1]

    print(scene_dirs)

    #scene_dirs = ['/home/dalina/David/Uni/BachelorThesis/Tools/D435 dataset old/scenes/002/']

    for scene_dir in scene_dirs:
        #scene_dir = '/home/david/UNI/BachelorThesis/implementation/D435 dataset/scenes/014b'
        print(scene_dir)

        depth_imgs = generate_gt_depth(scene_dir, poses='poses.yaml')

        #for depth in depth_imgs:
        #    cv2.imshow('img', depth)
        #    cv2.waitKey(0)

        masks = generate_mask(depth_imgs, scene_dir)

        generate_gt_with_background(scene_dir, width, height)

        # #plot_rendering(color, depth)

        generate_normals(depth_imgs, scene_dir, save=True)
        eval_annotation(depth_imgs, scene_dir)
        try:
            eval_annotation(None, scene_dir)
        except:
            pass
        #masks = generate_mask(depth_imgs, scene_dir)

        
        #depth_path = os.path.join(scene_dir, 'depth/*.png')
        #rgb_path = os.path.join(scene_dir, 'rgb/*.png')
        #output_path = os.path.join(scene_dir, 'input_pcd/')
        #generate_pointcloud(scene_dir, depth_path, rgb_path, width, height, output_path, normals_colouring=True)
        #generate_pointcloud(scene_dir, width, height, backround_scene=True)
        #generate_pointcloud(scene_dir, width, height, backround_scene=False, normals_colouring=True)
        #generate_pointcloud(scene_dir, width, height, backround_scene=True, normals_colouring=True)