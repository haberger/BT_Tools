#!/usr/bin/env python3

import open3d as o3d
import trimesh
import numpy as np

# mesh = o3d.io.read_triangle_mesh("/home/dalina/David/Uni/BachelorThesis/Tools/D435 dataset old/objects/Canister/Canister_.ply")
mesh = o3d.io.read_triangle_mesh("/home/dalina/David/Uni/BachelorThesis/Dataset/Canister_hull.ply")

count = 10

for i in range(0,4):
    pcd = o3d.geometry.sample_points_poisson_disk(mesh, number_of_points=count)
    #pcd = o3d.geometry.uniform_down_sample(pcd, 2)
    #pcd = o3d.geometry.sample_points_uniformly(mesh, number_of_points=count)
    o3d.io.write_point_cloud("canister"+str(count)+".pcd", pcd)
    if count == 10000:
        count = 20000
    else:
        count=count*10
    print(count)


# mesh = trimesh.load_mesh("/home/dalina/David/Uni/BachelorThesis/Dataset/Canister_hull.ply")
# p,i = trimesh.sample.sample_surface_even(mesh, count = 30000)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.array(p))
# o3d.io.write_point_cloud('test.pcd', pcd)
# print(type(p))