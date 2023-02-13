#!/usr/bin/env python3

import open3d as o3d


for i in range(1, 64):
    pcd = o3d.io.read_point_cloud("gt-point-cloud/0000000" + str(i) + '-gt-pointcloud.ply')
    o3d.io.write_point_cloud("gt_640/" + str(i) + '.pcd', pcd)