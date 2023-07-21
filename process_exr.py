'''
MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import cv2
import Imath
import OpenEXR
import argparse
import array
import numpy as np
import os
import open3d as o3d

def read_exr(exr_path, height, width):
    file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[depth>5] = 0
    # pring the unique value of depth
    return depth


def depth2pcd(depth, intrinsics, pose):
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)
    y, x = np.where(depth > 0)
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0))
    # camera coordinates -> object coordinates
    points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0)).T[:, :3]
    print(points)
    return points


if __name__ == '__main__':
    intrinsics = np.load("./test-data/rendered/models/models_r_000_intrinsics.npy")
    print(intrinsics)
    width = int(intrinsics[0, 2] * 2)
    height = int(intrinsics[1, 2] * 2)
    depth_dir = os.path.join("./test-data/rendered/models/depth")
    pcd_dir = os.path.join("./test-data/rendered/models/pcd")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)
    exr_path = os.path.join("./test-data/rendered/models/models_r_072_depth0001.exr")
    pose_path = os.path.join("./test-data/rendered/models/models_r_072_pose2.npy")

    depth = read_exr(exr_path, height, width)
    pose = np.load(pose_path)
    points = depth2pcd(depth, intrinsics, pose)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(os.path.join(pcd_dir, 'test.ply'), pcd)
    # 归一化深度图到 0-255
    depth_norm = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    # 保存深度图
    cv2.imwrite(os.path.join(depth_dir, 'test.png'), depth_norm)

    #load ply file using open3d
    full_pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, 'model_normalized.ply'))
    #transfer it to numpy array
    full_pcd = np.asarray(full_pcd.points)
    print(full_pcd)