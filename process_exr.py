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
from joblib import Parallel, delayed
from easydict import EasyDict


def read_exr(exr_path, height, width):
    file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[depth > 5] = 0
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
    return points


def pointcloud_generation(instance_dir):
    intrinsics = np.load(os.path.join(instance_dir, "pose", "intrinsics.npy"))
    width = int(intrinsics[0, 2] * 2)
    height = int(intrinsics[1, 2] * 2)
    depth_path = os.path.join(instance_dir, "rendered", "depth")
    depth_dirs = os.listdir(depth_path)
    for depth_dir in depth_dirs:
        depth = read_exr(os.path.join(depth_path, depth_dir), height, width)
        pose = np.load(os.path.join(instance_dir, "pose", f"pose_r_" + depth_dir.split(".")[0].split("_")[-1] + ".npy"))
        points = depth2pcd(depth, intrinsics, pose)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(instance_dir, "rendered", "pointclouds",
                                              f"partial_r_" + depth_dir.split(".")[0].split("_")[-1] + ".ply"), pcd)
        depth_norm = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(depth_dir, depth_dir.split('.')[0] + '.png'), depth_norm)


def pointclouds_generation(args):
    instance_dirs = sorted(list(os.listdir(args.rendered_path)))
    instance_dirs = [os.path.join(args.rendered_path, instance_dir) for instance_dir in instance_dirs]
    # create a new file name "data_generated" in the same directory

    instance_dirs = instance_dirs[:min(2, len(instance_dirs))]  # Keep 200 first

    _ = Parallel(n_jobs=len(instance_dirs))(
        delayed(pointcloud_generation)(instance_dir) for instance_dir in instance_dirs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rendered_path', type=str, help='Input/Output folder', default="./data-generated")
    args = parser.parse_args()

    pointclouds_generation(EasyDict(args.__dict__))


if __name__ == '__main__':
    main()
