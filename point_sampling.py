import argparse
import os
import glob
import joblib
from joblib import Parallel, delayed
from collections import defaultdict
import numpy as np
from easydict import EasyDict


class Count(object):
    failed_example = 0
    failed_example_path = []

    @staticmethod
    def add(path):
        Count.failed_example += 1
        Count.failed_example_path.append(path)
class BatchCompletionCallBack(object):
    completed = defaultdict(int)

    def __init__(self, time, index, parallel):
        self.index = index
        self.parallel = parallel

    def __call__(self, index):
        BatchCompletionCallBack.completed[self.parallel] += 1
        print("Progress : %s %% " %
              str(BatchCompletionCallBack.completed[self.parallel] * 100 / len(obj_instances)))
        if self.parallel._original_iterator is not None:
            self.parallel.dispatch_next()

def sample_instance(instance_path, virtualscan):
    print(instance_path)
    raise ValueError
    path = "/".join(instance_path.split(sep='/')[:-3])
    category = instance_path.split(sep='/')[-3]
    instance_name = instance_path.split(sep='/')[-2]
    ply_path = os.path.join(path, "ply", category)
    if not os.path.exists(ply_path):
        os.makedirs(ply_path)
    out_path = os.path.join(ply_path, instance_name + ".points.ply")
    command = virtualscan + " " + instance_path + " 10"
    if not os.path.exists(out_path):
        try:
            os.system(command)
            os.rename(instance_path[:-3] + "ply", out_path)
        except:
            Count.add(instance_path)
            print(Count.failed_example_path)


def shoot_rays(args):
    obj_instance_dirs = sorted(list(os.listdir(args.shapenet_path)))
    # create a new file name "data_generated" in the same directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)



    obj_instance_dirs = obj_instance_dirs[:min(200, len(obj_instance_dirs))]  # Keep 200 first

    instance_paths = [os.path.join(args.shapenet_path, obj_instance_dir, "models", "model_normalized.obj")
                      for obj_instance_dir in obj_instance_dirs]


    _ = Parallel(n_jobs=-1) \
        (delayed(sample_instance)(*input_args) for input_args in zip(instance_paths, [args.virtualscan] * len(instance_paths)))

    print(f"{Count.failed_example} failed examples")
    print(Count.failed_example_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shapenet_path', type=str, help='Input folder',default="./shapenet-car")
    parser.add_argument('--output_path', type=str, help='Output folder',default="./data-generated")
    parser.add_argument('--virtualscan', type=str, help='Path to the virtual scanner executable', default='O-CNN/virtual_scanner/build/virtualscanner')
    args = parser.parse_args()

    shoot_rays(EasyDict(args.__dict__))

if __name__ == '__main__':
    main()