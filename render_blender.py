# The MIT License (MIT)
#
# Copyright (c) 2016 Panmari
# Copyright (c) 2020 Markus Völk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse, sys, os, math, re
import bpy
import math



import argparse
import numpy as np
import os
import glob



parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=10,
                    help='number of views to be rendered')
parser.add_argument('--shapenet_path', type=str, default='./shapenet-car',
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='./data-generated',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='16',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--resolution', type=int, default=200,
                    help='Resolution of the images.')
parser.add_argument('--engine', type=str, default='BLENDER_EEVEE',
                    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

argv = sys.argv[sys.argv.index("--") + 1:]  # get the args after "--" for this python scripts
args = parser.parse_args(argv)

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine
render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = args.color_depth  # ('8', '16')
render.image_settings.file_format = "PNG"  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
render.film_transparent = True

scene.use_nodes = True
# scene.view_layers["View Layer"].use_pass_normal = True
scene.view_layers["View Layer"].use_pass_diffuse_color = True
# scene.view_layers["View Layer"].use_pass_object_index = True

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new('CompositorNodeRLayers')

# Create depth output nodes
depth_file_output = nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
depth_file_output.base_path = ''
depth_file_output.file_slots[0].use_node_format = True
depth_file_output.format.file_format = "OPEN_EXR"
depth_file_output.format.color_depth = args.color_depth
links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
obj_instance_dirs = sorted(list(os.listdir(args.shapenet_path)))
# create a new file name "data_generated" in the same directory
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

obj_instance_dirs = obj_instance_dirs[:min(2, len(obj_instance_dirs))]  # Keep 200 first

instance_paths = [os.path.join(args.shapenet_path, obj_instance_dir, "models", "model_normalized.obj")
                  for obj_instance_dir in obj_instance_dirs]
for instance_path in instance_paths:
    print(instance_path)
    # # Delete default cube
    context.active_object.select_set(True)
    bpy.ops.object.delete()

    # # Delete everything in the scene
    # bpy.ops.object.select_all(action='DESELECT')
    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.delete()

    # Import textured mesh
    bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.import_scene.obj(filepath=instance_path)
    # this import_scene will automated make the car y-forward,z-up

    obj = bpy.context.selected_objects[0]
    context.view_layer.objects.active = obj

    # Possibly disable specular shading
    for slot in obj.material_slots:
        node = slot.material.node_tree.nodes['Principled BSDF']
        node.inputs['Specular'].default_value = 0.05

    if args.scale != 1:
        bpy.ops.transform.resize(value=(args.scale, args.scale, args.scale))
        bpy.ops.object.transform_apply(scale=True)
    if args.remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if args.edge_split:
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    # Make light just directional, disable shadows.
    light = bpy.data.lights['Light']
    light.type = 'SUN'
    light.use_shadow = False
    # Possibly disable specular shading:
    light.specular_factor = 1.0
    light.energy = 10.0
    # Add another light source so stuff facing away from light is not completely dark
    bpy.ops.object.light_add(type='SUN')
    light2 = bpy.data.lights['Sun']
    light2.use_shadow = False
    light2.specular_factor = 1.0
    light2.energy = 0.015
    bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
    bpy.data.objects['Sun'].rotation_euler[0] += 180
    # Place camera
    cam = scene.objects['Camera']
    cam.location = (0, 1, 0.6)
    focal = 200
    cam.data.angle = np.arctan(args.resolution / 2 / focal) * 2
    intrinsics = np.array([[focal, 0, args.resolution / 2], [0, focal, args.resolution / 2], [0, 0, 1]])
    # cam.data.sensor_width = 32
    # cam.data.angle = 1.5707963267948966
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    cam.parent = cam_empty

    scene.collection.objects.link(cam_empty)
    context.view_layer.objects.active = cam_empty
    cam_constraint.target = cam_empty

    stepsize = 360.0 / args.views
    rotation_mode = 'XYZ'

    instance_id = instance_path.split('/')[-3]
    fp = os.path.join(os.path.abspath(args.output_folder), instance_id, "rendered")
    pose_dir = os.path.join(os.path.abspath(args.output_folder), instance_id, "pose")
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)
    color_dir = os.path.join(fp, "color")
    if not os.path.exists(color_dir):
        os.makedirs(color_dir)
    depth_dir = os.path.join(fp, "depth")
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    pcd_dir = os.path.join(fp, "pointclouds")
    if not os.path.exists(pcd_dir):
        os.makedirs(pcd_dir)
    bpy.context.scene.view_layers['View Layer'].update()
    obj_matrix_world_inv = obj.matrix_world.inverted()
    for i in range(0, args.views):
        # print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))

        # render_file_path = fp + f'_r_{int(i * stepsize)}'
        if i == 0:
            # save intrinsic matrix
            np.save(os.path.join(pose_dir, "intrinsics.npy"), intrinsics)
        scene.render.filepath = os.path.join(color_dir, "rendered_color") + f'_r_{int(i * stepsize)}'
        depth_file_output.file_slots[0].path = os.path.join(depth_dir, "rendered_depth") + f'_r_{int(i * stepsize)}'

        bpy.context.scene.view_layers['View Layer'].update()
        camera_matrix_world = cam.matrix_world
        obj_matrix_world_inv = obj.matrix_world.inverted()
        # Compute camera pose in object's coordinate system
        cam_pose_obj_coord_sys = np.dot(obj_matrix_world_inv, camera_matrix_world)
        np.save(os.path.join(pose_dir, f"pose_r_{int(i * stepsize)}.npy"), cam_pose_obj_coord_sys)
        bpy.ops.render.render(write_still=True)  # render still

        cam_empty.rotation_euler[2] += math.radians(stepsize)

    # the saved file name of depth file will have 0001 in the end, currently don't know how to fix it, so we just rename it
    pattern = "rendered_depth_r_*0001.exr"

    # get a list of files in the directory matching the pattern
    files = glob.glob(os.path.join(depth_dir, pattern))

    # loop through the files and rename them
    for file in files:
        new_name = file.replace("0001", "")
        os.rename(file, new_name)
