# shapnet-pointcloud-generator
This repository is for generating complete pointclouds, partial pointclouds, rendered depth maps and rendered rgb images from ShapeNet.

Most codes come from [stanford-shapenet-renderer]{https://github.com/panmari/stanford-shapenet-renderer}

To run these scripts, you need to have blender in your system.

I haven't organized these codes properly, these codes are just for reference only.

By running **render_blender.py**, blender will be used to render your object and some files will be saved.

Then, run **process_exr.py** to generate the final images.

To genrate partil pointclouds, check **point_sampling.py**.
