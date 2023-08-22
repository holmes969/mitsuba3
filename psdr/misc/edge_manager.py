import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
mitsuba_path = os.path.abspath("../..")
sys.path.insert(0, os.path.join(mitsuba_path, 'build\Release\python'))
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from time import time
import drjit as dr
from drjit.cuda.ad import Float as FloatD, Matrix4f as Matrix4fD, UInt as UIntD, Int as IntD
from drjit.cuda.ad import Array2u as Array2uD, Array3u as Array3uD, Array3i as Array3iD
import torch


# render options
max_depth = 5
spp = 128

# load 3D scene
scene_dir = '../scenes/'
scene_fn = 'cbox_bunny.xml'

scene_path = os.path.join(scene_dir, scene_fn)
sc = mi.load_file(scene_path, integrator='path', max_depth=max_depth)
for shape in sc.shapes():
    num_edge = shape.edge_count()
    for i in range(num_edge):
        print(shape.edge_indices_v(i))
    print("------------------------------------")
    break
