
import sys
mitsuba_path = "C:\\Users\\holmes969\\mitsuba3\\"
# mitsuba_path = "D:\\Cheng\\holmes969\\mitsuba3\\"
sys.path.insert(0, mitsuba_path + "build\\Release\\python")
sys.path.insert(0, mitsuba_path + "psdr\\integrator")

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import drjit as dr
import psdr_basic
import psdr_direct
from time import time

spp = 64
scene_path = '../scenes/cbox_bunny.xml'
scene = mi.load_file(scene_path, integrator='psdr_basic', max_depth='4')
# Set parameter to be differentiated
var = mi.Float(0.0)
dr.enable_grad(var)
key = 'bunny_shape.vertex_positions'
params = mi.traverse(scene)
initial_vertex_positions = dr.unravel(mi.Point3f, params[key])
trafo = mi.Transform4f.translate([var, 0.0, 0.0])
params[key] = dr.ravel(trafo @ initial_vertex_positions)
# Propagate this change to the scene internal state
params.update()

def primal_benchmark():
    t0 = time()
    image = mi.render(scene, params, spp=spp)
    dr.eval(image)
    t1 = time()
    print(f"[Benchmark] primal rendering (spp = {spp}) takes {t1 - t0} sec")

def forward_ad_benchmark():
    # Render and record the computational graph
    image = mi.render(scene, params, spp=spp)
    t0 = time()
    # Forward-propagate gradients through the computation graph
    dr.forward(var, dr.ADFlag.ClearEdges)
    # Fetch the image gradient values
    grad_image = dr.grad(image)
    t1 = time()
    print(f"[Benchmark] forward ad (spp = {spp}) takes {t1 - t0} sec")
    mi.util.write_bitmap("../results/derivative.exr", grad_image)


def backward_ad_benchmark():
    # Render and record the computational graph
    image_ref = dr.detach(mi.render(scene, params, spp=spp)) + 1.0
    image = mi.render(scene, params, spp=spp)
    loss = dr.mean(dr.sqr(image - image_ref))
    t0 = time()
    dr.backward(loss)
    t1 = time()
    print(f"[Benchmark] backward ad (spp = {spp}) takes {t1 - t0} sec")

primal_benchmark()
forward_ad_benchmark()
backward_ad_benchmark()