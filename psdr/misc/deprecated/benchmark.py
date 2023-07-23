
import sys
import os
mitsuba_path = os.path.abspath("../..")
sys.path.insert(0, os.path.join(mitsuba_path, 'build\Release\python'))
sys.path.insert(0, os.path.join(mitsuba_path, 'psdr\integrator'))
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr
import psdr_basic
import psdr_jit
from time import time

spp = 64
max_depth = 2
scene_path = '../scenes/cbox_bunny.xml'
scene = mi.load_file(scene_path, integrator='psdr_jit', max_depth=max_depth)
# Set parameter to be differentiated
var = mi.Float(0.0)
dr.enable_grad(var)
key = 'emitter.vertex_positions'
params = mi.traverse(scene)
initial_vertex_positions = dr.unravel(mi.Point3f, params[key])
trafo = mi.Transform4f.translate([var, 0.0, 0.0])
params[key] = dr.ravel(trafo @ initial_vertex_positions)
# Propagate this change to the scene internal state
params.update()
dr.eval() 
num_iters = 1
# dr.set_log_level(3)
# dr.set_flag(dr.JitFlag.KernelHistory, True)

def primal_benchmark():
    avg_time_elapsed = 0.0
    for i in range(num_iters):
        t0 = time()
        image = mi.render(scene, params, spp=spp)
        dr.eval(image)
        dr.sync_thread()
        t1 = time()
        avg_time_elapsed += t1 - t0
    avg_time_elapsed /= num_iters
    print(f"[Benchmark] primal rendering (spp = {spp}) takes {avg_time_elapsed} sec")
    mi.util.write_bitmap("../results/primal_mi.exr", image)

def forward_ad_benchmark():  
    avg_time_elapsed = 0.0
    for i in range(num_iters):
        t0 = time()
        # Render and record the computational graph
        image = mi.render(scene, params, spp=spp)
        # Forward-propagate gradients through the computation graph
        dr.forward(var)
        # Fetch the image gradient values
        grad_image = dr.grad(image)
        dr.eval(grad_image)
        dr.sync_thread()
        t1 = time()
        avg_time_elapsed += t1 - t0
    avg_time_elapsed /= num_iters
    # dr.set_log_level(0)
    grad_image[:, :, 1] = 0.0
    grad_image[:, :, 2] = 0.0
    print(f"[Benchmark] forward ad (spp = {spp}) takes {avg_time_elapsed} sec")
    mi.util.write_bitmap("../results/derivative_mi.exr", grad_image)

def backward_ad_benchmark():
    # Render and record the computational graph
    film_size = params['PerspectiveCamera.film.size'] 
    image_ref = dr.zeros(dr.cuda.ad.TensorXf, [film_size[0], film_size[1], 3])
    avg_time_elapsed = 0.0
    for i in range(num_iters):
        t0 = time()
        image = mi.render(scene, params, spp=spp)
        loss = dr.mean(dr.sqr(image - image_ref))
        dr.backward(loss)
        grad_val = dr.grad(var)
        dr.eval(grad_val)
        print(grad_val)
        dr.sync_thread()
        t1 = time()
        avg_time_elapsed += t1 - t0
    avg_time_elapsed /= num_iters
    print(f"[Benchmark] backward ad (spp = {spp}) takes {avg_time_elapsed} sec")

# primal_benchmark()
# forward_ad_benchmark()
backward_ad_benchmark()

# history = dr.kernel_history()
# print(len(history))
