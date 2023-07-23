
import sys
import os
mitsuba_path = os.path.abspath("../..")
sys.path.insert(0, os.path.join(mitsuba_path, 'build\Release\python'))
sys.path.insert(0, os.path.join(mitsuba_path, 'psdr\integrator'))
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr
import test
from time import time

spp = 64
max_depth = 2
scene_path = '../scenes/cbox_bunny.xml'
scene = mi.load_file(scene_path, integrator='psdr_test', max_depth=max_depth)
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

def primal_test():
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
    mi.util.write_bitmap("../results/primal_test.exr", image)

def write_str_to_file(fn, str):
    text_file = open(fn, "wt")
    n = text_file.write(str)
    text_file.close()

dr.set_flag(dr.JitFlag.KernelHistory, True)
primal_test()
history = dr.kernel_history()
for index, entry in enumerate(history):
    if 'ir' in entry:
        ptx_str = entry['ir'].getvalue()
        write_str_to_file("ptx_%d.txt" % index, ptx_str)
