import sys
mitsuba_path = "C:\\Users\\holmes969\\mitsuba3\\"
# mitsuba_path = "D:\\Cheng\\holmes969\\mitsuba3\\"
sys.path.insert(0, mitsuba_path + "build\\Release\\python")
sys.path.insert(0, mitsuba_path + "psdr\\integrator")

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import drjit as dr
import psdr_basic


# dr.set_log_level(3)
# Load scene
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
# Render and record the computational graph
image = mi.render(scene, params, spp=64)
mi.util.write_bitmap("../results/forward.exr", image)

# Forward-propagate gradients through the computation graph
dr.forward(var, dr.ADFlag.ClearEdges)


# Fetch the image gradient values
grad_image = dr.grad(image)
# write to files
grad_image[:, :, 1] = 0.0
grad_image[:, :, 2] = 0.0
mi.util.write_bitmap("../results/derivative.exr", grad_image)