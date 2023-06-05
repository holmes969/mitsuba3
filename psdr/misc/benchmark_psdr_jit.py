import sys
psdr_path = "C:\\Users\\holmes969\\psdr-jit\\"
sys.path.insert(0, psdr_path + "build\\python")
import psdr_jit as psdr
import drjit as dr
from drjit.cuda.ad import Float as FloatD, Matrix4f as Matrix4fD
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from time import time
import cv2

spp = 64
mitsuba_path = "C:\\Users\\holmes969\\mitsuba3\\"
scene_path = mitsuba_path + 'psdr\\scenes\\'
os.chdir(scene_path)

sc = psdr.Scene()
sc.opts.log_level = 0
sc.opts.spp = spp
sc.load_file("cbox_bunny.xml")

var = FloatD(0.0)
dr.enable_grad(var)
sc.param_map["Mesh[1]"].set_transform(Matrix4fD([[1.,0.,0.,var],
                                                    [0.,1.,0.,0.0],
                                                    [0.,0.,1.,0.0],
                                                    [0.,0.,0.,1.],]))
sc.configure([0])

integrator = psdr.PathTracer(3)

def primal_benchmark():
    t0 = time()
    image = integrator.renderC(sc, 0)
    t1 = time()
    print(f"[Benchmark] primal rendering (spp = {spp}) takes {t1 - t0} sec")

def forward_ad_benchmark():
    # Render and record the computational graph
    image = integrator.renderD(sc, 0)
    t0 = time()
    # Forward-propagate gradients through the computation graph
    dr.forward(var, dr.ADFlag.ClearEdges)
    # Fetch the image gradient values
    grad_image = dr.grad(image)
    t1 = time()
    print(f"[Benchmark] forward ad (spp = {spp}) takes {t1 - t0} sec")
    out = grad_image.numpy().reshape((sc.opts.width, sc.opts.height, 3))
    output = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mitsuba_path + "\\psdr\\results\\psdr-jit.exr", output)

def backward_ad_benchmark():
    image_ref = integrator.renderC(sc, 0) + 1.0
    image = integrator.renderD(sc, 0)
    loss = dr.mean(dr.sqr(image - image_ref))
    t0 = time()
    dr.backward(loss)
    t1 = time()
    print(f"[Benchmark] backward ad (spp = {spp}) takes {t1 - t0} sec")

primal_benchmark()
forward_ad_benchmark()
backward_ad_benchmark()


# scene = mi.load_file(scene_path, integrator='psdr_basic', max_depth='4')
# # Set parameter to be differentiated
# var = mi.Float(0.0)
# dr.enable_grad(var)
# key = 'bunny_shape.vertex_positions'
# params = mi.traverse(scene)
# initial_vertex_positions = dr.unravel(mi.Point3f, params[key])
# trafo = mi.Transform4f.translate([var, 0.0, 0.0])
# params[key] = dr.ravel(trafo @ initial_vertex_positions)
# # Propagate this change to the scene internal state
# params.update()

