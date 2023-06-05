import sys
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
psdr_path = os.path.abspath("../../../psdr-jit/")
sys.path.insert(0, os.path.join(psdr_path, 'build\python'))
import psdr_jit as psdr
import drjit as dr
from drjit.cuda.ad import Float as FloatD, Matrix4f as Matrix4fD
from time import time
import cv2

spp = 64
max_depth = 2
mitsuba_path = os.path.abspath("../../")
scene_path = os.path.join(mitsuba_path, 'psdr/scenes')
result_path = os.path.join(mitsuba_path, 'psdr/results')
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
                                                 [0.,0.,0.,1.0],]))
sc.configure([0])

integrator = psdr.PathTracer(max_depth-1)

def primal_benchmark():
    t0 = time()
    image = integrator.renderC(sc, 0)
    t1 = time()
    print(f"[Benchmark] primal rendering (spp = {spp}) takes {t1 - t0} sec")
    out = image.numpy().reshape((sc.opts.width, sc.opts.height, 3))
    output = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path + '/primal_psdr.exr', output)

def forward_ad_benchmark():
    t0 = time()
    # Render and record the computational graph
    image = integrator.renderD(sc, 0)
    # Forward-propagate gradients through the computation graph
    # dr.forward(var, dr.ADFlag.ClearEdges)
    dr.set_grad(var, 1.0)
    dr.forward_to(image)    
    # Fetch the image gradient values
    grad_image = dr.grad(image)
    t1 = time()
    print(f"[Benchmark] forward ad (spp = {spp}) takes {t1 - t0} sec")
    out = grad_image.numpy().reshape((sc.opts.width, sc.opts.height, 3))
    out[:, :, 1] = 0.0
    out[:, :, 2] = 0.0
    output = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path + '/derivative_psdr.exr', output)

def backward_ad_benchmark():
    image_ref = integrator.renderC(sc, 0) + 1.0
    image = integrator.renderD(sc, 0)
    loss = dr.mean(dr.sqr(image - image_ref))
    t0 = time()
    dr.backward(loss)
    t1 = time()
    print(f"[Benchmark] backward ad (spp = {spp}) takes {t1 - t0} sec")

primal_benchmark()
# forward_ad_benchmark()
# backward_ad_benchmark()
