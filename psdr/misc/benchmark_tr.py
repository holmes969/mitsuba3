import sys
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
tr_path = os.path.abspath("../../../TensorRay/")
sys.path.insert(0, os.path.join(tr_path, 'build\python'))
import TensorRay as tr
import cv2
import torch
import drjit as dr
from drjit.cuda.ad import Float as FloatD, Matrix4f as Matrix4fD
from drjit.cuda import Array3f as Vector3fC
from time import time

spp = 64
max_depth = 2
mitsuba_path = os.path.abspath("../../")
scene_path = os.path.join(mitsuba_path, 'psdr/scenes')
result_path = os.path.join(mitsuba_path, 'psdr/results')
os.chdir(scene_path)

# setup scene_manager
sc_manager = tr.SceneManager()
sc_manager.load_file("cbox_bunny.xml")
sc = sc_manager.scene
P = FloatD(0.0)
dr.enable_grad(P)
sc.param_map["Shape[0]"].set_transform(Matrix4fD([[1.,0.,0.,P],
                                                    [0.,1.,0.,0.],
                                                    [0.,0.,1.,0.],
                                                    [0.,0.,0.,1.],]))
options = tr.RenderOptions()
options.spp = spp
options.spp_batch = spp
options.max_bounces = max_depth - 1
# options.export_deriv = True
integrator = tr.PathTracer()
integrator.set_param(options)
integrator.set_manager(sc_manager)
img_width = sc.param_map["Sensor[0]"].width
img_height = sc.param_map["Sensor[0]"].height

num_iters = 1
def primal_benchmark():
    avg_time_elapsed = 0.0
    sc_manager.configure()
    dr.eval()
    for i in range(num_iters):
        t0 = time()
        image = integrator.renderC()
        dr.eval(image)
        dr.sync_thread()
        t1 = time()
        avg_time_elapsed += t1 - t0
    avg_time_elapsed /= num_iters
    print(f"[Benchmark] primal rendering (spp = {spp}) takes {avg_time_elapsed} sec")
    out = image.numpy().reshape((img_width, img_height, 3))
    output = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path + '/primal_tr.exr', output)

def forward_ad_benchmark():
    avg_time_elapsed = 0.0
    sc_manager.configure()
    dr.eval()
    for i in range(num_iters): 
        t0 = time()
        image = integrator.renderD(Vector3fC())
        dr.forward(P)
        grad_image = dr.grad(image)
        dr.eval(grad_image)
        dr.sync_thread()
        t1 = time()
        avg_time_elapsed += t1 - t0
    avg_time_elapsed /= num_iters
    print(f"[Benchmark] forward ad (spp = {spp}) takes {avg_time_elapsed} sec")
    grad_image = grad_image.numpy().reshape((img_height, img_width, 3))
    grad_image[:, :, 1] = 0.0
    grad_image[:, :, 2] = 0.0
    output = cv2.cvtColor(grad_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path + '/derivative_tr.exr', output)

def backward_ad_benchmark():
    image_ref = dr.zeros(dr.cuda.ad.Array3f, img_width * img_height)
    avg_time_elapsed = 0.0
    sc_manager.configure()
    dr.eval()
    for i in range(num_iters):
        t0 = time()
        image = integrator.renderD(Vector3fC())
        loss = dr.mean(dr.sqr(image - image_ref))
        dr.backward(loss)
        grad_val = dr.grad(P)
        dr.eval(grad_val)
        dr.sync_thread()
        t1 = time()
        avg_time_elapsed += t1 - t0
    avg_time_elapsed /= num_iters
    print(f"[Benchmark] backward ad (spp = {spp}) takes {avg_time_elapsed} sec")

# primal_benchmark()
# forward_ad_benchmark()
# backward_ad_benchmark()