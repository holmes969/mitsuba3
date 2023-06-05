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
max_depth = 4
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
sc.param_map["Shape[1]"].set_transform(Matrix4fD([[1.,0.,0.,P],
                                                    [0.,1.,0.,0.],
                                                    [0.,0.,1.,0.],
                                                    [0.,0.,0.,1.],]))
sc_manager.configure()
options = tr.RenderOptions()
options.spp = spp
options.spp_batch = spp
options.max_bounces = max_depth-1
options.export_deriv = True
integrator = tr.PathTracer()
integrator.set_param(options)
integrator.set_manager(sc_manager)
img_width = sc.param_map["Sensor[0]"].width
img_height = sc.param_map["Sensor[0]"].height

def primal_benchmark():
    t0 = time()
    image = integrator.renderC()
    t1 = time()
    print(f"[Benchmark] primal rendering (spp = {spp}) takes {t1 - t0} sec")
    out = image.numpy().reshape((img_width, img_height, 3))
    output = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path + '/primal_tr.exr', output)

def forward_ad_benchmark():
    t0 = time()
    image = integrator.renderD(Vector3fC())
    dr.set_grad(P, 1.0)
    dr.forward_to(image, flags=dr.ADFlag.ClearInterior)
    grad_image = dr.grad(image)
    t1 = time()
    print(f"[Benchmark] forward ad (spp = {spp}) takes {t1 - t0} sec")
    grad_image = grad_image.numpy().reshape((img_height, img_width, 3))
    grad_image[:, :, 1] = 0.0
    grad_image[:, :, 2] = 0.0
    output = cv2.cvtColor(grad_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path + '/derivative_tr.exr', output)


# primal_benchmark()
forward_ad_benchmark()

# def render_target(xml_path):
#     # setup scene_manager
#     sc_manager = tr.SceneManager()
#     sc_manager.load_file(xml_path)
#     sc = sc_manager.scene
#     sc_manager.configure()
#     # setup render_options
#     options = tr.RenderOptions()
#     options.spp = 256
#     options.spp_batch = 256
#     options.max_bounces = 2
#     # setup integrator
#     integrator = tr.PathTracer()
#     integrator.set_param(options)
#     integrator.set_manager(sc_manager)
#     img_width = sc.param_map["Sensor[0]"].width
#     img_height = sc.param_map["Sensor[0]"].height
#     # render and save to exr file
#     img = integrator.renderC().numpy().reshape((img_height, img_width, 3))
#     output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(output_dir + "forward.exr", output)

# def test_forward_ad(xml_path):
#     # setup scene_manager
#     sc_manager = tr.SceneManager()
#     sc_manager.load_file(xml_path)
#     sc = sc_manager.scene
#     P = FloatD(0.0)
#     drjit.enable_grad(P)
#     sc.param_map["Shape[0]"].set_transform(Matrix4fD([[1.,0.,0.,P],
# 													  [0.,1.,0.,P],
# 													  [0.,0.,1.,P],
# 													  [0.,0.,0.,1.],]))
#     sc_manager.configure()
#     # setup render_options
#     options = tr.RenderOptions()
#     options.spp = 256               # interior
#     options.spp_batch = 256
#     options.sppe = 256              # primary
#     options.sppe_batch = 256
#     options.sppe0 = 256             # pixel
#     options.sppe0_batch = 256
#     options.sppse0 = 256            # direct
#     options.sppse0_batch = 256
#     options.sppse1 = 256            # indirect
#     options.sppse1_batch = 256
#     options.max_bounces = 5
#     options.export_deriv = True
#     # setup integrator
#     integrators = [tr.PathTracer(), tr.BoundaryPixel(), tr.BoundaryPrimary(), tr.BoundaryDirect(), tr.BoundaryIndirect()]
#     fns = ["deriv_interior.exr", "deriv_pixel.exr", "deriv_primary.exr", "deriv_direct.exr", "deriv_indirect.exr"]
#     for idx, integrator in enumerate(integrators):
#         integrator.set_param(options)
#         integrator.set_manager(sc_manager)
#         img_width = sc.param_map["Sensor[0]"].width
#         img_height = sc.param_map["Sensor[0]"].height
#         # render and save to exr file
#         img = integrator.renderD(Vector3fC())
#         drjit.set_grad(P, 1.0)
#         drjit.forward_to(img, flags=drjit.ADFlag.ClearInterior)
#         diff_img = drjit.grad(img).numpy().reshape((img_height, img_width, 3))
#         diff_img[:, :, 1] = 0.0
#         diff_img[:, :, 2] = 0.0
#         output = cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(output_dir + fns[idx], output)
#         print("[INFO] Derivative image rendered!")

# render_target("scene_tr.xml")
# # test_forward_ad("scene_tr.xml")