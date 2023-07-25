import os
import sys
import torch
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

# load pbdr system
pbdr_sys = sys.argv[1]                                      # 'psdr' OR 'mitsuba'
print("[Info] Benchmark using [" + pbdr_sys + "]")
mitsuba_path = os.path.abspath("../..")
psdr_path = os.path.abspath("../../../psdr-jit/")
if pbdr_sys == 'mitsuba':
    sys.path.insert(0, os.path.join(mitsuba_path, 'build\Release\python'))
    sys.path.insert(0, os.path.join(mitsuba_path, 'psdr\integrator'))
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import psdr_jit
    import psdr_basic
    import psdr_jit_prb
elif pbdr_sys == 'psdr':
    psdr_path = os.path.abspath("../../../psdr-jit/")
    sys.path.insert(0, os.path.join(psdr_path, 'build\python'))
    import psdr_jit as psdr
else:
    assert(False)
import drjit as dr
from drjit.cuda.ad import Float as FloatD, Matrix4f as Matrix4fD
from drjit.cuda import Array3f as Vector3fC
from time import time

# render options
max_depth = 2
spp = 128

# load 3D scene
scene_dir = '../scenes/'
scene_fn = 'cbox_bunny.xml'
result_dir = '../results/'

if pbdr_sys == 'mitsuba':
    scene_path = os.path.join(scene_dir, scene_fn)
    sc = mi.load_file(scene_path, integrator='prb', max_depth=max_depth)
elif pbdr_sys == 'psdr':
    curr_dir = os. getcwd()
    os.chdir(scene_dir)
    sc = psdr.Scene()
    sc.opts.log_level = 0
    sc.load_file(scene_fn)
    w = sc.opts.width
    h = sc.opts.height
    os.chdir(curr_dir)

# config scene with differentiable params
var = torch.tensor([0.0], device='cuda', dtype=torch.float32, requires_grad = True)
if pbdr_sys == 'mitsuba':
    dr_var = mi.Float(var)
    dr.enable_grad(dr_var)
    key = 'bunny_shape.vertex_positions'
    params = mi.traverse(sc)
    initial_vertex_positions = dr.unravel(mi.Point3f, params[key])
    trafo = mi.Transform4f.translate([dr_var, 0.0, 0.0])
    params[key] = dr.ravel(trafo @ initial_vertex_positions)
    params.update()
elif pbdr_sys == 'psdr':
    dr_var = FloatD(var)
    dr.enable_grad(dr_var)

    sc.opts.spp = spp
    sc.param_map["Mesh[0]"].set_transform(Matrix4fD([[1.,0.,0.,dr_var],
                                                 [0.,1.,0.,0.0],
                                                 [0.,0.,1.,0.0],
                                                 [0.,0.,0.,1.0],]))
    sc.configure([0])
    integrator = psdr.PathTracer(max_depth - 1)
dr.eval()

class RenderFunctionMitsuba(torch.autograd.Function):
    @staticmethod
    def forward(ctx, var):
        with dr.suspend_grad():
            image = sc.integrator().render(sc, spp = spp).torch()
        return image

    @staticmethod
    def backward(ctx, grad_out):
        mi.set_variant('cuda_ad_rgb')
        integrator = sc.integrator()
        params = mi.traverse(sc)
        integrator.render_backward(sc, params, grad_out)
        grad_val = dr.grad(dr_var).torch()
        return tuple([grad_val])

class RenderFunctionPSDR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, var):
        with dr.suspend_grad():
            image = integrator.renderC(sc, 0).torch()
        return image.reshape((w, h, 3))

    @staticmethod
    def backward(ctx, grad_out):
        image_grad = Vector3fC(grad_out.reshape(-1,3))
        image = integrator.renderD(sc, 0)
        tmp = dr.dot(image_grad, image)
        dr.backward(tmp)
        grad_val = dr.grad(dr_var).torch()
        return tuple([grad_val])

if pbdr_sys == 'mitsuba':
    renderer = RenderFunctionMitsuba.apply
else:
    renderer = RenderFunctionPSDR.apply

test_case = sys.argv[2]
t0 = time()
if test_case == 'primal':
    res = renderer(var)     # primal rendering
elif test_case == 'fwd':
    if pbdr_sys == 'mitsuba':
        dr.set_grad(dr_var, 1.0)
        res = sc.integrator().render_forward(sc, params, spp = spp).torch()
    elif pbdr_sys == 'psdr':
        image = integrator.renderD(sc, 0)
        dr.forward(dr_var)
        res = dr.grad(image).torch().reshape((w, h, 3))
elif test_case == 'bwd':
    img = renderer(var)
    img_target = torch.zeros(img.shape, device = img.device)
    loss = torch.mean(torch.square(img_target - img))
    loss.backward()
    del loss
dr.eval()
dr.sync_thread()
t1 = time()
print(f"[Benchmark] {test_case} using {pbdr_sys} (spp = {spp}) takes {t1-t0} sec")

if test_case != 'bwd':
    output = res.detach().cpu().numpy()
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(result_dir, test_case + '_' + pbdr_sys + '.exr')
    cv2.imwrite(out_path, output)