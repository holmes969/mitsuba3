import argparse
import os, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch, cv2


scene_dir = '../scenes/'
scene_fn = 'cbox_bunny.xml'
result_dir = '../results/'
max_depth = 4

def save_image(fn, image):
    output = image.detach().cpu().numpy()
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(result_dir,  fn)
    cv2.imwrite(out_path, output)

def run_mitsuba(mode, spp):
    mitsuba_path = os.path.abspath("../..")
    sys.path.insert(0, os.path.join(mitsuba_path, 'build\Release\python'))
    sys.path.insert(0, os.path.join(mitsuba_path, 'psdr\integrator'))
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import psdr_jit_prb
    import drjit as dr

    # select integrator based on mode
    integrator_name = {
        "forward" : "psdr_jit_prb",
        "interior": "psdr_jit_prb"
    }

    # load 3D scene from xml file
    sc_path = os.path.join(scene_dir, scene_fn)
    sc = mi.load_file(sc_path, integrator=integrator_name[mode], max_depth=max_depth)
    # configure scene with differentiable param.
    dr_var = mi.Float(0.0)
    dr.enable_grad(dr_var)
    key = 'bunny_shape.vertex_positions'
    params = mi.traverse(sc)
    initial_vertex_positions = dr.unravel(mi.Point3f, params[key])
    trafo = mi.Transform4f.translate([dr_var, 0.0, 0.0])
    params[key] = dr.ravel(trafo @ initial_vertex_positions)
    params.update()
    # start rendering under different modes
    if mode == "forward":
        with dr.suspend_grad():
            image = sc.integrator().render(sc, spp = spp).torch()
    else:
        dr.forward(dr_var, dr.ADFlag.ClearEdges)
        image = sc.integrator().render_forward(sc, params, spp = spp).torch()
    save_image(f'mi_{mode}.exr', image)
    print(f"[Info] Mitsuba [{mode}], spp = {spp}!")

def run_tensorray(mode, spp):
    tr_path = os.path.abspath("../../../TensorRay/")
    mi_path = os.path.abspath("../../")
    sys.path.insert(0, os.path.join(tr_path, 'build\python'))
    import TensorRay as tr
    import drjit as dr
    from drjit.cuda.ad import Float as FloatD, Matrix4f as Matrix4fD
    from drjit.cuda import Array3f as Vector3fC
    scene_path = os.path.join(mi_path, 'psdr/scenes')
    result_path = os.path.join(mi_path, 'psdr/results')
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
    options = tr.RenderOptions()
    options.spp = spp                 # interior
    options.spp_batch = spp
    options.sppe = spp              # primary
    options.sppe_batch = spp
    options.sppe0 = spp             # pixel
    options.sppe0_batch = spp
    options.sppse0 = spp            # direct
    options.sppse0_batch = spp
    options.sppse1 = spp            # indirect
    options.sppse1_batch = spp

    options.max_bounces = max_depth - 1
    img_width = sc.param_map["Sensor[0]"].width
    img_height = sc.param_map["Sensor[0]"].height
    sc_manager.configure()
    if mode == 'forward':
        integrator = tr.PathTracer()
        integrator.set_param(options)
        integrator.set_manager(sc_manager)
        image = integrator.renderC().torch()
    else:
        options.export_deriv = True
        integrator_map = {'interior': tr.PathTracer(),
                          'pixel': tr.BoundaryPixel(),
                          'primary': tr.BoundaryPrimary(),
                          'direct': tr.BoundaryDirect(),
                          'indirect': tr.BoundaryIndirect()}
        integrator = integrator_map[mode]
        integrator.set_param(options)
        integrator.set_manager(sc_manager)
        _tmp = integrator.renderD(Vector3fC())
        dr.forward(P)
        # dr.set_grad(P, 1.0)
        # dr.forward_to(_tmp, flags=dr.ADFlag.ClearInterior)
        image = dr.grad(_tmp).torch()
    image = image.reshape((img_width, img_height, 3))
    save_image(f'tr_{mode}.exr', image)
    print(f"[Info] TensorRay [{mode}], spp = {spp}!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script used to validate the correct implementation of PSDR integrators")
    parser.add_argument("-b", "--backend", type=str, required=True, help="The PBDR backend [1] mi (mitsuba) [2] tr (tensorray)")
    parser.add_argument("-m", "--mode", type=str, required=True, help="Mode for validation [1] forward [2] interior [3] pixel [4] primary [5] direct [6] indirect")
    parser.add_argument("-s", "--spp", type=int, default=256, help="Sample per pixel")
    args = parser.parse_args()
    if args.backend in {"mi", "mitsuba"}:
        run_mitsuba(args.mode, args.spp)
    elif args.backend in {"tr","tensorray"}:
        run_tensorray(args.mode, args.spp)
    else:
        print(f"[Error] Backend '{args.backend}' is not supported!")