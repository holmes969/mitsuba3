import sys
mitsuba_path = "C:\\Users\\holmes969\\mitsuba3\\build\\Release\\python"
sys.path.insert(0, mitsuba_path)
import drjit as dr
dr.set_log_level(3)                     # Reduce the log level to only see the kernel launch
dr.set_flag(dr.JitFlag.PrintIR, True)   # Ask Dr.Jit to print the generate IR code
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
scene = mi.load_dict({
    'type': 'scene',
    # 'sphere': { 'type': 'sphere' }
    'sphere': {
        'type': 'obj',
        'filename': '../../tutorials/scenes/meshes/sphere.obj'
    },
})
ray_origin = mi.Point3f(0, 0, -5)
ray_dir = dr.normalize(mi.Vector3f(0, 0, 1))
ray = mi.Ray3f(o=ray_origin, d=ray_dir)
si = scene.ray_intersect_preliminary(ray)
dr.eval(si)