import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
mitsuba_path = os.path.abspath("../..")
sys.path.insert(0, os.path.join(mitsuba_path, 'build\Release\python'))
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from time import time
import drjit as dr
from drjit.cuda.ad import Float as FloatD, Matrix4f as Matrix4fD, UInt as UIntD, Int as IntD
from drjit.cuda.ad import Array2u as Array2uD, Array3u as Array3uD, Array3i as Array3iD
import torch


# render options
max_depth = 5
spp = 128

# load 3D scene
scene_dir = '../scenes/'
scene_fn = 'cbox_bunny.xml'

scene_path = os.path.join(scene_dir, scene_fn)
sc = mi.load_file(scene_path, integrator='path', max_depth=max_depth)

class EdgeList:
    def __init__(self, shape):
        assert(shape.is_mesh())
        nface = shape.face_count()
        idx_v = shape.face_indices(dr.arange(dr.cuda.UInt, nface))
        idx_v_torch = Array3iD(idx_v).torch()                       # [bug?] idx_v_torch = idx_v.torch() => Error code: Unsupported kUInt bits 32
        indices = torch.full((nface*3, 5), -1, dtype=torch.int64)
        for i in range(3):
            v0 = idx_v_torch[:, i]
            v1 = idx_v_torch[:, (i+1)%3]
            indices[i::3, 0] = torch.where(v0 < v1, v0, v1)         # (smaller) index of edge vertices
            indices[i::3, 1] = torch.where(v0 > v1, v0, v1)         # (larger)  index of edge vertices
            indices[i::3, 2] = idx_v_torch[:, (i+2)%3]              # index of neighbor-triangle vertex 
        indices[:, 3] = torch.arange(nface).repeat_interleave(3)    # index of neighbor triangle
        edge_dict = {}
        # [cz] can we avoid this large FOR loop?
        for i in range(3*nface):
            key = (indices[i, 0].item(), indices[i, 1].item())
            if key not in edge_dict:
                edge_dict[key] = i
            else:
                indices[edge_dict[key], 4] = indices[i, 3]          # index of the other neighbor triangle (-1 if not exist)
        indices = indices[list(edge_dict.values()), :].cuda()       # remove duplicates
        self.num_edges = indices.shape[0]
        self.idx_v0 = UIntD(list(indices[:, 0]))                    # [bug?] Can't directly convert pytorch array to dr.cuda.ad.UInt
        self.idx_v1 = UIntD(list(indices[:, 1]))
        self.idx_v2 = UIntD(list(indices[:, 2]))
        self.idx_f0 = UIntD(list(indices[:, 3]))
        self.idx_f1 = UIntD(list(indices[:, 4]))

    def print(self):
        print(f"[EdgeList] num_edges = {self.num_edges}")
        
# e = EdgeList(sc.shapes()[1])

for shape in sc.shapes():
    e = EdgeList(shape)
    e.print()
    