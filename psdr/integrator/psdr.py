from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import numpy as np
import common
from psdr_interior import PathSpaceInteriorIntegrator
from psdr_primary import PathSpacePrimaryIntegrator
from psdr_direct import PathSpaceDirectIntegrator
from psdr_indirect import PathSpaceIndirectIntegrator
from psdr_pixel import PathSpacePixelIntegrator


import cv2
import os
def save_image(fn, image):
    output = image.detach().cpu().numpy()
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    out_path = os.path.join('./',  fn)
    cv2.imwrite(out_path, output)

import mitsuba as mi

class PathSpaceIntegrator(mi.CppADIntegrator):
    def __init__(self, props = mi.Properties()):
        super().__init__(props)
        self.interior = PathSpaceInteriorIntegrator(props)
        self.primary = PathSpacePrimaryIntegrator(props)
        self.direct = PathSpaceDirectIntegrator(props)
        self.indirect = PathSpaceIndirectIntegrator(props)
        self.pixel = PathSpacePixelIntegrator(props)
        self.spp_pixel = 0
        self.spp_primary = 0
        self.spp_direct = 0
        self.spp_indirect = 0
    
    def set_boundary_spp(self,
                         spp_pixel = 0,
                         spp_primary = 0,
                         spp_direct = 0,
                         spp_indirect = 0):
        self.spp_pixel = spp_pixel
        self.spp_primary = spp_primary
        self.spp_direct = spp_direct
        self.spp_indirect = spp_indirect

    
    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:
        return self.interior.render(scene, sensor, seed, spp, develop, evaluate)

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        grad_interior = self.interior.render_forward(scene, params, sensor, seed, spp)
        # save_image(f'mi_debug_interior.exr', grad_interior.torch())
        grad_primary = self.primary.render_forward(scene, params, sensor, seed, self.spp_primary)
        # save_image(f'mi_debug_primary.exr', grad_primary.torch())
        grad_direct = self.direct.render_forward(scene, params, sensor, seed, self.spp_direct)
        # save_image(f'mi_debug_direct.exr', grad_direct.torch())
        grad_indirect = self.indirect.render_forward(scene, params, sensor, seed, self.spp_indirect)
        # save_image(f'mi_debug_indirect.exr', grad_indirect.torch())
        grad_pixel = self.pixel.render_forward(scene, params, sensor, seed, self.spp_pixel)
        # save_image(f'mi_debug_pixel.exr', grad_pixel.torch())
        return grad_interior + grad_primary + grad_direct + grad_indirect + grad_pixel

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:
        self.interior.render_backward(scene, params, grad_in, sensor, seed, spp)
        self.primary.render_backward(scene, params, grad_in, sensor, seed, self.spp_primary)
        self.direct.render_backward(scene, params, grad_in, sensor, seed, self.spp_direct)
        self.indirect.render_backward(scene, params, grad_in, sensor, seed, self.spp_indirect)
        self.pixel.render_backward(scene, params, grad_in, sensor, seed, self.spp_pixel)
    
mi.register_integrator("psdr", lambda props: PathSpaceIntegrator(props))


