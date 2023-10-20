from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import numpy as np
import common
from psdr_interior import PathSpaceInteriorIntegrator
from psdr_primary import PathSpacePrimaryIntegrator
from psdr_direct import PathSpaceDirectIntegrator


import mitsuba as mi

class PathSpaceIntegrator(mi.CppADIntegrator):
    def __init__(self, props = mi.Properties()):
        super().__init__(props)
        self.interior = PathSpaceInteriorIntegrator(props)
        self.primary = PathSpacePrimaryIntegrator(props)
        self.direct = PathSpaceDirectIntegrator(props)
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
        grad_primary = self.primary.render_forward(scene, params, sensor, seed, self.spp_primary)
        grad_direct = self.direct.render_forward(scene, params, sensor, seed, self.spp_direct)
        return grad_interior + grad_primary + grad_direct

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
    
mi.register_integrator("psdr", lambda props: PathSpaceIntegrator(props))


