from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

import common

class PathSpaceBasicIntegrator(common.PSIntegrator):
    def __init__(self, props):
        super().__init__(props)
    
    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sensor: mi.Sensor,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               active: mi.Bool,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, mi.Spectrum]:
        
        β = mi.Spectrum(1)              # Path throughput weight
        L = mi.Spectrum(0)              # Radiance accumulator
        active = mi.Bool(active)        # Active SIMD lanes

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()
        # Detach ray to avoid differentiable ray intersect
        si = scene.ray_intersect(dr.detach(ray), active)
        # # Overwrite wi using path-space formulation
        # tmp = dr.normalize(ray.o - si.p)
        # si.wi = dr.select(si.active, si.to_local(tmp), si.wi)
        # # Differentiable evaluation of intersected emitter / envmap
        # Le = si.emitter(scene).eval(si)
        # # 
        # β = dr.select(si.active, )          # 
        L += si.emitter(scene).eval(si)

        return L, active, None

mi.register_integrator("psdr_basic", lambda props: PathSpaceBasicIntegrator(props))
