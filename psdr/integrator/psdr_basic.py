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
        # Should we continue tracing to reach one more vertex?
        active_next = si.is_valid()
        # Get the BSDF. Potentially computes texture-space differentials.
        bsdf = si.bsdf(ray)
        # Overwrite wi using path-space formulation
        wi = dr.normalize(ray.o - si.p)
        si.wi = dr.select(active_next, si.to_local(wi), si.wi)
        # path throughput
        _, cam_imp = sensor.sample_direction(si, mi.Point2f(), active_next)
        contrb = cam_imp * dr.abs(dr.dot(wi, si.n))
        contrb = contrb / dr.detach(contrb)
        β = dr.select(active_next, si.j * contrb, β)
        # Differentiable evaluation of intersected emitter / envmap
        Le = si.emitter(scene).eval(si)
        L += Le * β
        # emitter sample
        active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
        ds, weight_em = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
        print(ds.j)
        active_em &= dr.neq(ds.pdf, 0.0)

        return L, active, None

mi.register_integrator("psdr_basic", lambda props: PathSpaceBasicIntegrator(props))
