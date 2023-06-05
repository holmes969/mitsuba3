from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

import common

class PathSpaceDirectIntegrator(common.PSIntegrator):
    def __init__(self, props = mi.Properties()):
        super().__init__(props)

        max_depth = props.get('max_depth', 6)
        if max_depth < 0 and max_depth != -1:
            raise Exception("\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        # Map -1 (infinity) to 2^32-1 bounces
        self.max_depth = max_depth if max_depth != -1 else 0xffffffff

        self.rr_depth = props.get('rr_depth', 5)
        if self.rr_depth <= 0:
            raise Exception("\"rr_depth\" must be set to a value greater than zero!")
    
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
        bsdf_ctx = mi.BSDFContext()     # Standard BSDF evaluation context for path tracing

        si = scene.ray_intersect(ray, mi.RayFlags.All | mi.RayFlags.FollowShape, active)
        # Overwrite wi using path-space formulation
        active_next = si.is_valid()
        wi = dr.normalize(ray.o - si.p)
        si.wi = dr.select(active_next, si.to_local(wi), si.wi)
        # Overwrite the primary-ray throughput using path-space formulation
        ds, cam_imp = sensor.sample_direction(si, mi.Point2f(), active_next)
        contrb = cam_imp * dr.abs(dr.dot(wi, si.n))
        contrb = dr.select(contrb > 0.0, contrb / dr.detach(contrb), 0.0)           # why is this needed?
        β = β * dr.select(active_next, si.j * contrb, 1.0)
        # ---------------------- Direct Emission ----------------------
        Le = si.emitter(scene).eval(si)
        L += Le * β
        # ---------------------- Emitter sampling ----------------------
        bsdf = si.bsdf(ray)
        active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
        ds, weight_em = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
        active_em &= dr.neq(ds.pdf, 0.0)
        wo = si.to_local(ds.d)
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        L += bsdf_value_em * weight_em * β

        return L, active, None

mi.register_integrator("psdr_direct", lambda props: PathSpaceDirectIntegrator(props))
