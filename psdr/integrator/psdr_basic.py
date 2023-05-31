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
        
        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------
        # Copy input arguments to avoid mutating the caller's state
        β = mi.Spectrum(1)              # Path throughput weight
        L = mi.Spectrum(0)              # Radiance accumulator
        depth = mi.UInt32(0)            # Depth of current vertex
        active = mi.Bool(active)        # Active SIMD lanes

        # Record the following loop in its entirety
        loop = mi.Loop(name="Path-Space Diff. Rendering (%s)" % mode.name,
                       state=lambda: (sampler, ray, depth, L, β, active))
        while loop(active):
            # Enable RayFlags.FollowShape to avoid diff. ray-surface intersect
            si = scene.ray_intersect(ray, mi.RayFlags.All | mi.RayFlags.FollowShape, active)
            # Overwrite wi using path-space formulation
            active_next = si.is_valid()
            wi = dr.normalize(ray.o - si.p)
            si.wi = dr.select(active_next, si.to_local(wi), si.wi)
            # Overwrite the primary-ray throughput using path-space formulation
            is_primary_ray = dr.eq(depth, 0)
            overwrite_imp = is_primary_ray & active_next
            ds, cam_imp = sensor.sample_direction(si, mi.Point2f(), overwrite_imp)
            contrb = cam_imp * dr.abs(dr.dot(wi, si.n))
            contrb = dr.select(contrb > 0.0, contrb / dr.detach(contrb), 0.0)           # why is this needed?
            β = β * dr.select(overwrite_imp, si.j * contrb, 1.0)
            # Direct emission (only for primary ray)
            Le = si.emitter(scene).eval(si, is_primary_ray)
            L += Le * β


            # # Should we continue tracing to reach one more vertex?
            # active_next = (depth + 1 < self.max_depth) & active_next

            # # Get the BSDF. Potentially computes texture-space differentials.
            # bsdf = si.bsdf(ray)

            # # ---------------------- Emitter sampling ----------------------
            # # Is emitter sampling even possible on the current vertex?
            # active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            # ds, weight_em = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
            # active_em &= dr.neq(ds.pdf, 0.0)
            # if dr.any(active_em):
            #     wo = si.to_local(ds.d)
            #     bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            #     L += bsdf_value_em * weight_em * β
            


            active = active_next
            depth[active] += 1
            


        #     # path throughput
        #     ds, cam_imp = sensor.sample_direction(si, mi.Point2f(), active_next)
        #     contrb = cam_imp * dr.abs(dr.dot(wi, si.n))
        #     contrb = dr.select(contrb > 0.0, contrb / dr.detach(contrb), 0.0)       # why is this needed?
        #     β = β * dr.select(active_next, si.j * contrb, 0.0)

        #     # Is emitter sampling even possible on the current vertex?
        #     active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        #     active = active_next


        # # Get the BSDF. Potentially computes texture-space differentials.
        # bsdf = si.bsdf(ray)
        # # Overwrite wi using path-space formulation
        # wi = dr.normalize(ray.o - si.p)
        # si.wi = dr.select(active_next, si.to_local(wi), si.wi)
        # # path throughput
        # ds, cam_imp = sensor.sample_direction(si, mi.Point2f(), active_next)
        # contrb = cam_imp * dr.abs(dr.dot(wi, si.n))
        # contrb = dr.select(contrb > 0.0, contrb / dr.detach(contrb), 0.0)       # why is this needed?
        # β = β * dr.select(active_next, si.j * contrb, 0.0)
        # # Differentiable evaluation of intersected emitter / envmap
        # Le = si.emitter(scene).eval(si)
        # L += Le * β
        # # emitter sample
        # active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
        # ds, weight_em = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
        # active_em &= dr.neq(ds.pdf, 0.0)
        # wo = si.to_local(ds.d)
        # bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        # L += bsdf_value_em * weight_em * β
        return L, active, None

mi.register_integrator("psdr_basic", lambda props: PathSpaceBasicIntegrator(props))
