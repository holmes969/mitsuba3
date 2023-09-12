from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

import common

class PathSpaceJitIntegrator(common.PSIntegrator):
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
        # Copy the arguments
        ray = mi.Ray3f(ray)
        active = mi.Bool(active)

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # Primary ray-surface intersection
        pi = scene.ray_intersect_preliminary(ray, coherent=mi.Bool(True), active=active)
        si = pi.compute_surface_interaction(ray, mi.RayFlags.All, active)
        active &= si.is_valid()

        # --------------------- Configure loop state ----------------------
        L = si.emitter(scene).eval(si, active)      # Radiance accumulator
        β = mi.Spectrum(1)                          # Path throughput weight

        for depth in range(self.max_depth - 1):
            bsdf = si.bsdf(ray)
            # ---------------------- Emitter sampling ----------------------
            ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active)
            active_em = active & dr.neq(ds.pdf, 0.0)
            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            mis_em = dr.select(ds.delta, 1, common.mis_weight(ds.pdf, bsdf_pdf_em))
            L[active_em] += mis_em * bsdf_value_em * em_weight * β * ds.j

            # ---------------------- BSDF sampling ----------------------
            with dr.suspend_grad():
                bsdf_sample, _ = bsdf.sample(bsdf_ctx, si,
                                             sampler.next_1d(),
                                             sampler.next_2d(),
                                             active)
            ray = si.spawn_ray(dr.detach(si.to_world(bsdf_sample.wo)))
            pi_next = scene.ray_intersect_preliminary(ray, coherent=dr.eq(depth, 0), active=active)
            # Avoid differentiable ray-surface intersection here!
            si_next = pi_next.compute_surface_interaction(ray, mi.RayFlags.All | mi.RayFlags.FollowShape, active)
            active &= si_next.is_valid()
            # Update wi and wo based on PS formulation
            wo = si_next.p - si.p
            dist = dr.norm(wo)
            wo /= dist
            si_next.wi = dr.select(active, si_next.to_local(-wo), si_next.wi)
            # Eval BSDF
            bsdf_val = bsdf.eval(bsdf_ctx, si, si.to_local(wo), active)
            geo_term = dr.abs(si_next.wi.z) / (dist * dist)
            # MIS
            ds = mi.DirectionSample3f(scene, si=si_next, ref=si)
            bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            mis_bsdf = common.mis_weight(bsdf_sample.pdf,
                                         scene.pdf_emitter_direction(si, ds, ~bsdf_delta))
            # Updatye throughput and accumulated radiance
            β *= bsdf_val * geo_term / (bsdf_sample.pdf * dr.detach(geo_term)) * si_next.j
            L[active] += β * si_next.emitter(scene).eval(si_next, active) * mis_bsdf
            
            # Update si
            si = si_next

        return L, active, None

mi.register_integrator("psdr_jit", lambda props: PathSpaceJitIntegrator(props))
