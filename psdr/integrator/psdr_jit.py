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
        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------
        ray = mi.Ray3f(ray)
        β = mi.Spectrum(1)              # Path throughput weight
        L = mi.Spectrum(0)              # Radiance accumulator
        depth = mi.UInt32(0)            # Depth of current vertex
        active = mi.Bool(active)        # Active SIMD lanes

        # --------------------- Variables caching information from the previous bounce ----------------------
        prev_ray = dr.zeros(mi.Ray3f)
        prev_pi = dr.zeros(mi.PreliminaryIntersection3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        for idepth in range(self.max_depth):
            prev_si = prev_pi.compute_surface_interaction(prev_ray, mi.RayFlags.All | mi.RayFlags.FollowShape, active)
            # Enable RayFlags.FollowShape to avoid diff. ray-surface intersect
            pi = scene.ray_intersect_preliminary(ray, coherent=dr.eq(depth, 0), active=active)
            si = pi.compute_surface_interaction(ray, mi.RayFlags.All | mi.RayFlags.FollowShape, active)
            # Overwrite wi using path-space formulation
            active_next = si.is_valid()
            dir = ray.o - si.p
            wi = dr.normalize(dir)
            si.wi = dr.select(active_next, si.to_local(wi), si.wi)
            β_mult = mi.Float(0.0)
            β_pdf = mi.Float(1.0)

            if idepth == 0:
                # Primary ray: re-write sensor importance
                valid_primary_its = mi.Bool(active_next)
                _, cam_imp = sensor.sample_direction(si, mi.Point2f(), valid_primary_its)
                cam_cos_imp = cam_imp * dr.abs(dr.dot(wi, si.n))
                valid_primary_its &= cam_cos_imp > 0.0              # to remove nan values (why needed?)
                β_mult = dr.select(valid_primary_its, si.j * cam_cos_imp, β_mult)
                β_pdf = dr.select(valid_primary_its, dr.detach(cam_cos_imp), β_pdf)
            else:
                # Secondary ray: update based on 3-point BSDF
                valid_sec_its = mi.Bool(active_next)
                prev_bsdf = prev_si.bsdf()
                prev_bsdf_val = prev_bsdf.eval(bsdf_ctx,
                                            prev_si,
                                            prev_si.to_local(-wi),
                                            active_next)

                geo_term = dr.abs(si.wi.z) / dr.squared_norm(dir)
                β_mult = dr.select(valid_sec_its, si.j * prev_bsdf_val * geo_term, β_mult)
                β_pdf = dr.select(valid_sec_its, prev_bsdf_pdf * dr.detach(geo_term), β_pdf)

            # Update path throughput based on path-space formulation
            β *= β_mult / β_pdf

            # ---------------------- Direct emission ----------------------
            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            mis = common.mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )
            L += β * mis * ds.emitter.eval(si)

            # ---------------------- Emitter sampling ----------------------
            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)
            # Should we continue tracing to reach one more vertex?
            active_next = active_next & (depth + 1 < self.max_depth)
            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)
            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            mis_em = dr.select(ds.delta, 1, common.mis_weight(ds.pdf, bsdf_pdf_em))
            L += mis_em * bsdf_value_em * em_weight * β * ds.J

            # ------------------ Detached BSDF sampling -------------------
            with dr.suspend_grad():
                bsdf_sample, _ = bsdf.sample(bsdf_ctx, si,
                                             sampler.next_1d(),
                                             sampler.next_2d(),
                                             active_next)

            # ---- Update loop variables based on current interaction -----
            prev_ray = ray
            ray = si.spawn_ray(dr.detach(si.to_world(bsdf_sample.wo)))
            prev_pi = pi
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            depth[si.is_valid()] += 1
            active = active_next

        return L, active, None

mi.register_integrator("psdr_jit", lambda props: PathSpaceJitIntegrator(props))
