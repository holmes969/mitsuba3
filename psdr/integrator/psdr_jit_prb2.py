from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

import common

class PathSpaceJitIntegratorPRB2(common.PSIntegratorPRB):
    def __init__(self, props):
        super().__init__(props)

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sensor: mi.Sensor,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, mi.Spectrum]:
        
        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()
        
        # --------------------- Configure loop state ----------------------
        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(ray)                           # [cz] do we need to resume grad?
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        β = mi.Spectrum(1)                            # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes
        scalar_depth = 0
        
        # Variables caching information from the previous iteration
        pi              = scene.ray_intersect_preliminary(ray, coherent=dr.eq(depth, 0))
        prev_pi         = dr.zeros(mi.PreliminaryIntersection3f)
        prev_ray        = dr.zeros(mi.Ray3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)


        # Record the following loop in its entirety
        loop = mi.Loop(name="Path Replay Backpropagation (%s) for PSDR" % mode.name,
                       state=lambda: (sampler, ray, depth, L, δL, β, η, active,
                                      pi, prev_pi, prev_bsdf_pdf, prev_bsdf_delta))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)        
        while loop(active):
            with dr.resume_grad(when=not primal):
                if scalar_depth == 0:
                    si = pi.compute_surface_interaction(ray, mi.RayFlags.All)
                    prev_si = dr.zeros(mi.SurfaceInteraction3f)
                    active_next = si.is_valid()
                else:
                    si = pi.compute_surface_interaction(ray, mi.RayFlags.All | mi.RayFlags.FollowShape)
                    prev_ray_flag = mi.RayFlags.All
                    if scalar_depth > 1:
                        prev_ray_flag |= mi.RayFlags.FollowShape
                    # [cz] We only need to compute prev.si (lightweight?)
                    prev_si = prev_pi.compute_surface_interaction(prev_ray, prev_ray_flag)
                    active_next = si.is_valid()
                    # Update si.wi using PSDR formula
                    wo = si.p - prev_si.p
                    dist = dr.norm(wo)
                    wo /= dist
                    si.wi = dr.select(active_next, si.to_local(-wo), si.wi)
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------
            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            mis = common.mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )
            with dr.resume_grad(when=not primal):
                Le = β * mis * ds.emitter.eval(si)

            # Should we continue tracing to reach one more vertex?
            active_next &= depth + 1 < self.max_depth

            # ---------------------- Emitter sampling ----------------------
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            with dr.resume_grad(when=not primal):
                ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
                active_em = active & dr.neq(ds.pdf, 0.0)
                wo = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
                mis_em = dr.select(ds.delta, 1, common.mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = β * mis_em * bsdf_value_em * em_weight * ds.j

            # ------------------ Detached BSDF sampling -------------------
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)
            # Information about the current vertex needed by the next iteration
            prev_ray = ray
            prev_pi = pi
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # ---- Update loop variables based on current interaction -----
            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β *= bsdf_weight
            depth[si.is_valid()] += 1
            pi = scene.ray_intersect_preliminary(ray, coherent=dr.eq(depth, 0))

            # -------------------- Stopping criterion ---------------------
            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            scalar_depth += 1
            active = active_next

        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L                    # State for the differential phase
        )

mi.register_integrator("psdr_jit_prb2", lambda props: PathSpaceJitIntegratorPRB2(props))
