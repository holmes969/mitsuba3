from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

import common

class PathSpaceJitIntegratorPRB(common.PSIntegratorPRB):
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
        ray_in = ray
        # --------------------- Configure loop state ----------------------
        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        β = mi.Spectrum(1)                            # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes
        
        # Variables caching information from the previous iteration
        prev_pi         = dr.zeros(mi.PreliminaryIntersection3f)
        prev_ray        = dr.zeros(mi.Ray3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        pi              = scene.ray_intersect_preliminary(ray, coherent=dr.eq(depth, 0))

        # Handle differentiation of camera importance outside of Loop
        if not primal:
            with dr.resume_grad():
                si = pi.compute_surface_interaction(ray, mi.RayFlags.All | mi.RayFlags.FollowShape)
                _, cam_imp = sensor.sample_direction(si, mi.Point2f(), si.is_valid())             
                dir = ray_in.o - si.p
                wi = dr.normalize(dir)
                val = cam_imp * dr.abs(dr.dot(wi, si.n))
                imp = dr.select(dr.neq(val, 0), val / dr.detach(val), 0)
                Lo = imp * L
                if dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo):
                    raise Exception(
                        "The contribution computed by the differential "
                        "rendering phase is not attached to the AD graph! "
                        "Raising an exception since this is usually "
                        "indicative of a bug (for example, you may have "
                        "forgotten to call dr.enable_grad(..) on one of "
                        "the scene parameters, or you may be trying to "
                        "optimize a parameter that does not generate "
                        "derivatives in detached PRB.)")

                # Propagate derivatives from/to 'Lo' based on 'mode'
                if mode == dr.ADMode.Backward:
                    dr.backward_from(δL * Lo)
                else:
                    δL += dr.forward_to(Lo)

        # Record the following loop in its entirety
        loop = mi.Loop(name="Path-Space Diff. Rendering (%s)" % mode.name,
                       state=lambda: (sampler, ray, depth, L, β, active, pi,
                                      prev_pi, prev_ray, prev_bsdf_pdf, prev_bsdf_delta))
        loop.set_max_iterations(self.max_depth)
        while loop(active):
            # Enable RayFlags.FollowShape to avoid diff. ray-surface intersect
            with dr.resume_grad(when=not primal):
                prev_si = prev_pi.compute_surface_interaction(prev_ray, mi.RayFlags.All | mi.RayFlags.FollowShape)
                si = pi.compute_surface_interaction(ray, mi.RayFlags.All | mi.RayFlags.FollowShape)
                prev_p = dr.select(dr.eq(depth, 0), ray_in.o, prev_si.p)
                # Overwrite wi using path-space formulation
                active_next = si.is_valid()                
                dir = prev_p - si.p
                wi = dr.normalize(dir)
                si.wi = dr.select(active_next, si.to_local(wi), si.wi)

            # ---------------------- Direct emission ----------------------
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            mis = common.mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )
            with dr.resume_grad(when=not primal):
                Le = β * mis * ds.emitter.eval(si)

            bsdf = si.bsdf(ray)
            # ---------------------- Emitter sampling ----------------------
            active_next &= depth + 1 < self.max_depth
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            with dr.resume_grad(when=not primal):
                ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
                active_em &= dr.neq(ds.pdf, 0.0)
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
            depth[si.is_valid()] += 1
            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            β *= bsdf_weight
            pi = scene.ray_intersect_preliminary(ray, coherent=dr.eq(depth, 0))

            # ------------------ Differential phase only ------------------
            if not primal:
                with dr.resume_grad():
                    si_next = pi.compute_surface_interaction(ray, mi.RayFlags.All | mi.RayFlags.FollowShape, active_next)
                    wo = si_next.p - si.p
                    dist = dr.norm(wo)
                    wo /= dist
                    # Detached version of the above term and inverse
                    bsdf_val_det = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_det = dr.select(dr.neq(bsdf_val_det, 0),
                                                 dr.rcp(bsdf_val_det), 0)
                    # Re-evaluate (differentiably) BSDF * cos(theta) * G
                    bsdf_val = bsdf.eval(bsdf_ctx, si, si.to_local(wo), active_next)
                    geo_term = dr.abs(dr.dot(wo, si.n)) / (dist * dist)
                    vert_val = bsdf_val * geo_term / dr.detach(geo_term) * si.j
                    vert_val = dr.select(dr.neq(vert_val, 0), vert_val, 0)
                    Lr_ind = L * dr.replace_grad(1, inv_bsdf_val_det * vert_val)
                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + Lr_dir + Lr_ind
                    if dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            # -------------------- Stopping criterion ---------------------
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)
            active = active_next

        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L                    # State for the differential phase
        )
mi.register_integrator("psdr_jit_prb", lambda props: PathSpaceJitIntegratorPRB(props))
