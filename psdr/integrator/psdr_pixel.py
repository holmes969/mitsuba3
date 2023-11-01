from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
import numpy as np
import common


class PathSpacePixelIntegrator(common.PSIntegratorBoundary):
    def __init__(self, props):
        super().__init__(props)
    
    def eval_boundary_segment(
        self,
        edge_sample,
        si_0,           # sensor
        si_1,
    ):
        active = si_0.is_valid() & si_1.is_valid()
        with dr.suspend_grad():
            # non-differentiable component
            ray_dir = si_1.p - si_0.p
            dist = dr.norm(ray_dir)
            ray_dir /= dist
            dist1 = dr.norm(edge_sample.p - si_0.p)
            cos2 = dr.abs_dot(si_1.n, -ray_dir)
            e = dr.cross(edge_sample.e, ray_dir)
            sinphi = dr.norm(e)
            proj = dr.normalize(dr.cross(e, si_1.n))
            sinphi2 = dr.norm(dr.cross(ray_dir, proj))
            n = dr.normalize(dr.cross(si_1.n, proj))
            sign0 = dr.dot(e, edge_sample.e2) > 0.0
            sign1 = dr.dot(e, n) > 0.0
            active &= (sinphi > mi.math.EdgeEpsilon) & (sinphi2 > mi.math.EdgeEpsilon)
            baseVal = (dist / dist1) * (sinphi / sinphi2) * cos2 * dr.select(dr.eq(sign0, sign1), 1.0, -1.0)
        # differential component
        x_dot_n = dr.dot(n, si_1.p)
        return baseVal * x_dot_n / edge_sample.pdf, active

    def sample_boundary_segment(
        self,
        scene: mi.Scene,
        sensor_id: int,
        sampler: mi.Sampler,
    ):
        sensor = scene.sensors()[sensor_id]
        film = sensor.film()
        rfilter = film.rfilter()
        if not rfilter.is_box_filter():
            raise Exception("Currently, only box filter is supported for pixel boundary term.")

        # sample point on geometric edge
        edge_sample = scene.sample_edge_point(sampler.next_1d(), mi.BoundaryFlags.Pixel, sensor_id)
        sensor_pos = sensor.world_transform().translation()
        tmp_si = dr.zeros(mi.Interaction3f)
        tmp_si.p = edge_sample.p
        dir = dr.normalize(edge_sample.p - sensor_pos)

        # sensor-side end point of the boundary segment (fixed at camera position for primary boundary)
        endpoint_s = dr.zeros(mi.Interaction3f)
        endpoint_s.p = sensor_pos
        tmp_si.n = -dir
        sensor_ray = tmp_si.spawn_ray_to(endpoint_s.p)
        visible = ~scene.ray_test(sensor_ray)
        endpoint_s.t = dr.select(visible, 0.0, dr.inf)

        # emitter-side end point of the boundary segment
        tmp_si.n = dir
        tmp_ray = tmp_si.spawn_ray(dir)
        pi = scene.ray_intersect_preliminary(tmp_ray, coherent=False)
        tmp_ray.o = endpoint_s.p
        tmp_ray.d = dr.normalize(edge_sample.p - tmp_ray.o)
        endpoint_e = pi.compute_surface_interaction(tmp_ray, mi.RayFlags.PathSpace | mi.RayFlags.All)

        # evaluate the boundary segment
        weight, active = self.eval_boundary_segment(edge_sample, endpoint_s, endpoint_e)
        return edge_sample, endpoint_s, endpoint_e, weight, active

    def sample_sensor_subpath(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        edge_sample,
        endpoint_s,
        endpoint_e,
        sensor: mi.Sensor,
        active: mi.Bool
    ):
        active = mi.Bool(active) 
        tmp_si = dr.zeros(mi.Interaction3f)
        tmp_si.p = edge_sample.p
        ds, cam_imp = sensor.sample_direction(tmp_si, mi.Point2f(), active)
        film = sensor.film()
        pos = ds.uv + film.crop_offset()
        dist2_a = dr.squared_norm(tmp_si.p - endpoint_s.p)
        dist2_b = dr.squared_norm(endpoint_e.p - endpoint_s.p)
        active &= dr.neq(ds.pdf, 0.0)
        weight = (cam_imp * dist2_a / dist2_b) & active
        return [weight], [pos]

    def sample_emitter_subpath(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        edge_sample,
        endpoint_e,
        active: mi.Bool
    ):
        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------
        ray_dir = dr.normalize(endpoint_e.p - edge_sample.p)
        ray = mi.Ray3f(edge_sample.p + mi.math.ShadowEpsilon * ray_dir,    # CZ: any way to avoid explicitedly add this shadowEpsilon?
                                   ray_dir)
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0)                            # Radiance accumulator
        β = mi.Spectrum(1)                            # Path throughput weight
        active = mi.Bool(active)                      # Active SIMD lanes
        si = mi.SurfaceInteraction3f(endpoint_e)

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        # Record the following loop in its entirety
        loop = mi.Loop(name="Trace emitter subpath for primary boundary",
                       state=lambda: (sampler, ray, depth, L, β, active, si,
                                      prev_si, prev_bsdf_pdf, prev_bsdf_delta))
        loop.set_max_iterations(self.max_depth)
        while loop(active):
            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)
            
            # ---------------------- Direct emission ----------------------
            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            mis = common.mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )
            Le = β * mis * ds.emitter.eval(si)
            
            # ---------------------- Emitter sampling ----------------------
            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()
            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)
            # Evaluate BSDF * cos(theta)
            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            mis_em = dr.select(ds.delta, 1, common.mis_weight(ds.pdf, bsdf_pdf_em))
            Lr_dir = β * mis_em * bsdf_value_em * em_weight

            # ------------------ Detached BSDF sampling -------------------
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)
            
            # ---- Update loop variables based on current interaction -----
            L = L + Le + Lr_dir
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            β *= bsdf_weight

            # Information about the current vertex needed by the next iteration
            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------
            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            depth[si.is_valid()] += 1
            active = active_next
            si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0), active=active)

        return [L]

mi.register_integrator("psdr_pixel", lambda props: PathSpacePixelIntegrator(props))
