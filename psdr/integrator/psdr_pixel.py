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
        dir = dr.normalize(edge_sample.p - sensor_pos)

        # np.savetxt("sampleP.xyz", edge_sample.p.numpy(), delimiter=" ", fmt="%.6f")


        # sensor-side end point of the boundary segment (fixed at camera position for pixel boundary)
        endpoint_s = dr.zeros(mi.Interaction3f)
        endpoint_s.p = sensor_pos
        endpoint_s.t = 0.0

        # emitter-side end point of the boundary segment
        primary_ray = mi.Ray3f(sensor_pos, dir)
        pi = scene.ray_intersect_preliminary(primary_ray, coherent=False)
        endpoint_e = pi.compute_surface_interaction(primary_ray, mi.RayFlags.PathSpace | mi.RayFlags.All)

        # evaluate the boundary segment
        weight, active = self.eval_boundary_segment(edge_sample, endpoint_s, endpoint_e)
        index = dr.compress(active)
        # np.savetxt("sampleP.xyz", dr.gather(mi.Point3f, edge_sample.p, index).numpy(), delimiter=" ", fmt="%.6f")

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
    
    def to_antithetic(self, rnd, num_antithetic = 4):
        sz = dr.shape(rnd)[-1]
        assert(sz % 4 == 0)
        index = dr.arange(mi.UInt32, 0, sz, 4)
        _rnd = dr.gather(type(rnd), rnd, index)
        dr.scatter(rnd, _rnd, index + 1)
        dr.scatter(rnd, _rnd, index + 2)
        dr.scatter(rnd, _rnd, index + 3)
        dr.eval(rnd)    # without eval: get error "inputs remain dirty after evaluation"
        return rnd

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

        # CZ: Will drjit.loop (with drjit.scatter_reduce) a better implementation?
        for scalar_depth in range(self.max_depth):
            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            mis = common.mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )
            L += β * mis * ds.emitter.eval(si)
            
            # ---------------------- Emitter sampling ----------------------
            bsdf = si.bsdf(ray)
            # Should we continue tracing to reach one more vertex?
            active &= scalar_depth + 1 < self.max_depth
            # Is emitter sampling even possible on the current vertex?
            active_em = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            # If so, randomly sample an emitter without derivative tracking.
            # print("rnd_antithetic = ", rnd_antithetic)
            ds, em_weight = scene.sample_emitter_direction(
                si, self.to_antithetic(sampler.next_2d()), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)
            # Evaluate BSDF * cos(theta)
            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            mis_em = dr.select(ds.delta, 1, common.mis_weight(ds.pdf, bsdf_pdf_em))
            L += β * mis_em * bsdf_value_em * em_weight

            # ------------------ Detached BSDF sampling -------------------
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   self.to_antithetic(sampler.next_1d()),
                                                   self.to_antithetic(sampler.next_2d()),
                                                   active)
            
            # ---- Update loop variables based on current interaction -----
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            β *= bsdf_weight

            # Information about the current vertex needed by the next iteration
            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------
            # Don't run another iteration if the throughput has reached zero
            si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=mi.Bool(False), active=active)
            active &= si.is_valid() & dr.neq(dr.max(β), 0)

            # Check if we still need to trace
            if dr.none(active):
                break

        return [L]

mi.register_integrator("psdr_pixel", lambda props: PathSpacePixelIntegrator(props))
