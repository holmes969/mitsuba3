from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
import numpy as np
import common

class PathSpaceIndirectIntegrator(common.PSIntegratorBoundary):
    def __init__(self, props):
        super().__init__(props)
    
    def eval_boundary_segment(
        self,
        edge_sample,
        si_0,       # vertex not on emitter
        si_1,       # vertex on emitter
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
            baseVal = (dist1 / dist) * (sinphi / sinphi2) * cos2 * dr.select(dr.eq(sign0, sign1), 1.0, -1.0)
        # differential component
        x_dot_n = dr.dot(n, si_1.p)
        return baseVal * x_dot_n / edge_sample.pdf, active

    def sample_non_boundary_edge_dir(self, n0, n1, rnd):
        phi0 = dr.acos(dr.dot(n0, n1))
        z = dr.normalize(n0 + n1)
        y = dr.normalize(dr.cross(n0, z))
        x = dr.cross(y, z)
        phi = (rnd[0] - 0.5) * phi0
        phi = dr.maximum(phi, -0.5 * phi0 + mi.math.EdgeEpsilon)
        phi = dr.minimum(phi, 0.5 * phi0 - mi.math.EdgeEpsilon)
        x1 = x * dr.cos(phi) + z * dr.sin(phi)
        b = 4.0 * rnd[1] - dr.select(rnd[1] > 0.5, 3.0, 1.0)
        a = dr.sqrt(1.0 - b * b) * dr.select(rnd[1] > 0.5, -1.0, 1.0)
        dir = a * x1 + b * y
        pdf = 0.25 / phi0
        return dir, pdf

    def sample_boundary_segment(
        self,
        scene: mi.Scene,
        sensor_id: int,
        sampler: mi.Sampler,
    ):
        sensor = scene.sensors()[sensor_id]
        film = sensor.film()
        em = scene.edge_manager()
        # sample point on geometric edge
        edge_sample = scene.sample_edge_point(sampler.next_1d(), mi.BoundaryFlags.Indirect)
        # np.savetxt("sampleP_ori.xyz", edge_sample.p.numpy(), delimiter=" ", fmt="%.6f")

        # sample a direction
        is_boundary = dr.gather(mi.Bool, em.boundary, edge_sample.idx)
        n0 = dr.gather(mi.Normal3f, em.n0, edge_sample.idx)
        n1 = dr.gather(mi.Normal3f, em.n1, edge_sample.idx, ~is_boundary)
        dir0_local = mi.warp.square_to_uniform_sphere(sampler.next_2d())
        dir0_pdf = mi.warp.square_to_uniform_sphere_pdf(dir0_local)
        dir0 = mi.Frame3f(n0).to_world(dir0_local)                                      # for boundary edge
        dir1, dir1_pdf = self.sample_non_boundary_edge_dir(n0, n1, sampler.next_2d())   # for non-boundary edge
        dir = dr.select(is_boundary, dir0, dir1)
        dir_pdf = dr.select(is_boundary, dir0_pdf, dir1_pdf)
        edge_sample.pdf *= dr.detach(dir_pdf)

        tmp_si = dr.zeros(mi.Interaction3f)
        tmp_si.p = edge_sample.p
        # sensor-side end point of the boundary segment
        tmp_si.n = -dir
        tmp_ray = tmp_si.spawn_ray(-dir)
        pi = scene.ray_intersect_preliminary(tmp_ray, coherent=False)
        endpoint_s = pi.compute_surface_interaction(tmp_ray, mi.RayFlags.FollowShape | mi.RayFlags.All)
        active = endpoint_s.is_valid()
        # emitter-side end point of the boundary segment
        tmp_si.n = dir
        tmp_ray = tmp_si.spawn_ray(dir)
        pi = scene.ray_intersect_preliminary(tmp_ray, coherent=False, active=active)
        diff_ray = mi.Ray3f(endpoint_s.p, dr.normalize(edge_sample.p - endpoint_s.p))
        endpoint_e = pi.compute_surface_interaction(diff_ray, mi.RayFlags.PathSpace | mi.RayFlags.All, active)
        active &= endpoint_e.is_valid()

        # index = dr.compress(active)
        # np.savetxt("sampleP.xyz", dr.gather(mi.Point3f, edge_sample.p, index).numpy(), delimiter=" ", fmt="%.6f")
        # np.savetxt("lightEndP.xyz", dr.gather(mi.Point3f, endpoint_e.p, index).numpy(), delimiter=" ", fmt="%.6f")
        # np.savetxt("sensorEndP.xyz", dr.gather(mi.Point3f, endpoint_s.p, index).numpy(), delimiter=" ", fmt="%.6f")

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
        # per-bounce importance & screen-space coordinates
        weights = []
        cam_pos = []
        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext(mi.TransportMode.Importance)
        film = sensor.film()

        # --------------------- Configure loop state ----------------------
        ray_dir = dr.normalize(endpoint_s.p - edge_sample.p)
        ray_org = edge_sample.p + mi.math.ShadowEpsilon * ray_dir
        ray = mi.Ray3f(ray_org, ray_dir)
        β = mi.Spectrum(1)                            # Path throughput weight
        active = mi.Bool(active)                      # Active SIMD lanes
        si = mi.SurfaceInteraction3f(endpoint_s)

        # CZ: Will drjit.loop (with drjit.scatter_reduce) a better implementation?
        max_trace = self.max_depth - 2
        for scalar_depth in range(max_trace):
            bsdf = si.bsdf(ray)
            # Connect to sensor
            ds, cam_imp = sensor.sample_direction(si, mi.Point2f(), active)
            sensor_ray = si.spawn_ray_to(ds.p)
            active_s = active & (~scene.ray_test(sensor_ray)) & (ds.pdf > 0.0)
            local_d = si.to_local(sensor_ray.d)
            # Prevent light leak
            wi_dot_geo_n = dr.dot(si.n, si.to_world(si.wi))
            wo_dot_geo_n = dr.dot(si.n, sensor_ray.d)
            wi_dot_sh_n = mi.Frame3f.cos_theta(si.wi)
            wo_dot_sh_n = mi.Frame3f.cos_theta(local_d)
            valid = (wi_dot_geo_n * wi_dot_sh_n > 0.0) & (wo_dot_geo_n * wo_dot_sh_n > 0.0)
            # Correction term in Veach's thesis
            correction = dr.select(valid,
                                   dr.abs(wi_dot_sh_n * wo_dot_geo_n / (wo_dot_sh_n * wi_dot_geo_n)),
                                   0.0)
            bsdf_val, _ = bsdf.eval_pdf(bsdf_ctx, si, local_d, active_s)
            weight = β * bsdf_val * correction * cam_imp
            weights.append(weight)
            cam_pos.append(ds.uv + film.crop_offset())
            # Keep tracing to next si
            if scalar_depth < max_trace - 1:
                bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                       sampler.next_1d(),
                                                       sampler.next_2d(),
                                                       active)
                wo = si.to_world(bsdf_sample.wo)
                wo_dot_geo_n = dr.dot(si.n, wo)
                wo_dot_sh_n = mi.Frame3f.cos_theta(bsdf_sample.wo)
                valid = (wi_dot_geo_n * wi_dot_sh_n > 0.0) & (wo_dot_geo_n * wo_dot_sh_n > 0.0)
                correction = dr.select(valid,
                                       dr.abs(wi_dot_sh_n * wo_dot_geo_n / (wo_dot_sh_n * wi_dot_geo_n)),
                                       0.0)
                β *= bsdf_weight * correction
                ray = si.spawn_ray(wo)

                # -------------------- Stopping criterion ---------------------
                si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=mi.Bool(False), active=active)
                active &= si.is_valid() & dr.neq(dr.max(β), 0)
                # Check if we still need to trace
                if dr.none(active):
                    break
        return weights, cam_pos

    def sample_emitter_subpath(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        edge_sample,
        endpoint_e,
        active: mi.Bool
    ):
        return [1.0]
        # per-bounce radiance
        weights = []
        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------
        ray_dir = dr.normalize(endpoint_e.p - edge_sample.p)
        ray_org = edge_sample.p + mi.math.ShadowEpsilon * ray_dir
        ray = mi.Ray3f(ray_org, ray_dir)
        β = mi.Spectrum(1)                            # Path throughput weight
        active = mi.Bool(active)                      # Active SIMD lanes
        si = mi.SurfaceInteraction3f(endpoint_e)
        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        L = 0.0

        max_trace = self.max_depth - 1
        # CZ: Will drjit.loop (with drjit.scatter_reduce) a better implementation?
        for scalar_depth in range(max_trace):
            bsdf = si.bsdf(ray)
            if scalar_depth > 0:
                 # Compute MIS weight for emitter sample from previous bounce
                ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
                mis = common.mis_weight(
                    prev_bsdf_pdf,
                    scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
                )
                L += β * mis * ds.emitter.eval(si)

            weights.append(L)
            
            # ---------------------- Emitter sampling ----------------------
            # Should we continue tracing to reach one more vertex?
            active &= scalar_depth + 1 < max_trace
            # Is emitter sampling even possible on the current vertex?
            active_em = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)
            # Evaluate BSDF * cos(theta)
            wo = si.to_local(ds.d)
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            mis_em = dr.select(ds.delta, 1, common.mis_weight(ds.pdf, bsdf_pdf_em))
            L += β * mis_em * bsdf_value_em * em_weight

            # ------------------ Detached BSDF sampling -------------------
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
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

        return weights

mi.register_integrator("psdr_indirect", lambda props: PathSpaceIndirectIntegrator(props))
