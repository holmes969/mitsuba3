from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
import numpy as np
import common

class PathSpaceDirectIntegrator(common.PSIntegratorBoundary):
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
        # x_dot_n = dr.dot(n, edge_sample.p)
        return baseVal * x_dot_n / edge_sample.pdf, active
        # return edge_sample.p[0] / edge_sample.pdf, active

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
        edge_sample = scene.sample_edge_point(sampler.next_1d(), mi.BoundaryFlags.Direct)
        tmp_si = dr.zeros(mi.Interaction3f)
        tmp_si.p = edge_sample.p

        # sample point on emitter
        ds, _ = scene.sample_emitter_direction(tmp_si, sampler.next_2d(), False)
        dir = dr.normalize(ds.p - tmp_si.p)
        # check if the direction is valid
        is_boundary = dr.gather(mi.Bool, em.boundary, edge_sample.idx)
        n0 = dr.gather(mi.Normal3f, em.n0, edge_sample.idx)
        n1 = dr.gather(mi.Normal3f, em.n1, edge_sample.idx, ~is_boundary)
        active = dr.dot(ds.n, -dir) > mi.math.EdgeEpsilon

        tmp0 = dr.dot(n0, dir)
        tmp1 = dr.dot(n1, dir)
        active &= (~is_boundary) | (dr.abs(tmp0) > mi.math.EdgeEpsilon)
        active &= (is_boundary) | (((tmp0 >  mi.math.EdgeEpsilon) & (tmp1 < -mi.math.EdgeEpsilon)) | 
                                 ((tmp0 < -mi.math.EdgeEpsilon) & (tmp1 >  mi.math.EdgeEpsilon)) )
        tmp_si.n = dir
        active &= ~scene.ray_test(tmp_si.spawn_ray_to(ds.p))
        edge_sample.pdf *= dr.detach(ds.pdf)

        # sensor-side end point of the boundary segment
        tmp_si.n = -dir
        tmp_ray = tmp_si.spawn_ray(-dir)
        pi = scene.ray_intersect_preliminary(tmp_ray, coherent=False, active=active)
        endpoint_s = pi.compute_surface_interaction(tmp_ray, mi.RayFlags.FollowShape | mi.RayFlags.All, active)
        active &= endpoint_s.is_valid()

        # emitter-side end point of the boundary segment
        tmp_si.n = dir
        tmp_ray = tmp_si.spawn_ray(dir)
        pi = scene.ray_intersect_preliminary(tmp_ray, coherent=False, active=active)    # CZ: is it possible to avoid the redundant ray intersection?
        diff_ray = mi.Ray3f(endpoint_s.p, dr.normalize(edge_sample.p - endpoint_s.p))
        endpoint_e = pi.compute_surface_interaction(diff_ray, mi.RayFlags.PathSpace | mi.RayFlags.All, active)

        # index = dr.compress(active)
        # np.savetxt("sampleP.xyz", dr.gather(mi.Point3f, edge_sample.p, index).numpy(), delimiter=" ", fmt="%.6f")
        # np.savetxt("lightEndP.xyz", dr.gather(mi.Point3f, endpoint_e.p, index).numpy(), delimiter=" ", fmt="%.6f")
        # np.savetxt("sensorEndP.xyz", dr.gather(mi.Point3f, endpoint_s.p, index).numpy(), delimiter=" ", fmt="%.6f")

        # evaluate the boundary segment
        weight, active = self.eval_boundary_segment(edge_sample, endpoint_s, endpoint_e)
        # weight *= dr.detach(scene.eval_emitter_direction(tmp_si, ds, active))

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
        film = sensor.film()
        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext(mi.TransportMode.Importance)

        # --------------------- Configure loop state ----------------------
        ray_dir = dr.normalize(endpoint_s.p - edge_sample.p)
        ray_org = edge_sample.p + mi.math.ShadowEpsilon * ray_dir
        ray = mi.Ray3f(ray_org, ray_dir)
        depth = mi.UInt32(0)                          # Depth of current vertex
        β = mi.Spectrum(1)                            # Path throughput weight
        active = mi.Bool(active)                      # Active SIMD lanes
        si = mi.SurfaceInteraction3f(endpoint_s)

        weights = []
        cam_pos = []

        bsdf = si.bsdf(ray)

        # Connect to sensor
        ds, cam_imp = sensor.sample_direction(si, mi.Point2f(), active)
        sensor_ray = si.spawn_ray_to(ds.p)
        visible = ~scene.ray_test(sensor_ray)
        # active_s = active & visible & (ds.pdf > 0.0)
        active_s = active & visible
        active_s &= ds.pdf > 0.0
        local_d = si.to_local(sensor_ray.d)
        # Prevent light leak
        wi_dot_geo_n = dr.dot(si.n, si.to_world(si.wi))
        wo_dot_geo_n = dr.dot(si.n, sensor_ray.d)
        wi_dot_sh_n = mi.Frame3f.cos_theta(si.wi)
        wo_dot_sh_n = mi.Frame3f.cos_theta(local_d)
        valid = (wi_dot_geo_n * wi_dot_sh_n) > 0.0
        valid &= (wo_dot_geo_n * wo_dot_sh_n) > 0.0
        # Correction term in Veach's thesis
        correction = dr.select(valid, dr.abs(wi_dot_sh_n * wo_dot_geo_n / (wo_dot_sh_n * wi_dot_geo_n)), 0.0)
        bsdf_val, _ = bsdf.eval_pdf(bsdf_ctx, si, local_d, active_s)
        # weight = bsdf_val * correction * cam_imp
        weight = dr.select(active_s, 1.0, 0.0)
        weights.append(weight)
        cam_pos.append(ds.uv + film.crop_offset())


        # # Record the following loop in its entirety
        # loop = mi.Loop(name="Trace emitter subpath for primary boundary",
        #                state=lambda: (sampler, ray, depth, β, active, si))
        # loop.set_max_iterations(self.max_depth)
        # while loop(active):
        #     # Get the BSDF, potentially computes texture-space differentials
        #     bsdf = si.bsdf(ray)

        #     # Connect to sensor
        #     ds, cam_imp = sensor.sample_direction(si, mi.Point2f(), active)
        #     sensor_ray = si.spawn_ray_to(ds.p)
        #     visible = ~scene.ray_test(sensor_ray)
        #     # active_s = active & visible & (ds.pdf > 0.0)
        #     active_s = active & visible
        #     active_s &= ds.pdf > 0.0
        #     local_d = si.to_local(sensor_ray.d)
        #     # Prevent light leak
        #     wi_dot_geo_n = dr.dot(si.n, si.to_world(si.wi))
        #     wo_dot_geo_n = dr.dot(si.n, sensor_ray.d)
        #     wi_dot_sh_n = mi.Frame3f.cos_theta(si.wi)
        #     wo_dot_sh_n = mi.Frame3f.cos_theta(local_d)
        #     valid = (wi_dot_geo_n * wi_dot_sh_n) > 0.0
        #     valid &= (wo_dot_geo_n * wo_dot_sh_n) > 0.0
        #     # Correction term in Veach's thesis
        #     correction = dr.select(valid, dr.abs(wi_dot_sh_n * wo_dot_geo_n / (wo_dot_sh_n * wi_dot_geo_n)), 0.0)
        #     bsdf_val, _ = bsdf.eval_pdf(bsdf_ctx, si, local_d, active_s)
        #     weight = bsdf_val * correction * cam_imp
        #     weights.append(weight)
        #     cam_pos.append(ds.uv + film.crop_offset())

        #     # Trace to next si
        #     active_next = active & (depth + 1 < self.max_depth)
        #     bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
        #                                            sampler.next_1d(),
        #                                            sampler.next_2d(),
        #                                            active_next)
        #     wo = si.to_world(bsdf_sample.wo)
        #     wi_dot_geo_n = dr.dot(si.n, -ray.d)
        #     wo_dot_geo_n = dr.dot(si.n, wo)
        #     wo_dot_sh_n = mi.Frame3f.cos_theta(bsdf_sample.wo)
        #     valid = wi_dot_geo_n * wi_dot_sh_n > 0.0
        #     valid &= wo_dot_geo_n * wo_dot_sh_n > 0.0
        #     correction = dr.select(valid, dr.abs(wi_dot_sh_n * wo_dot_geo_n / (wo_dot_sh_n * wi_dot_geo_n)), 0.0)
        #     β *= bsdf_weight * correction
        #     ray = si.spawn_ray(wo)

        #     # -------------------- Stopping criterion ---------------------
        #     active = active_next & dr.neq(dr.max(β), 0)
        #     depth[si.is_valid()] += 1
        #     si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0), active=active)

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

mi.register_integrator("psdr_direct", lambda props: PathSpaceDirectIntegrator(props))
