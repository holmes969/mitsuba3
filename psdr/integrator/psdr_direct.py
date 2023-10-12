from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
import numpy as np
import common


class PathSpaceDirectIntegrator(common.PSIntegratorBoundary):
    def __init__(self, props):
        super().__init__(props)
    
    def sample_boundary_segment(
        self,
        scene: mi.Scene,
        sensor_id: int,
        sampler: mi.Sampler,
    ):
        sensor = scene.sensors()[sensor_id]
        film = sensor.film()

        # sample point on geometric edge
        edge_sample = scene.sample_edge_point(sampler.next_1d(), mi.BoundaryFlags.Direct)
        tmp_si = dr.zeros(mi.Interaction3f)
        tmp_si.p = edge_sample.p
        ds, em_weight = scene.sample_emitter_direction(tmp_si, sampler.next_2d(), False)
        dir = dr.normalize(ds.p - tmp_si.p)
        tmp_si.n = dir
        active = ~scene.ray_test(tmp_si.spawn_ray(dir))
        edge_sample.pdf *= ds.pdf

        # sensor-side end point of the boundary segment
        tmp_si.n = -dir
        tmp_ray = tmp_si.spawn_ray(dir)
        pi = scene.ray_intersect_preliminary(tmp_ray, coherent=False)
        tmp_ray.o = endpoint_e.p
        tmp_ray.d = dr.normalize(edge_sample.p - tmp_ray.o)
        endpoint_s = pi.compute_surface_interaction(tmp_ray, mi.RayFlags.PathSpace | mi.RayFlags.All)

        # emitter-side end point of the boundary segment
        endpoint_e = dr.zeros(mi.Interaction3f)
        endpoint_e.p = ds.p
        endpoint_e.t = dr.select(active, 0.0, dr.inf)
        
        # evaluate the boundary segment
        weight, active = self.eval_boundary_segment(edge_sample, endpoint_s, endpoint_e)
        weight *= scene.eval_emitter_direction(tmp_si, ds, active)

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
        sensor_ray = endpoint_s.spawn_ray_to(endpoint_s.p)
        sensor_ray.o = sensor_ray.o + mi.math.ShadowEpsilon * sensor_ray.d      # add shadow epsilon for better numerical stability
        active = ~scene.ray_test(sensor_ray)
        ds, cam_imp = sensor.sample_direction(endpoint_s, mi.Point2f(), active)
        film = sensor.film()
        pos = ds.uv + film.crop_offset()
        active &= dr.neq(ds.pdf, 0.0)
        weight = cam_imp & active
        return [weight], [pos]

    def sample_emitter_subpath(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        edge_sample,
        endpoint_e,
        active: mi.Bool
    ):
        return [1.0]

mi.register_integrator("psdr_primary", lambda props: PathSpaceDirectIntegrator(props))
