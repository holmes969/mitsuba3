from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
import numpy as np
import common


class PathSpacePrimaryIntegrator(common.PSIntegratorBoundary):
    def __init__(self, props):
        super().__init__(props)
    
    def sample_boundary_segment(
        self,
        scene: mi.Scene,
        sensor_id: int,
        sampler: mi.Sampler,
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Bool]:
        sensor = scene.sensors()[sensor_id]
        film = sensor.film()
        rfilter = film.rfilter()
        if not rfilter.is_box_filter():
            raise Exception("Currently, only box filter is supported for primary boundary term.")
        # sample edge rays
        edge_sample = scene.sample_edge_ray(sampler.next_1d(), sampler.next_2d(), mi.BoundaryFlags.Primary, sensor_id)
        cam_pos = sensor.world_transform().translation()    # we may want to detach?
        ray = mi.RayDifferential3f(edge_sample.p, edge_sample.d)
        # trace towards sensor
        with dr.suspend_grad():
            # connect edge_sample.p to sensor
            it = dr.zeros(mi.Interaction3f)
            it.p = edge_sample.p
            ds, cam_imp = sensor.sample_direction(it, mi.Point2f())
            pos = ds.uv + film.crop_offset()
            # compute the throughput
            dist2 = dr.squared_norm(edge_sample.p - cam_pos)
            weight = cam_imp * dist2  / edge_sample.pdf
            active = dr.neq(ds.pdf, 0.0)        
        si = scene.ray_intersect(ray, mi.RayFlags.All, coherent=False, active=active)
        active &= si.is_valid()
        with dr.suspend_grad():
            ray_dir = si.p - cam_pos
            dist = dr.norm(ray_dir)
            ray_dir /= dist
            dist1 = dr.norm(edge_sample.p - cam_pos)
            cos2 = dr.abs_dot(si.n, -ray_dir)
            e = dr.cross(edge_sample.e, ray_dir)
            sinphi = dr.norm(e)
            proj = dr.normalize(dr.cross(e, si.n))
            sinphi2 = dr.norm(dr.cross(ray_dir, proj))
            n = dr.normalize(dr.cross(si.n, proj))
            sign0 = dr.dot(e, edge_sample.e2) > 0.0
            sign1 = dr.dot(e, n) > 0.0
            active &= (sinphi > 1e-4) & (sinphi2 > 1e-4)
            baseVal = (dist / dist1) * (sinphi / sinphi2) * cos2 * dr.select(dr.eq(sign0, sign1), 1.0, -1.0)
        # Important: the only differential component in boundary term
        x_dot_n = dr.dot(n, si.p)  # this is wrong for now (to be fixed later)
        weight *= (baseVal * x_dot_n) & active
        return ray, weight, pos, active

    def sample_sensor_subpath(
        self,
        ray: mi.RayDifferential3f,
        scene: mi.Scene,
        sesor: mi.Sensor
    ):
        return [mi.Spectrum(1.0)], [None]

    def sample_emitter_subpath(
        self,
        ray: mi.RayDifferential3f,
        scene: mi.Scene
    ):
        return [mi.Spectrum(1.0)]

mi.register_integrator("psdr_primary", lambda props: PathSpacePrimaryIntegrator(props))
