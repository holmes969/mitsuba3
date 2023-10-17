from __future__ import annotations as __annotations__ # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr
import gc
import numpy as np
import torch

class PSIntegrator(mi.CppADIntegrator):
    """
    Abstract base class of path-space differentiable integrators in Mitsuba

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (Default: 6)
    """

    def __init__(self, props = mi.Properties()):
        super().__init__(props)

        max_depth = props.get('max_depth', 6)
        if max_depth < 0 and max_depth != -1:
            raise Exception("\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        # Map -1 (infinity) to 2^32-1 bounces
        self.max_depth = max_depth if max_depth != -1 else 0xffffffff

    def aovs(self):
        return []

    def to_string(self):
        return f'{type(self).__name__}[max_depth = {self.max_depth}]'

    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:

        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aovs()
            )

            # Generate a set of rays starting at the sensor
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            L, valid, state = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sensor=sensor,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                active=mi.Bool(True),
            )

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            # Accumulate into the image block
            PSIntegrator._splat_to_block(
                block, film, pos,
                value=L * weight,
                weight=1.0,
                alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                wavelengths=ray.wavelengths
            )

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, pos, L, valid
            gc.collect()

            # Perform the weight division and return an image tensor
            film.put_block(block)
            self.primal_image = film.develop()

            return self.primal_image

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            with dr.resume_grad():
                with dr.scoped_set_flag(dr.JitFlag.LoopRecord, False):
                    L, valid, _ = self.sample(
                        mode=dr.ADMode.Forward,
                        scene=scene,
                        sensor=sensor,
                        sampler=sampler,
                        ray=ray,
                        active=mi.Bool(True)
                    )

                block = film.create_block()
                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)

                # Deposit samples with gradient tracking for 'pos'.
                # After reparameterizing the camera ray, we need to evaluate
                #   Σ (fi Li det)
                #  ---------------
                #   Σ (fi det)
                PSIntegrator._splat_to_block(
                    block, film, pos,
                    value=L * weight,
                    weight=1.0,
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    wavelengths=ray.wavelengths
                )

                # Perform the weight division and return an image tensor
                film.put_block(block)
                result_img = film.develop()

                dr.forward_to(result_img)

        return dr.grad(result_img)

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            # When the underlying integrator supports reparameterizations,
            # perform necessary initialization steps and wrap the result using
            # the _ReparamWrapper abstraction defined above

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            with dr.resume_grad():
                with dr.scoped_set_flag(dr.JitFlag.LoopRecord, False):
                    L, valid, _ = self.sample(
                        mode=dr.ADMode.Backward,
                        scene=scene,
                        sensor=sensor,
                        sampler=sampler,
                        ray=ray,
                        active=mi.Bool(True)
                    )

                # Prepare an ImageBlock as specified by the film
                block = film.create_block()

                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)

                # Accumulate into the image block
                PSIntegrator._splat_to_block(
                    block, film, pos,
                    value=L * weight,
                    weight=1.0,
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    wavelengths=ray.wavelengths
                )

                film.put_block(block)

                del valid
                gc.collect()

                # This step launches a kernel
                dr.schedule(block.tensor())
                image = film.develop()

                # Differentiate sample splatting and weight division steps to
                # retrieve the adjoint radiance
                dr.set_grad(image, grad_in)
                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(mi.Float, dr.ADMode.Backward)

            # We don't need any of the outputs here
            del ray, weight, pos, block, sampler
            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()

    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f]:
        """
        Sample a 2D grid of primary rays for a given sensor

        Returns a tuple containing

        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous 2D image-space positions associated with each ray
        """

        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(-film_size[0], pos.y, idx)

        if film.sample_border():
            pos -= border_size

        pos += mi.Vector2i(film.crop_offset())

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )

        # With box filter, ignore random offset to prevent numerical instabilities
        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

        return ray, weight, splatting_pos

    def prepare(self,
                sensor: mi.Sensor,
                seed: int = 0,
                spp: int = 0,
                aovs: list = []):
        """
        Given a sensor and a desired number of samples per pixel, this function
        computes the necessary number of Monte Carlo samples and then suitably
        seeds the sampler underlying the sensor.

        Returns the created sampler and the final number of samples per pixel
        (which may differ from the requested amount depending on the type of
        ``Sampler`` being used)

        Parameter ``sensor`` (``int``, ``mi.Sensor``):
            Specify a sensor to render the scene from a different viewpoint.

        Parameter ``seed` (``int``)
            This parameter controls the initialization of the random number
            generator during the primal rendering step. It is crucial that you
            specify different seeds (e.g., an increasing sequence) if subsequent
            calls should produce statistically independent images (e.g. to
            de-correlate gradient-based optimization steps).

        Parameter ``spp`` (``int``):
            Optional parameter to override the number of samples per pixel for the
            primal rendering step. The value provided within the original scene
            specification takes precedence if ``spp=0``.
        """

        film = sensor.film()
        sampler = sensor.sampler().clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        film_size = film.crop_size()

        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()

        wavefront_size = dr.prod(film_size) * spp

        if wavefront_size > 2**32:
            raise Exception(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size)

        sampler.seed(seed, wavefront_size)
        film.prepare(aovs)

        return sampler, spp

    def _splat_to_block(block: mi.ImageBlock,
                       film: mi.Film,
                       pos: mi.Point2f,
                       value: mi.Spectrum,
                       weight: mi.Float,
                       alpha: mi.Float,
                       wavelengths: mi.Spectrum,
                       active: mi.Bool = mi.Bool(True)):
        '''Helper function to splat values to a imageblock'''
        if (dr.all(mi.has_flag(film.flags(), mi.FilmFlags.Special))):
            aovs = film.prepare_sample(value, wavelengths,
                                        block.channel_count(),
                                        weight=weight,
                                        alpha=alpha)
            block.put(pos, aovs, active)
            del aovs
        else:
            block.put(
                pos=pos,
                wavelengths=wavelengths,
                value=value,
                weight=weight,
                alpha=alpha,
                active=active
            )

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sensor: mi.Sensor,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               depth: mi.UInt32,
               δL: Optional[mi.Spectrum],
               state_in: Any,
               active: mi.Bool) -> Tuple[mi.Spectrum, mi.Bool]:

        raise Exception('PSIntegrator does not provide the sample() method. '
                        'It should be implemented by subclasses that '
                        'specialize the abstract RBIntegrator interface.')
class PSIntegratorPRB(PSIntegrator):
    """
    Abstract base class of PRB-style path-space differentiable integrators.
    """

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            # Generate a set of rays starting at the sensor
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)
            
            # Launch the Monte Carlo sampling process in primal mode (1)
            L, valid, state_out = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sensor=sensor,
                sampler=sampler.clone(),
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                active=mi.Bool(True)
            )

            # Launch the Monte Carlo sampling process in forward mode (2)
            δL, valid_2, state_out_2 = self.sample(
                mode=dr.ADMode.Forward,
                scene=scene,
                sensor=sensor,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=state_out,
                active=mi.Bool(True)
            )

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            # Accumulate into the image block
            PSIntegrator._splat_to_block(
                block, film, pos,
                value=δL * weight,
                weight=1.0,
                alpha=dr.select(valid_2, mi.Float(1), mi.Float(0)),
                wavelengths=ray.wavelengths
            )

            # Perform the weight division and return an image tensor
            film.put_block(block)

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, pos, L, valid, δL, valid_2, params, \
                state_out, state_out_2, block

            # Probably a little overkill, but why not.. If there are any
            # DrJit arrays to be collected by Python's cyclic GC, then
            # freeing them may enable loop simplifications in dr.eval().
            gc.collect()
            result_grad = film.develop()

        return result_grad

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            # Generate a set of rays starting at the sensor
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            def splatting_and_backward_gradient_image(value: mi.Spectrum,
                                                      weight: mi.Float,
                                                      alpha: mi.Float):
                '''
                Backward propagation of the gradient image through the sample
                splatting and weight division steps.
                '''

                # Prepare an ImageBlock as specified by the film
                block = film.create_block()

                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)

                PSIntegrator._splat_to_block(
                    block, film, pos,
                    value=value,
                    weight=weight,
                    alpha=alpha,
                    wavelengths=ray.wavelengths
                )

                film.put_block(block)

                # Probably a little overkill, but why not.. If there are any
                # DrJit arrays to be collected by Python's cyclic GC, then
                # freeing them may enable loop simplifications in dr.eval().
                gc.collect()

                image = film.develop()

                dr.set_grad(image, grad_in)
                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(mi.Float, dr.ADMode.Backward)

            # Differentiate sample splatting and weight division steps to
            # retrieve the adjoint radiance (e.g. 'δL')
            with dr.resume_grad():
                with dr.suspend_grad(pos, ray, weight):
                    L = dr.full(mi.Spectrum, 1.0, dr.width(ray))
                    dr.enable_grad(L)

                    splatting_and_backward_gradient_image(
                        value=L * weight,
                        weight=1.0,
                        alpha=1.0
                    )

                    δL = dr.grad(L)

            # Clear the dummy data splatted on the film above
            film.clear()

            # Launch the Monte Carlo sampling process in primal mode (1)
            L, valid, state_out = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sensor=sensor,
                sampler=sampler.clone(),
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                active=mi.Bool(True)
            )

            # Launch Monte Carlo sampling in backward AD mode (2)
            L_2, valid_2, state_out_2 = self.sample(
                mode=dr.ADMode.Backward,
                scene=scene,
                sensor=sensor,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=δL,
                state_in=state_out,
                active=mi.Bool(True)
            )

            # We don't need any of the outputs here
            del L_2, valid_2, state_out, state_out_2, δL, \
                ray, weight, pos, sampler

            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()

class PSIntegratorBoundary(PSIntegrator):

    def __init__(self, props = mi.Properties()):
        super().__init__(props)
        max_depth = props.get('max_depth', 6)
        if max_depth < 0 and max_depth != -1:
            raise Exception("\"max_depth\" must be set to -1 (infinite) or a value >= 0")
        # Map -1 (infinity) to 2^32-1 bounces
        self.max_depth = max_depth if max_depth != -1 else 0xffffffff
    
    def sample_boundary_segment(
        self,
        scene: mi.Scene,
        sensor: int,
        sampler: mi.Sampler,
    ):
        raise Exception('PSIntegratorBoundary does not provide the sample_boundary_segment() method.')
    
    def eval_boundary_segment(
        self,
        edge_sample,
        si_0,
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
            active &= (sinphi > 1e-6) & (sinphi2 > 1e-6)
            baseVal = (dist / dist1) * (sinphi / sinphi2) * cos2 * dr.select(dr.eq(sign0, sign1), 1.0, -1.0)
        # differential component
        x_dot_n = dr.dot(n, si_1.p)
        return baseVal * x_dot_n / edge_sample.pdf, active

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
        raise Exception('PSIntegratorBoundary does not provide the trace_sensor_subpath() method.')

    def sample_emitter_subpath(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        edge_sample,
        endpoint_e,
        active: mi.Bool,
    ):
        raise Exception('PSIntegratorBoundary does not provide the trace_emitter_subpath() method.')
    
    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor_id: int = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:
        raise Exception('PSIntegratorBoundary does not provide the render() method.')
    
    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor_id: int = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        sensor = scene.sensors()[sensor_id]
        film = sensor.film()
        aovs = self.aovs()
        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor, seed, spp, aovs)
            with dr.resume_grad():
                edge_sample, endpoint_s, endpoint_e, bseg_weight, active = self.sample_boundary_segment(scene, sensor_id, sampler)
            weight_s, pos = self.sample_sensor_subpath(scene, sampler, edge_sample, endpoint_s, endpoint_e, sensor, active)
            weight_e = self.sample_emitter_subpath(scene, sampler, edge_sample, endpoint_e, active)
            block = film.create_block()
            block.set_coalesce(False)
            res = weight_e[0] / spp
            PSIntegrator._splat_to_block(
                block, film, pos[0],
                value=res,
                weight=0.0,         # avoid division by weights
                alpha=dr.select(active, mi.Float(1), mi.Float(0)),
                wavelengths=[],
                active=active
            )
            film.put_block(block)

        result_img = film.develop()
        return result_img
        # dr.forward_to(result_img)

            # weight_e = self.sample_emitter_subpath(scene, sampler, edge_sample, endpoint_e, active)
            # with dr.resume_grad():
            #     len_s = len(weight_s)
            #     len_e = len(weight_e)
            #     for idx_s in range(len_s):
            #         idx_e = min(len_e, self.max_depth) - 1
            #         block = film.create_block()
            #         block.set_coalesce(False)
            #         res = bseg_weight * weight_s[idx_s] * weight_e[idx_e] / spp
            #         PSIntegrator._splat_to_block(
            #             block, film, pos[idx_s],
            #             value=res,
            #             weight=0.0,         # avoid division by weights
            #             alpha=dr.select(active, mi.Float(1), mi.Float(0)),
            #             wavelengths=[],
            #             active=active
            #         )
            #         film.put_block(block)
            #     result_img = film.develop()
            #     dr.forward_to(result_img)
        return dr.grad(result_img)
    
    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor_id: int = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:
        sensor = scene.sensors()[sensor_id]
        film = sensor.film()
        aovs = self.aovs()
        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor, seed, spp, aovs)
            with dr.resume_grad():
                edge_sample, endpoint_s, endpoint_e, bseg_weight, active = self.sample_boundary_segment(scene, sensor_id, sampler)
            weight_s, pos = self.sample_sensor_subpath(scene, sampler, edge_sample, endpoint_s, endpoint_e, sensor, active)
            weight_e = self.sample_emitter_subpath(scene, sampler, edge_sample, endpoint_e, active)
            with dr.resume_grad():
                len_s = len(weight_s)
                len_e = len(weight_e)
                for idx_s in range(len_s):
                    idx_e = min(len_e, self.max_depth) - 1
                    block = film.create_block()
                    block.set_coalesce(False)
                    res = bseg_weight * weight_s[idx_s] * weight_e[idx_e] / spp
                    PSIntegrator._splat_to_block(
                        block, film, pos[idx_s],
                        value=res,
                        weight=0.0,         # avoid division by weights
                        alpha=dr.select(active, mi.Float(1), mi.Float(0)),
                        wavelengths=[],
                        active=active
                    )
                    film.put_block(block)
        
                # This step launches a kernel
                dr.schedule(block.tensor())                    
                image = film.develop()

                # Differentiate sample splatting and weight division steps to
                # retrieve the adjoint radiance
                dr.set_grad(image, grad_in)
                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(mi.Float, dr.ADMode.Backward)

            # We don't need any of the outputs here
            del edge_sample, endpoint_s, endpoint_e, bseg_weight, weight_s, weight_e, pos, block, sampler
            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()
    
def mis_weight(pdf_a, pdf_b):
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    b2 = dr.sqr(pdf_b)
    w = a2 / (a2 + b2)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))