#include <mitsuba/render/sensor.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/bbox.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-perspective:

Perspective pinhole camera (:monosp:`perspective`)
--------------------------------------------------

.. pluginparameters::
 :extra-rows: 7

 * - to_world
   - |transform|
   - Specifies an optional camera-to-world transformation.
     (Default: none (i.e. camera space = world space))
   - |exposed|, |differentiable|, |discontinuous|

 * - fov
   - |float|
   - Denotes the camera's field of view in degrees---must be between 0 and 180,
     excluding the extremes. Alternatively, it is also possible to specify a
     field of view using the :monosp:`focal_length` parameter.

 * - focal_length
   - |string|
   - Denotes the camera's focal length specified using *35mm* film
     equivalent units. Alternatively, it is also possible to specify a field of
     view using the :monosp:`fov` parameter. See the main description for further
     details. (Default: :monosp:`50mm`)

 * - fov_axis
   - |string|
   - When the parameter :monosp:`fov` is given (and only then), this parameter further specifies
     the image axis, to which it applies.

     1. :monosp:`x`: :monosp:`fov` maps to the :monosp:`x`-axis in screen space.
     2. :monosp:`y`: :monosp:`fov` maps to the :monosp:`y`-axis in screen space.
     3. :monosp:`diagonal`: :monosp:`fov` maps to the screen diagonal.
     4. :monosp:`smaller`: :monosp:`fov` maps to the smaller dimension
        (e.g. :monosp:`x` when :monosp:`width` < :monosp:`height`)
     5. :monosp:`larger`: :monosp:`fov` maps to the larger dimension
        (e.g. :monosp:`y` when :monosp:`width` < :monosp:`height`)

     The default is :monosp:`x`.

 * - near_clip, far_clip
   - |float|
   - Distance to the near/far clip planes. (Default: :monosp:`near_clip=1e-2` (i.e. :monosp:`0.01`)
     and :monosp:`far_clip=1e4` (i.e. :monosp:`10000`))
   - |exposed|

 * - principal_point_offset_x, principal_point_offset_y
   - |float|
   - Specifies the position of the camera's principal point relative to the center of the film.

 * - srf
   - |spectrum|
   - Sensor Response Function that defines the :ref:`spectral sensitivity <explanation_srf_sensor>`
     of the sensor (Default: :monosp:`none`)

 * - x_fov
   - |float|
   - Denotes the camera's field of view in degrees along the horizontal axis.
   - |exposed|, |differentiable|, |discontinuous|

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/sensor_perspective.jpg
   :caption: The material test ball viewed through a perspective pinhole camera. (:monosp:`fov=28`)
.. subfigure:: ../../resources/data/docs/images/render/sensor_perspective_large_fov.jpg
   :caption: The material test ball viewed through a perspective pinhole camera. (:monosp:`fov=40`)
.. subfigend::
   :label: fig-perspective

This plugin implements a simple idealized perspective camera model, which
has an infinitely small aperture. This creates an infinite depth of field,
i.e. no optical blurring occurs.

By default, the camera's field of view is specified using a 35mm film
equivalent focal length, which is first converted into a diagonal field
of view and subsequently applied to the camera. This assumes that
the film's aspect ratio matches that of 35mm film (1.5:1), though the
parameter still behaves intuitively when this is not the case.
Alternatively, it is also possible to specify a field of view in degrees
along a given axis (see the :monosp:`fov` and :monosp:`fov_axis` parameters).

The exact camera position and orientation is most easily expressed using the
:monosp:`look_at` tag, i.e.:

.. tabs::
    .. code-tab:: xml
        :name: perspective-sensor

        <sensor type="perspective">
            <float name="fov" value="45"/>
            <transform name="to_world">
                <!-- Move and rotate the camera so that looks from (1, 1, 1) to (1, 2, 1)
                    and the direction (0, 0, 1) points "up" in the output image -->
                <look_at origin="1, 1, 1" target="1, 2, 1" up="0, 0, 1"/>
            </transform>
            <!-- film -->
            <!-- sampler -->
        </sensor>

    .. code-tab:: python

        'type': 'perspective',
        'fov': 45,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[1, 1, 1],
            target=[1, 2, 1],
            up=[0, 0, 1]
        ),
        'film_id': {
            'type': '<film_type>',
            # ...
        },
        'sampler_id': {
            'type': '<sampler_type>',
            # ...
        }

 */

template <typename Float, typename Spectrum>
class PerspectiveCamera final : public ProjectiveCamera<Float, Spectrum> {
public:
    MI_IMPORT_BASE(ProjectiveCamera, m_to_world, m_needs_sample_3,
                   m_film, m_sampler, m_resolution, m_shutter_open,
                   m_shutter_open_time, m_near_clip, m_far_clip,
                   sample_wavelengths)
    MI_IMPORT_TYPES()

    PerspectiveCamera(const Properties &props) : Base(props) {
        ScalarVector2i size = m_film->size();
        m_x_fov = (ScalarFloat) parse_fov(props, size.x() / (double) size.y());

        if (m_to_world.scalar().has_scale())
            Throw("Scale factors in the camera-to-world transformation are not allowed!");

        m_principal_point_offset = ScalarPoint2f(
            props.get<ScalarFloat>("principal_point_offset_x", 0.f),
            props.get<ScalarFloat>("principal_point_offset_y", 0.f)
        );

        update_camera_transforms();
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("x_fov",     m_x_fov,          ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_parameter("to_world", *m_to_world.ptr(), ParamFlags::Differentiable | ParamFlags::Discontinuous);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        Base::parameters_changed(keys);
        if (keys.empty() || string::contains(keys, "to_world")) {
            if (m_to_world.scalar().has_scale())
                Throw("Scale factors in the camera-to-world transformation are not allowed!");
        }

        update_camera_transforms();
    }

    void update_camera_transforms() {
        m_camera_to_sample = perspective_projection(
            m_film->size(), m_film->crop_size(), m_film->crop_offset(),
            m_x_fov, Float(m_near_clip), Float(m_far_clip));

        m_sample_to_camera = m_camera_to_sample.inverse();

        // Position differentials on the near plane
        m_dx = m_sample_to_camera * Point3f(1.f / m_resolution.x(), 0.f, 0.f) -
               m_sample_to_camera * Point3f(0.f);
        m_dy = m_sample_to_camera * Point3f(0.f, 1.f / m_resolution.y(), 0.f)
             - m_sample_to_camera * Point3f(0.f);

        /* Precompute some data for importance(). Please
           look at that function for further details. */
        Point3f pmin(m_sample_to_camera * Point3f(0.f, 0.f, 0.f)),
                pmax(m_sample_to_camera * Point3f(1.f, 1.f, 0.f));

        m_image_rect.reset();
        m_image_rect.expand(Point2f(pmin.x(), pmin.y()) / pmin.z());
        m_image_rect.expand(Point2f(pmax.x(), pmax.y()) / pmax.z());
        m_normalization = 1.f / m_image_rect.volume();
        m_needs_sample_3 = false;

        dr::make_opaque(m_camera_to_sample, m_sample_to_camera, m_dx, m_dy, m_x_fov,
                        m_image_rect, m_normalization, m_principal_point_offset);
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &position_sample,
                                          const Point2f & /*aperture_sample*/,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] =
            sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
                               wavelength_sample,
                               active);
        Ray3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        Vector2f scaled_principal_point_offset =
            m_film->size() * m_principal_point_offset / m_film->crop_size();

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                         Point3f(position_sample.x() + scaled_principal_point_offset.x(),
                                 position_sample.y() + scaled_principal_point_offset.y(),
                                 0.f);

        // Convert into a normalized ray direction; adjust the ray interval accordingly.
        Vector3f d = dr::normalize(Vector3f(near_p));

        ray.o = m_to_world.value().translation();
        ray.d = m_to_world.value() * d;

        Float inv_z = dr::rcp(d.z());
        Float near_t = m_near_clip * inv_z,
              far_t  = m_far_clip * inv_z;
        ray.o += ray.d * near_t;
        ray.maxt = far_t - near_t;

        return { ray, wav_weight };
    }

    std::pair<RayDifferential3f, Spectrum>
    sample_ray_differential(Float time, Float wavelength_sample, const Point2f &position_sample,
                            const Point2f & /*aperture_sample*/, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] =
            sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
                               wavelength_sample,
                               active);
        RayDifferential3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        Vector2f scaled_principal_point_offset =
            m_film->size() * m_principal_point_offset / m_film->crop_size();

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                         Point3f(position_sample.x() + scaled_principal_point_offset.x(),
                                 position_sample.y() + scaled_principal_point_offset.y(),
                                 0.f);

        // Convert into a normalized ray direction; adjust the ray interval accordingly.
        Vector3f d = dr::normalize(Vector3f(near_p));

        ray.o = m_to_world.value().translation();
        ray.d = m_to_world.value() * d;

        Float inv_z = dr::rcp(d.z());
        Float near_t = m_near_clip * inv_z,
              far_t  = m_far_clip * inv_z;
        ray.o += ray.d * near_t;
        ray.maxt = far_t - near_t;

        ray.o_x = ray.o_y = ray.o;

        ray.d_x = m_to_world.value() * dr::normalize(Vector3f(near_p) + m_dx);
        ray.d_y = m_to_world.value() * dr::normalize(Vector3f(near_p) + m_dy);
        ray.has_differentials = true;

        return { ray, wav_weight };
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f & /*sample*/,
                     Mask active) const override {
        // Transform the reference point into the local coordinate system
        Transform4f trafo = m_to_world.value();
        Point3f ref_p     = trafo.inverse().transform_affine(it.p);

        // Check if it is outside of the clip range
        DirectionSample3f ds = dr::zeros<DirectionSample3f>();
        ds.pdf = 0.f;
        active &= (ref_p.z() >= m_near_clip) && (ref_p.z() <= m_far_clip);
        if (dr::none_or<false>(active))
            return { ds, dr::zeros<Spectrum>() };

        Vector2f scaled_principal_point_offset =
            m_film->size() * m_principal_point_offset / m_film->crop_size();

        Point3f screen_sample = m_camera_to_sample * ref_p;
        ds.uv = Point2f(screen_sample.x() - scaled_principal_point_offset.x(),
                        screen_sample.y() - scaled_principal_point_offset.y());
        active &= (ds.uv.x() >= 0) && (ds.uv.x() <= 1) && (ds.uv.y() >= 0) &&
                  (ds.uv.y() <= 1);
        if (dr::none_or<false>(active))
            return { ds, dr::zeros<Spectrum>() };

        ds.uv *= m_resolution;

        Vector3f local_d(ref_p);
        Float dist     = dr::norm(local_d);
        Float inv_dist = dr::rcp(dist);
        local_d *= inv_dist;

        ds.p    = trafo.transform_affine(Point3f(0.0f));
        ds.d    = (ds.p - it.p) * inv_dist;
        ds.dist = dist;
        ds.n    = trafo * Vector3f(0.0f, 0.0f, 1.0f);
        ds.pdf  = dr::select(active, Float(1.f), Float(0.f));

        return { ds, Spectrum(importance(local_d) * inv_dist * inv_dist) };
    }

    Float augment_antithetic_samples(Float sample, int num_antithetic) const {
        assert (num_antithetic == 4);
        if constexpr (dr::is_jit_v<Float>) {
            size_t sz = sample.size();
            assert(sz % 4 == 0);
            UInt32 index = dr::arange<UInt32>(0, sz, 4);
            Float _sample = dr::gather<Float>(sample, index);
            Float sample_1 = dr::select(_sample < 0.5, _sample + 0.5, _sample - 0.5);
            Float sample_2 = dr::select(_sample < 0.75, 0.75 - _sample, 1.75 - _sample);
            Float sample_3 = dr::select(_sample < 0.25, 0.25 - _sample, 1.25 - _sample);
            dr::scatter(sample, sample_1, index + 1);
            dr::scatter(sample, sample_2, index + 2);
            dr::scatter(sample, sample_3, index + 3);
            return sample;
        } else {
            return sample;
        }
    }

    void sample_pixel_boundary(Float sample1, EdgeSample<Float>& es) const override {
        ScalarVector2u film_size = m_film->crop_size();
        assert(!film->sample_border());
        if constexpr (dr::is_jit_v<Float>) {
            Float sample_antithetic = augment_antithetic_samples(sample1, 4);
            size_t wavefront_size = sample_antithetic.size();
            assert(wavefront_size <= 0xffffffffu);
            size_t num_pixels = (size_t) film_size.x() * (size_t) film_size.y();
            assert(wavefront_size % num_pixels == 0);
            size_t spp = wavefront_size / num_pixels;
            // Compute discrete sample position
            UInt32 idx = dr::arange<UInt32>((uint32_t) wavefront_size);
            // Try to avoid a division by an unknown constant if we can help it
            uint32_t log_spp = dr::log2i(spp);
            if ((1u << log_spp) == spp)
                idx >>= dr::opaque<UInt32>(log_spp);
            else
                idx /= dr::opaque<UInt32>(spp);
            // Compute the position on the image plane
            Vector2u pos;
            pos.y() = idx / film_size[0];
            pos.x() = dr::fnmadd(film_size[0], pos.y(), idx);
            pos += m_film->crop_offset();
            sample_antithetic *= 4.0;
            Int32 index_boundary = dr::floor(sample_antithetic);
            auto left = dr::eq(index_boundary, 0);
            auto top = dr::eq(index_boundary, 1);
            auto right = dr::eq(index_boundary, 2);
            auto bottom = dr::eq(index_boundary, 3);
            Point2f pos_offset = dr::zeros<Point2f>(wavefront_size);
            pos_offset.x()[top] = sample_antithetic - 1.0;
            pos_offset.x()[right] = 1.0f;
            pos_offset.x()[bottom] = 4.0 - sample_antithetic;
            pos_offset.y()[left] = sample_antithetic;
            pos_offset.y()[top] = 1.0f;
            pos_offset.y()[right] = 3.0 - sample_antithetic;
            Vector2f sample_pos = pos + pos_offset;
            ScalarVector2f scale = 1.f / film_size,
                           offset = -ScalarVector2f(m_film->crop_offset()) * scale;
            Vector2f adjusted_pos = dr::fmadd(sample_pos, scale, offset);   // [0, 1]
            Vector2f scaled_principal_point_offset = m_film->size() * m_principal_point_offset / film_size;
            // Compute the sample position on the near plane (local camera space).
            Point3f near_p = m_sample_to_camera *
                            Point3f(adjusted_pos.x() + scaled_principal_point_offset.x(),
                                    adjusted_pos.y() + scaled_principal_point_offset.y(),
                                    0.f);
            es.p = m_to_world.value() * near_p;

            Vector3f e = dr::zeros<Vector3f>(wavefront_size);
            e.x()[top || bottom] = 1.0;
            e.y()[left || right] = 1.0;
            es.e = dr::normalize(m_to_world.value() * m_sample_to_camera * e);

            Vector3f e2 = dr::zeros<Vector3f>(wavefront_size);
            e2.x()[left] = 1.0;
            e2.x()[right] = -1.0;
            e2.y()[top] = -1.0;
            e2.y()[bottom] = 1.0;
            es.e2 = dr::normalize(m_to_world.value() * m_sample_to_camera * e2);

            Float cam_img_len_x = (m_image_rect.max.x() - m_image_rect.min.x()) * m_near_clip;
            Float cam_img_len_y = (m_image_rect.max.y() - m_image_rect.min.y()) * m_near_clip;
            es.pdf = dr::empty<Float>(wavefront_size);
            es.pdf[left || right] = 0.25 / (cam_img_len_y * film_size[0]);
            es.pdf[top || bottom] = 0.25 / (cam_img_len_x * film_size[1]);
        }
    }


    ScalarBoundingBox3f bbox() const override {
        ScalarPoint3f p = m_to_world.scalar() * ScalarPoint3f(0.f);
        return ScalarBoundingBox3f(p, p);
    }

    /**
     * \brief Compute the directional sensor response function of the camera
     * multiplied with the cosine foreshortening factor associated with the
     * image plane
     *
     * \param d
     *     A normalized direction vector from the aperture position to the
     *     reference point in question (all in local camera space)
     */
    Float importance(const Vector3f &d) const {
        /* How is this derived? Imagine a hypothetical image plane at a
           distance of d=1 away from the pinhole in camera space.

           Then the visible rectangular portion of the plane has the area

              A = (2 * dr::tan(0.5 * xfov in radians))^2 / aspect

           Since we allow crop regions, the actual visible area is
           potentially reduced:

              A' = A * (cropX / filmX) * (cropY / filmY)

           Perspective transformations of such aligned rectangles produce
           an equivalent scaled (but otherwise undistorted) rectangle
           in screen space. This means that a strategy, which uniformly
           generates samples in screen space has an associated area
           density of 1/A' on this rectangle.

           To compute the solid angle density of a sampled point P on
           the rectangle, we can apply the usual measure conversion term:

              d_omega = 1/A' * distance(P, origin)^2 / dr::cos(theta)

           where theta is the angle that the unit direction vector from
           the origin to P makes with the rectangle. Since

              distance(P, origin)^2 = Px^2 + Py^2 + 1

           and

              dr::cos(theta) = 1/sqrt(Px^2 + Py^2 + 1),

           we have

              d_omega = 1 / (A' * cos^3(theta))
        */

        Float ct     = Frame3f::cos_theta(d),
              inv_ct = dr::rcp(ct);

        // Compute the position on the plane at distance 1
        Point2f p(d.x() * inv_ct, d.y() * inv_ct);

        /* Check if the point lies to the front and inside the
           chosen crop rectangle */
        Mask valid = ct > 0 && m_image_rect.contains(p);

        return dr::select(valid, m_normalization * inv_ct * inv_ct * inv_ct, 0.f);
    }

    std::string to_string() const override {
        using string::indent;

        std::ostringstream oss;
        oss << "PerspectiveCamera[" << std::endl
            << "  x_fov = " << m_x_fov << "," << std::endl
            << "  near_clip = " << m_near_clip << "," << std::endl
            << "  far_clip = " << m_far_clip << "," << std::endl
            << "  film = " << indent(m_film) << "," << std::endl
            << "  sampler = " << indent(m_sampler) << "," << std::endl
            << "  resolution = " << m_resolution << "," << std::endl
            << "  shutter_open = " << m_shutter_open << "," << std::endl
            << "  shutter_open_time = " << m_shutter_open_time << "," << std::endl
            << "  to_world = " << indent(m_to_world, 13) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    Transform4f m_camera_to_sample;
    Transform4f m_sample_to_camera;
    BoundingBox2f m_image_rect;
    Float m_normalization;
    Float m_x_fov;
    Vector3f m_dx, m_dy;
    Vector2f m_principal_point_offset;
};

MI_IMPLEMENT_CLASS_VARIANT(PerspectiveCamera, ProjectiveCamera)
MI_EXPORT_PLUGIN(PerspectiveCamera, "Perspective Camera");
NAMESPACE_END(mitsuba)
