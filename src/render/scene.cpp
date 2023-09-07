#include <mitsuba/core/properties.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/integrator.h>

#if defined(MI_ENABLE_EMBREE)
#  include "scene_embree.inl"
#else
#  include <mitsuba/render/kdtree.h>
#  include "scene_native.inl"
#endif

#if defined(MI_ENABLE_CUDA)
#  include "scene_optix.inl"
#endif

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT Scene<Float, Spectrum>::Scene(const Properties &props) {
    for (auto &[k, v] : props.objects()) {
        Scene *scene           = dynamic_cast<Scene *>(v.get());
        Shape *shape           = dynamic_cast<Shape *>(v.get());
        Mesh *mesh             = dynamic_cast<Mesh *>(v.get());
        Emitter *emitter       = dynamic_cast<Emitter *>(v.get());
        Sensor *sensor         = dynamic_cast<Sensor *>(v.get());
        Integrator *integrator = dynamic_cast<Integrator *>(v.get());

        if (!scene)
            m_children.push_back(v.get());

        if (shape) {
            if (shape->is_emitter())
                m_emitters.push_back(shape->emitter());
            if (shape->is_sensor())
                m_sensors.push_back(shape->sensor());
            if (shape->is_shapegroup()) {
                m_shapegroups.push_back((ShapeGroup*)shape);
            } else {
                m_bbox.expand(shape->bbox());
                m_shapes.push_back(shape);
            }
            if (mesh)
                mesh->set_scene(this);
        } else if (emitter) {
            // Surface emitters will be added to the list when attached to a shape
            if (!has_flag(emitter->flags(), EmitterFlags::Surface))
                m_emitters.push_back(emitter);

            if (emitter->is_environment()) {
                if (m_environment)
                    Throw("Only one environment emitter can be specified per scene.");
                m_environment = emitter;
            }
        } else if (sensor) {
            m_sensors.push_back(sensor);
        } else if (integrator) {
            if (m_integrator)
                Throw("Only one integrator can be specified per scene.");
            m_integrator = integrator;
        }
    }

    // Create sensors' shapes (environment sensors)
    for (Sensor *sensor: m_sensors)
        sensor->set_scene(this);

    if constexpr (dr::is_cuda_v<Float>)
        accel_init_gpu(props);
    else
        accel_init_cpu(props);

    if (!m_emitters.empty()) {
        // Inform environment emitters etc. about the scene bounds
        for (Emitter *emitter: m_emitters)
            emitter->set_scene(this);
    }

    m_shapes_dr = dr::load<DynamicBuffer<ShapePtr>>(
        m_shapes.data(), m_shapes.size());

    m_emitters_dr = dr::load<DynamicBuffer<EmitterPtr>>(
        m_emitters.data(), m_emitters.size());

    update_emitter_sampling_distribution();

    m_shapes_grad_enabled = false;
}

MI_VARIANT
void Scene<Float, Spectrum>::update_emitter_sampling_distribution() {
    // Check if we need to use non-uniform emitter sampling.
    bool non_uniform_sampling = false;
    for (auto &e : m_emitters) {
        if (e->sampling_weight() != ScalarFloat(1.0)) {
            non_uniform_sampling = true;
            break;
        }
    }
    size_t n_emitters = m_emitters.size();
    if (non_uniform_sampling) {
        std::unique_ptr<ScalarFloat[]> sample_weights(new ScalarFloat[n_emitters]);
        for (size_t i = 0; i < n_emitters; ++i)
            sample_weights[i] = m_emitters[i]->sampling_weight();
        m_emitter_distr = std::make_unique<DiscreteDistribution<Float>>(
            sample_weights.get(), n_emitters);
    } else {
        // By default use uniform sampling with constant PMF
        m_emitter_pmf = m_emitters.empty() ? 0.f : (1.f / n_emitters);
    }
    // Clear emitter's dirty flag
    for (auto &e : m_emitters)
        e->set_dirty(false);
}

MI_VARIANT Scene<Float, Spectrum>::~Scene() {
    if constexpr (dr::is_cuda_v<Float>)
        accel_release_gpu();
    else
        accel_release_cpu();

    // Trigger deallocation of all instances
    m_emitters.clear();
    m_shapes.clear();
    m_shapegroups.clear();
    m_sensors.clear();
    m_children.clear();
    m_integrator = nullptr;
    m_environment = nullptr;

    if constexpr (dr::is_jit_v<Float>) {
        // Clean up JIT pointer registry now that the above has happened
        jit_registry_trim();
    }
}

// -----------------------------------------------------------------------

MI_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect(const Ray3f &ray, uint32_t ray_flags, Mask coherent, Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::RayIntersect, active);
    DRJIT_MARK_USED(coherent);

    if constexpr (dr::is_cuda_v<Float>)
        return ray_intersect_gpu(ray, ray_flags, active);
    else
        return ray_intersect_cpu(ray, ray_flags, coherent, active);
}

MI_VARIANT typename Scene<Float, Spectrum>::PreliminaryIntersection3f
Scene<Float, Spectrum>::ray_intersect_preliminary(const Ray3f &ray, Mask coherent, Mask active) const {
    DRJIT_MARK_USED(coherent);
    if constexpr (dr::is_cuda_v<Float>)
        return ray_intersect_preliminary_gpu(ray, active);
    else
        return ray_intersect_preliminary_cpu(ray, coherent, active);
}

MI_VARIANT typename Scene<Float, Spectrum>::Mask
Scene<Float, Spectrum>::ray_test(const Ray3f &ray, Mask coherent, Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::RayTest, active);
    DRJIT_MARK_USED(coherent);

    if constexpr (dr::is_cuda_v<Float>)
        return ray_test_gpu(ray, active);
    else
        return ray_test_cpu(ray, coherent, active);
}

MI_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect_naive(const Ray3f &ray, Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::RayIntersect, active);

#if !defined(MI_ENABLE_EMBREE)
    if constexpr (!dr::is_cuda_v<Float>)
        return ray_intersect_naive_cpu(ray, active);
#endif
    DRJIT_MARK_USED(ray);
    DRJIT_MARK_USED(active);
    NotImplementedError("ray_intersect_naive");
}

// -----------------------------------------------------------------------

MI_VARIANT std::tuple<typename Scene<Float, Spectrum>::UInt32, Float, Float>
Scene<Float, Spectrum>::sample_emitter(Float index_sample, Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::SampleEmitter, active);

    if (unlikely(m_emitters.size() < 2)) {
        if (m_emitters.size() == 1)
            return { UInt32(0), 1.f, index_sample };
        else
            return { UInt32(-1), 0.f, index_sample };
    }

    if (m_emitter_distr != nullptr) {
        auto [index, reused_sample, pmf] = m_emitter_distr->sample_reuse_pmf(index_sample);
        return {index, dr::rcp(pmf), reused_sample};
    }

    uint32_t emitter_count = (uint32_t) m_emitters.size();
    ScalarFloat emitter_count_f = (ScalarFloat) emitter_count;
    Float index_sample_scaled = index_sample * emitter_count_f;

    UInt32 index = dr::minimum(UInt32(index_sample_scaled), emitter_count - 1u);

    return { index, emitter_count_f, index_sample_scaled - Float(index) };
}

MI_VARIANT Float Scene<Float, Spectrum>::pdf_emitter(UInt32 index,
                                                      Mask active) const {
    if (m_emitter_distr == nullptr)
        return m_emitter_pmf;
    else
        return m_emitter_distr->eval_pmf_normalized(index, active);
}

MI_VARIANT std::tuple<typename Scene<Float, Spectrum>::Ray3f, Spectrum,
                       const typename Scene<Float, Spectrum>::EmitterPtr>
Scene<Float, Spectrum>::sample_emitter_ray(Float time, Float sample1,
                                           const Point2f &sample2,
                                           const Point2f &sample3,
                                           Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::SampleEmitterRay, active);


    Ray3f ray;
    Spectrum weight;
    EmitterPtr emitter;

    // Potentially disable inlining of emitter sampling (if there is just a single emitter)
    bool vcall_inline = true;
    if constexpr (dr::is_jit_v<Float>)
         vcall_inline = jit_flag(JitFlag::VCallInline);

    size_t emitter_count = m_emitters.size();
    if (emitter_count > 1 || (emitter_count == 1 && !vcall_inline)) {
        auto [index, emitter_weight, sample_1_re] = sample_emitter(sample1, active);
        emitter = dr::gather<EmitterPtr>(m_emitters_dr, index, active);

        std::tie(ray, weight) =
            emitter->sample_ray(time, sample_1_re, sample2, sample3, active);

        weight *= emitter_weight;
    } else if (emitter_count == 1) {
        std::tie(ray, weight) =
            m_emitters[0]->sample_ray(time, sample1, sample2, sample3, active);
    } else {
        ray = dr::zeros<Ray3f>();
        weight = dr::zeros<Spectrum>();
        emitter = EmitterPtr(nullptr);
    }

    return { ray, weight, emitter };
}

MI_VARIANT std::pair<typename Scene<Float, Spectrum>::DirectionSample3f, Spectrum>
Scene<Float, Spectrum>::sample_emitter_direction(const Interaction3f &ref, const Point2f &sample_,
                                                 bool test_visibility, Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::SampleEmitterDirection, active);

    Point2f sample(sample_);
    DirectionSample3f ds;
    Spectrum spec;

    // Potentially disable inlining of emitter sampling (if there is just a single emitter)
    bool vcall_inline = true;
    if constexpr (dr::is_jit_v<Float>)
         vcall_inline = jit_flag(JitFlag::VCallInline);
    

    size_t emitter_count = m_emitters.size();
    if (emitter_count > 1 || (emitter_count == 1 && !vcall_inline)) {
        // Randomly pick an emitter
        auto [index, emitter_weight, sample_x_re] = sample_emitter(sample.x(), active);
        sample.x() = sample_x_re;

        // Sample a direction towards the emitter
        EmitterPtr emitter = dr::gather<EmitterPtr>(m_emitters_dr, index, active);
        std::tie(ds, spec) = emitter->sample_direction(ref, sample, active);

        // Account for the discrete probability of sampling this emitter
        ds.pdf *= pdf_emitter(index, active);
        spec *= emitter_weight;

        active &= dr::neq(ds.pdf, 0.f);

        // Mark occluded samples as invalid if requested by the user
        if (test_visibility && dr::any_or<true>(active)) {
            Mask occluded = ray_test(ref.spawn_ray_to(ds.p), active);
            dr::masked(spec, occluded) = 0.f;
            dr::masked(ds.pdf, occluded) = 0.f;
        }
    } else if (emitter_count == 1) {
        // Sample a direction towards the (single) emitter
        std::tie(ds, spec) = m_emitters[0]->sample_direction(ref, sample, active);

        active &= dr::neq(ds.pdf, 0.f);

        // Mark occluded samples as invalid if requested by the user
        if (test_visibility && dr::any_or<true>(active)) {
            Mask occluded = ray_test(ref.spawn_ray_to(ds.p), active);
            dr::masked(spec, occluded) = 0.f;
            dr::masked(ds.pdf, occluded) = 0.f;
        }
    } else {
        ds = dr::zeros<DirectionSample3f>();
        spec = 0.f;
    }

    return { ds, spec };
}

MI_VARIANT Float
Scene<Float, Spectrum>::pdf_emitter_direction(const Interaction3f &ref,
                                              const DirectionSample3f &ds,
                                              Mask active) const {
    MI_MASK_ARGUMENT(active);
    Float emitter_pmf;
    if (m_emitter_distr == nullptr)
        emitter_pmf = m_emitter_pmf;
    else
        emitter_pmf = ds.emitter->sampling_weight() * m_emitter_distr->normalization();
    return ds.emitter->pdf_direction(ref, ds, active) * emitter_pmf;
}

MI_VARIANT Spectrum Scene<Float, Spectrum>::eval_emitter_direction(
    const Interaction3f &ref, const DirectionSample3f &ds, Mask active) const {
    MI_MASK_ARGUMENT(active);
    return ds.emitter->eval_direction(ref, ds, active);
}

MI_VARIANT void Scene<Float, Spectrum>::build_geometric_edges() const {
    size_t num_tot_edges = 0;
    auto& em = m_edge_manager;
    for (auto&s : m_shapes) {
        if (s->is_mesh()) {
            const Mesh *m = (const Mesh *) s.get();
            num_tot_edges += m->edge_count();
        }
    }
    em.resize(num_tot_edges);
    
    uint32_t offset = 0;
    for (auto &s : m_shapes) {
        if (s->is_mesh()) {
            const Mesh *m = (const Mesh *) s.get();
            // auto v = m->vertex_positions_buffer();
            uint32_t num_edges = m->edge_count();
            UInt32 idx_in = dr::arange<UInt32>(0, num_edges);
            UInt32 idx_out = dr::arange<UInt32>(offset, offset + num_edges);
            // write to edge vertices
            const Vector3u idx_v = m->edge_indices_v(idx_in);       // CZ: is there a way to convert DynamicBuffer<UInt32> to Vector3u rather than using gather?
            Point3f p0 = m->vertex_position(idx_v[0]);
            Point3f p1 = m->vertex_position(idx_v[1]);
            Point3f p2 = m->vertex_position(idx_v[2]);
            if constexpr (dr::is_jit_v<Float>) {
                dr::scatter(em.p0, p0, idx_out);
                dr::scatter(em.p1, p1, idx_out);
                dr::scatter(em.p2, p2, idx_out);
            }
            // write to face normal
            const Vector2u idx_f = m->edge_indices_f(idx_in);
            if constexpr (dr::is_jit_v<Float>) {
                auto get_face_normal = [idx_f, &m](const Vector3u& fi) {
                    Point3f v0 = m->vertex_position(fi[0]);
                    Point3f v1 = m->vertex_position(fi[1]);
                    Point3f v2 = m->vertex_position(fi[2]);
                    return dr::normalize(dr::cross(v1-v0, v2-v0));
                };
                // first neighboring face
                Vector3u fi_0 = m->face_indices(idx_f[0], true);
                Normal3f n0 = get_face_normal(fi_0);
                dr::scatter(em.n0, n0, idx_out);
                // second neighboring face (if exist)
                Mask boundary = dr::eq(idx_f[1], 0);
                dr::scatter(em.boundary, boundary, idx_out);
                Vector3u fi_1 = m->face_indices(idx_f[1] - dr::select(boundary, 0, 1));
                Normal3f n1 = dr::select(boundary, 0.0, get_face_normal(fi_1));
                dr::scatter(em.n1, n1, idx_out);
            }
            offset += num_edges;
        }
    }
    // remove concave edges (including coplanar)
    Vector3f e = dr::normalize(em.p2 - em.p0);
    if constexpr (dr::is_jit_v<Float>) {
        Mask valid = em.boundary | (dr::dot(e, em.n1) < -math::EdgeEpsilon<Float>);
        auto valid_index = dr::compress(valid);
        em.p0 = dr::gather<Point3f>(em.p0, valid_index);
        em.p1 = dr::gather<Point3f>(em.p1, valid_index);
        em.p2 = dr::gather<Point3f>(em.p2, valid_index);
        em.n0 = dr::gather<Normal3f>(em.n0, valid_index);
        em.n1 = dr::gather<Normal3f>(em.n1, valid_index);
        em.boundary = dr::gather<Mask>(em.boundary, valid_index);
        em.count = valid_index.size();
    }
    
    auto edge_length = dr::detach(dr::norm(em.p1 - em.p0));       // sampling from edge length
    // initialize distribution for direct/indirect boundary terms 
    em.distr = std::make_unique<DiscreteDistribution<Float>>(edge_length);
    // initialzie distribution for primary boundary term (per sensor)
    em.pr_distr.resize(m_sensors.size());
    em.pr_idx.resize(m_sensors.size());
    for (size_t i = 0; i < m_sensors.size(); i++) {
        const Sensor* sensor = m_sensors[i].get();
        auto cam_pos = sensor->world_transform().translation();
        // remove invalid edges
        Vector3f d = cam_pos - em.p0;
        auto front_facing_0 = dr::dot(d, em.n0) > 0.f;
        auto front_facing_1 = dr::dot(d, em.n1) > 0.f;
        if constexpr (dr::is_jit_v<Float>) {
            Mask valid = dr::neq(front_facing_0, front_facing_1) | em.boundary;
            em.pr_idx[i] = dr::compress(valid);
            em.pr_distr[i] = std::make_unique<DiscreteDistribution<Float>>(dr::gather<Float>(edge_length, em.pr_idx[i]));
        }
    }
}

MI_VARIANT EdgeSample<Float> Scene<Float, Spectrum>::sample_edge_ray(Float sample1,
                                                                     const Point2f &sample2,
                                                                     uint32_t boundary_flags,
                                                                     uint32_t cam_id) const
{
    EdgeSample<Float> res;
    auto& em = m_edge_manager;
    if (has_flag(boundary_flags, BoundaryFlags::Pixel)) {
        // pixel boundary: sample on pixel boundary

    } else {
        // primary/direct/indirect boundary: sample on geometric edge
        bool is_pr = has_flag(boundary_flags, BoundaryFlags::Primary);
        auto [index, reused_sample, pmf] = is_pr ? em.pr_distr[cam_id]->sample_reuse_pmf(sample1)
                                                 : em.distr->sample_reuse_pmf(sample1);
        if (is_pr)
            index = dr::gather<UInt32>(em.pr_idx[cam_id], index);
        auto p0 = dr::gather<Point3f>(em.p0, index);
        auto p1 = dr::gather<Point3f>(em.p1, index);
        res.p = p0 + reused_sample * p1;
        res.pdf = pmf / dr::norm(p0 - p1);       // sample uniformly on the edge
        if (is_pr) {
            // primary: direct connect to the camera
            const Sensor* sensor = m_sensors[cam_id].get();
            auto cam_pos = sensor->world_transform().translation();
            res.d = dr::detach(dr::normalize(res.p - cam_pos));
        } else if (has_flag(boundary_flags, BoundaryFlags::Direct)) {
            // direct: sample point on emitter
            Point2f sample(sample2);
            size_t emitter_count = m_emitters.size();
            bool vcall_inline = true;
            if constexpr (dr::is_jit_v<Float>)
                vcall_inline = jit_flag(JitFlag::VCallInline);
            if (emitter_count > 1 || (emitter_count == 1 && !vcall_inline)) {
                // Randomly pick an emitter
                auto [emitter_index, emitter_weight, sample_x_re] = sample_emitter(sample.x());
                sample.x() = sample_x_re;
                EmitterPtr emitter = dr::gather<EmitterPtr>(m_emitters_dr, emitter_index);
                res.pdf *= pdf_emitter(emitter_index);
                // Sample a position on the emitter
                auto [ps, w] = emitter->sample_position(0.0, sample);
                res.d = dr::detach(dr::normalize(ps.p - res.p));
                res.pdf *= emitter->pdf_position(ps);
            } else if (emitter_count == 1) {
                // Sample a position on the emitter
                auto [ps, w] = m_emitters[0]->sample_position(0.0, sample);
                res.d = dr::detach(dr::normalize(ps.p - res.p));
                res.pdf *= m_emitters[0]->pdf_position(ps);
            } else {
                Throw("No emitter can be sampled in the scene");  
            }
        } else {
            // indirect: sample a direction
        }
    }

    return res;
}

MI_VARIANT void Scene<Float, Spectrum>::traverse(TraversalCallback *callback) {
    for (auto& child : m_children) {
        std::string id = child->id();
        if (id.empty() || string::starts_with(id, "_unnamed_"))
            id = child->class_()->name();
        callback->put_object(id, child.get(), +ParamFlags::Differentiable);
    }
}

MI_VARIANT void Scene<Float, Spectrum>::parameters_changed(const std::vector<std::string> &/*keys*/) {
    if (m_environment)
        m_environment->set_scene(this); // TODO use parameters_changed({"scene"})

    bool accel_is_dirty = false;
    for (auto &s : m_shapes) {
        if (s->dirty()) {
            accel_is_dirty = true;
            break;
        }
    }

    for (auto &s : m_shapegroups) {
        if (s->dirty()) {
            accel_is_dirty = true;
            break;
        }
    }

    if (accel_is_dirty) {
        if constexpr (dr::is_cuda_v<Float>)
            accel_parameters_changed_gpu();
        else
            accel_parameters_changed_cpu();
    }

    // Check whether any shape parameters have gradient tracking enabled
    m_shapes_grad_enabled = false;
    for (auto &s : m_shapes) {
        m_shapes_grad_enabled |= s->parameters_grad_enabled();
        if (m_shapes_grad_enabled)
            break;
    }

    // Check if emitters were modified and we potentially need to update
    // the emitter sampling distribution.
    for (auto &e : m_emitters) {
        if (e->dirty()) {
            update_emitter_sampling_distribution();
            break;
        }
    }
}

MI_VARIANT std::string Scene<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
    oss << "Scene[" << std::endl
        << "  children = [" << std::endl;
    for (size_t i = 0; i < m_children.size(); ++i) {
        oss << "    " << string::indent(m_children[i], 4);
        if (i + 1 < m_children.size())
            oss << ",";
        oss <<  std::endl;
    }
    oss << "  ]"<< std::endl
        << "]";
    return oss.str();
}

MI_VARIANT void Scene<Float, Spectrum>::static_accel_initialization() {
    if constexpr (dr::is_cuda_v<Float>)
        Scene::static_accel_initialization_gpu();
    else
        Scene::static_accel_initialization_cpu();
}

MI_VARIANT void Scene<Float, Spectrum>::static_accel_shutdown() {
    if constexpr (dr::is_cuda_v<Float>)
        Scene::static_accel_shutdown_gpu();
    else
        Scene::static_accel_shutdown_cpu();
}

MI_VARIANT void Scene<Float, Spectrum>::clear_shapes_dirty() {
    for (auto &s : m_shapes)
        s->m_dirty = false;
    for (auto &s : m_shapegroups)
        s->m_dirty = false;
}

MI_VARIANT void Scene<Float, Spectrum>::static_accel_initialization_cpu() { }
MI_VARIANT void Scene<Float, Spectrum>::static_accel_shutdown_cpu() { }

void librender_nop() { }

#if !defined(MI_ENABLE_CUDA)
MI_VARIANT void Scene<Float, Spectrum>::accel_init_gpu(const Properties &) {
    NotImplementedError("accel_init_gpu");
}
MI_VARIANT void Scene<Float, Spectrum>::accel_parameters_changed_gpu() {
    NotImplementedError("accel_parameters_changed_gpu");
}
MI_VARIANT void Scene<Float, Spectrum>::accel_release_gpu() {
    NotImplementedError("accel_release_gpu");
}
MI_VARIANT typename Scene<Float, Spectrum>::PreliminaryIntersection3f
Scene<Float, Spectrum>::ray_intersect_preliminary_gpu(const Ray3f &, Mask) const {
    NotImplementedError("ray_intersect_preliminary_gpu");
}
MI_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect_gpu(const Ray3f &, uint32_t, Mask) const {
    NotImplementedError("ray_intersect_naive_gpu");
}
MI_VARIANT typename Scene<Float, Spectrum>::Mask
Scene<Float, Spectrum>::ray_test_gpu(const Ray3f &, Mask) const {
    NotImplementedError("ray_test_gpu");
}
MI_VARIANT void Scene<Float, Spectrum>::static_accel_initialization_gpu() { }
MI_VARIANT void Scene<Float, Spectrum>::static_accel_shutdown_gpu() { }
#endif

MI_IMPLEMENT_CLASS_VARIANT(Scene, Object, "scene")
MI_INSTANTIATE_CLASS(Scene)
NAMESPACE_END(mitsuba)
