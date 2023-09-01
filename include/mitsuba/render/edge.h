#pragma once
#include <mitsuba/core/struct.h>
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/render/fwd.h>
#include <drjit/struct.h>
#include <mitsuba/core/vector.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float_>
struct EdgeManager
{
    using Float    = Float_;
    MI_IMPORT_CORE_TYPES();

    void initialize() {
        p0 = dr::empty<Point3f>(count);
        p1 = dr::empty<Point3f>(count);
        n0 = dr::empty<Normal3f>(count);
        n1 = dr::empty<Normal3f>(count);
        p2 = dr::empty<Point3f>(count);
        boundary = dr::empty<Mask>(count);
    }

    void initialize_distr() {
        // auto edge_length = dr::norm(p1 - p0);       // sampling from edge length
        // if constexpr (dr::is_jit_v<Float>) {
        //     // m_edge_distr = std::make_unique<DiscreteDistribution<Float>>(edge_length, m_edges.count);
        //     // auto&& data = dr::gather<DynamicBuffer<Float>>(edge_length, dr::arange<UInt32>(0, m_edges.count));
        //     // m_edge_distr = std::make_unique<DiscreteDistribution<Float>>(data, m_edges.count);
        //     // m_distr = std::make_unique<DiscreteDistribution<Float>>();
        // }
    }

    Mask boundary;
    Point3f p0, p1;
    Normal3f n0, n1;
    Point3f p2;
    size_t count;
    // Distribution for direct/indirect boundary
    // std::unique_ptr<DiscreteDistribution<Float>> m_distr = nullptr;
    // Distribution for primary boundary


    DRJIT_STRUCT(EdgeManager, boundary, p0, p1, n0, n1, p2)
};

template <typename Float_>
struct EdgeSample
{
    using Float    = Float_;
    MI_IMPORT_CORE_TYPES();

    Point3f p;
    Vector3f d;

    DRJIT_STRUCT(EdgeSample, p, d)
};

NAMESPACE_END(mitsuba)

