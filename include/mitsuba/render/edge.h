#pragma once
#include <mitsuba/core/struct.h>
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/render/fwd.h>
#include <drjit/struct.h>
#include <mitsuba/core/vector.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float_>
struct EdgeSample
{
    using Float    = Float_;
    MI_IMPORT_CORE_TYPES();

    Point3f p;
    Vector3f e;
    Vector3f e2;
    UInt32 idx;
    Float pdf;

    DRJIT_STRUCT(EdgeSample, p, e, e2, idx, pdf)
};

enum class BoundaryFlags : uint32_t {
    /// Primary boundary
    Pixel = 0x1,
    /// Pixel boundary
    Primary = 0x2,
    /// Direct boundary
    Direct = 0x4,
    /// Indirect boundary
    Indirect = 0x8,
};
MI_DECLARE_ENUM_OPERATORS(BoundaryFlags)

template <typename Float_>
struct EdgeManager
{
    using Float    = Float_;
    MI_IMPORT_CORE_TYPES();

    void resize(size_t num_edges) {
        count = num_edges;
        p0 = dr::empty<Point3f>(count);
        p1 = dr::empty<Point3f>(count);
        n0 = dr::empty<Normal3f>(count);
        n1 = dr::empty<Normal3f>(count);
        p2 = dr::empty<Point3f>(count);
        boundary = dr::empty<Mask>(count);
        distr = nullptr;
        pr_distr.clear();
        pr_idx.clear();
    }

    Mask boundary;
    Point3f p0, p1;
    Normal3f n0, n1;
    Point3f p2;
    size_t count;
    // Distribution for direct/indirect boundary
    std::unique_ptr<DiscreteDistribution<Float>> distr = nullptr;
    // Distribution for primary boundary
    std::vector<std::unique_ptr<DiscreteDistribution<Float>>> pr_distr;
    std::vector<UInt32> pr_idx;
};


NAMESPACE_END(mitsuba)

