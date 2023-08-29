#pragma once
#include <mitsuba/core/struct.h>
#include <mitsuba/render/fwd.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float_>
struct GeometricEdge
{
    using Float    = Float_;
    MI_IMPORT_CORE_TYPES();

    void initialize() {
        p0 = dr::empty<Point3f>(edge_count);
        p1 = dr::empty<Point3f>(edge_count);
        n0 = dr::empty<Normal3f>(edge_count);
        n1 = dr::empty<Normal3f>(edge_count);
        p2 = dr::empty<Point3f>(edge_count);
    }

    Point3f p0, p1;
    Normal3f n0, n1;
    Point3f p2;

    size_t edge_count;
};

NAMESPACE_END(mitsuba)

