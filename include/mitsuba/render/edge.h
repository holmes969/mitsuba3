#pragma once
#include <mitsuba/core/struct.h>
#include <mitsuba/render/fwd.h>
#include <drjit/struct.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float_>
struct GeometricEdge
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

    Mask boundary;
    Point3f p0, p1;
    Normal3f n0, n1;
    Point3f p2;

    size_t count;

    DRJIT_STRUCT(GeometricEdge, boundary, p0, p1, n0, n1, p2)
};

NAMESPACE_END(mitsuba)

