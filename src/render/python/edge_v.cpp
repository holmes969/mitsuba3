#include <mitsuba/render/edge.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(GeometricEdge) {
    MI_PY_IMPORT_TYPES()
    // auto e = py::class_<GeometricEdge<Float>>(m, "Edges")
    //     // Members
    //     .def_field(GeometricEdge<Float>, p0);
    // MI_PY_DRJIT_STRUCT(e, GeometricEdge<Float>, p0)
}