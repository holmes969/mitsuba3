#include <mitsuba/render/edge.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(EdgeManager) {
    MI_PY_IMPORT_TYPES()
    auto e = py::class_<EdgeManager<Float>>(m, "Edges")
        // Members
        .def_field(EdgeManager<Float>, p0)
        .def_field(EdgeManager<Float>, p1)
        .def_field(EdgeManager<Float>, p2)
        .def_field(EdgeManager<Float>, n0)
        .def_field(EdgeManager<Float>, n1)
        .def_field(EdgeManager<Float>, boundary)
        .def_field(EdgeManager<Float>, count);
    MI_PY_DRJIT_STRUCT(e, EdgeManager<Float>, p0, p1, p2, n0, n1, boundary, count)
}