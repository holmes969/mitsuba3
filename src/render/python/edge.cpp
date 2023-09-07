#include <mitsuba/render/edge.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(BoundaryFlags) {
    auto e = py::enum_<BoundaryFlags>(m, "BoundaryFlags", py::arithmetic())
        .def_value(BoundaryFlags, Pixel)
        .def_value(BoundaryFlags, Primary)
        .def_value(BoundaryFlags, Direct)
        .def_value(BoundaryFlags, Indirect);
    MI_PY_DECLARE_ENUM_OPERATORS(BoundaryFlags, e)
}
