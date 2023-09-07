#include <mitsuba/render/edge.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(EdgeManager) {
    MI_PY_IMPORT_TYPES()
    py::class_<EdgeManager<Float>>(m, "EdgeManager")
        // Members
        .def_field(EdgeManager<Float>, p0)
        .def_field(EdgeManager<Float>, p1)
        .def_field(EdgeManager<Float>, p2)
        .def_field(EdgeManager<Float>, n0)
        .def_field(EdgeManager<Float>, n1)
        .def_field(EdgeManager<Float>, boundary)
        .def_field(EdgeManager<Float>, count)
        .def_field(EdgeManager<Float>, pr_idx);
        // .def("sample_edge_ray", &EdgeManager<Float>::sample_edge_ray,
        //      "sample1"_a, "sample2"_a, "bflag"_a, "cam_id"_a = 0);
}

MI_PY_EXPORT(EdgeSample) {
    MI_PY_IMPORT_TYPES()
    py::class_<EdgeSample<Float>>(m, "EdgeSample")
        // Members
        .def_field(EdgeSample<Float>, p)
        .def_field(EdgeSample<Float>, d)
        .def_field(EdgeSample<Float>, pdf);
}