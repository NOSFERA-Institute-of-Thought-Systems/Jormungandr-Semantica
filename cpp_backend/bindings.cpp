#include "main.h"

PYBIND11_MODULE(_core, m) {
    m.doc() = "Principia Semantica's high-performance C++ backend";

    m.def(
        "build_faiss_knn_graph",
        &build_faiss_knn_graph,
        "Builds a k-NN graph using the Faiss C++ backend.",
        py::arg("data"),
        py::arg("k")
    );
}