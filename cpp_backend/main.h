#ifndef PRINCIPIA_MAIN_H
#define PRINCIPIA_MAIN_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Builds a k-NN graph using the Faiss C++ backend.
//
// @param data A 2D NumPy array of floats (n_points, n_dimensions).
// @param k The number of neighbors to find for each point.
// @return A tuple containing two NumPy arrays:
//         1. Neighbors: A 2D array of int64_t (n_points, k) with neighbor indices.
//         2. Distances: A 2D array of floats (n_points, k) with squared L2 distances.
std::pair<py::array_t<int64_t>, py::array_t<float>>
build_faiss_knn_graph(py::array_t<float, py::array::c_style | py::array::forcecast> data, int k);

#endif // PRINCIPIA_MAIN_H