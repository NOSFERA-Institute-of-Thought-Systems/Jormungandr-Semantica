#include "main.h"
#include <faiss/IndexFlat.h>

std::pair<py::array_t<int64_t>, py::array_t<float>>
build_faiss_knn_graph(py::array_t<float, py::array::c_style | py::array::forcecast> data, int k) {
    // 1. Get information from the input NumPy array
    py::buffer_info buf = data.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional.");
    }
    size_t n_points = buf.shape[0];
    size_t n_dims = buf.shape[1];
    auto *data_ptr = static_cast<float *>(buf.ptr);

    // 2. Create and train the Faiss index
    faiss::IndexFlatL2 index(n_dims);
    index.add(n_points, data_ptr);

    // 3. Prepare the output NumPy arrays
    // Faiss returns neighbor indices as int64_t and distances as float32
    auto neighbors = py::array_t<int64_t>(k * n_points);
    auto distances = py::array_t<float>(k * n_points);
    py::buffer_info neighbors_buf = neighbors.request();
    py::buffer_info distances_buf = distances.request();
    auto *neighbors_ptr = static_cast<int64_t *>(neighbors_buf.ptr);
    auto *distances_ptr = static_cast<float *>(distances_buf.ptr);

    // 4. Perform the k-NN search
    index.search(n_points, data_ptr, k, distances_ptr, neighbors_ptr);

    // 5. Reshape the output arrays to be 2D (n_points, k)
    neighbors.resize({n_points, (size_t)k});
    distances.resize({n_points, (size_t)k});

    // 6. Return the pair of arrays
    return std::make_pair(neighbors, distances);
}