cmake_minimum_required(VERSION 3.15)
project(PrincipiaCore LANGUAGES CXX)

# Find Pybind11.
find_package(pybind11 REQUIRED)

# Find Faiss.
find_package(Faiss REQUIRED)

# Create our Python module.
pybind11_add_module(_core SHARED
    main.cpp
    bindings.cpp
)

# Add the Faiss include directory for the compiler.
target_include_directories(_core PRIVATE ${FAISS_INCLUDE_DIRS})

# Add the OpenMP compiler flag.
target_compile_options(_core PRIVATE -fopenmp)

# Link against the Faiss and OpenMP libraries by name.
target_link_libraries(_core PRIVATE faiss omp)

# Use the correct relative RPATH for macOS.
set_target_properties(_core PROPERTIES
    INSTALL_RPATH "@loader_path/../lib"
)

# --- RENAMED ---
# Install our compiled module into the new package directory.
install(TARGETS _core
        DESTINATION aglt)
