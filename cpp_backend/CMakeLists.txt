cmake_minimum_required(VERSION 3.15)
project(PrincipiaCore LANGUAGES CXX)

# Find Pybind11.
find_package(pybind11 REQUIRED)

# Find Faiss. We need this for the include directories.
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
# The linker finds them because of the conda environment.
target_link_libraries(_core PRIVATE faiss omp)

# --- THE FINAL, CORRECT RPATH SOLUTION FOR MACOS ---
# This tells our compiled library to look for dependencies (like libfaiss.dylib)
# in the directory located at "../lib" relative to its own location.
set_target_properties(_core PROPERTIES
    INSTALL_RPATH "@loader_path/../lib"
)

# Install our compiled module into the final Python package.
install(TARGETS _core
        DESTINATION principia_semantica)