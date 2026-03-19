cmake_minimum_required(VERSION 3.16)
project(Dense3DReconstruction VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ──────────────────────────────────────────────────────────────────
option(BUILD_FILTERS   "Build C++ point-cloud filter accelerators" ON)
option(BUILD_TESTS     "Build unit tests"                          OFF)

# ── Find packages ────────────────────────────────────────────────────────────
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d)
find_package(Open3D QUIET)

# ── Point-cloud filter library (optional C++ accelerator) ────────────────────
if(BUILD_FILTERS)
    add_library(pcl_filters STATIC
        src/cpp/point_cloud_filter.cpp
        src/cpp/statistical_outlier_removal.cpp
    )
    target_include_directories(pcl_filters PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include
        ${EIGEN3_INCLUDE_DIR}
    )
    target_link_libraries(pcl_filters PUBLIC Eigen3::Eigen)

    if(Open3D_FOUND)
        target_link_libraries(pcl_filters PUBLIC Open3D::Open3D)
        target_compile_definitions(pcl_filters PUBLIC HAS_OPEN3D)
    endif()
endif()

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS pcl_filters
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# ── Tests ────────────────────────────────────────────────────────────────────
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_filters tests/test_filters.cpp)
    target_link_libraries(test_filters pcl_filters)
    add_test(NAME FilterTests COMMAND test_filters)
endif()

message(STATUS "Dense3DReconstruction ${PROJECT_VERSION}")
message(STATUS "  BUILD_FILTERS : ${BUILD_FILTERS}")
message(STATUS "  BUILD_TESTS   : ${BUILD_TESTS}")
message(STATUS "  Open3D found  : ${Open3D_FOUND}")
