#pragma once

/**
 * Dense3D C++ accelerator headers.
 */

#include <vector>
#include <cstddef>

namespace dense3d {

struct Point3D {
    double x, y, z;
};

struct VoxelPoint {
    double x, y, z;
    double r, g, b;
};

std::vector<size_t> statistical_outlier_removal(
    const std::vector<Point3D>& points,
    int nb_neighbors = 20,
    double std_ratio = 2.0
);

std::vector<VoxelPoint> voxel_downsample(
    const std::vector<VoxelPoint>& points,
    double voxel_size = 0.01
);

}  // namespace dense3d
