/**
 * Point cloud voxel filter (C++ accelerator stub).
 */

#include <vector>
#include <unordered_map>
#include <cmath>
#include <iostream>

namespace dense3d {

struct Point3D;  // Forward declaration from statistical_outlier_removal.cpp

struct VoxelPoint {
    double x, y, z;
    double r, g, b;
};

/**
 * Simple hash for voxel grid keys.
 */
struct VoxelKeyHash {
    std::size_t operator()(const std::tuple<int, int, int>& key) const {
        auto h1 = std::hash<int>{}(std::get<0>(key));
        auto h2 = std::hash<int>{}(std::get<1>(key));
        auto h3 = std::hash<int>{}(std::get<2>(key));
        return h1 ^ (h2 << 16) ^ (h3 << 32);
    }
};

/**
 * Voxel downsampling of a point cloud.
 *
 * @param points     Input points with colour.
 * @param voxel_size Side length of each voxel cube.
 * @return           Downsampled points (centroid per voxel).
 */
std::vector<VoxelPoint> voxel_downsample(
    const std::vector<VoxelPoint>& points,
    double voxel_size = 0.01
) {
    using Key = std::tuple<int, int, int>;
    std::unordered_map<Key, std::vector<size_t>, VoxelKeyHash> grid;

    double inv = 1.0 / voxel_size;
    for (size_t i = 0; i < points.size(); ++i) {
        int vx = static_cast<int>(std::floor(points[i].x * inv));
        int vy = static_cast<int>(std::floor(points[i].y * inv));
        int vz = static_cast<int>(std::floor(points[i].z * inv));
        grid[{vx, vy, vz}].push_back(i);
    }

    std::vector<VoxelPoint> result;
    result.reserve(grid.size());

    for (auto& [key, indices] : grid) {
        double sx = 0, sy = 0, sz = 0, sr = 0, sg = 0, sb = 0;
        for (size_t idx : indices) {
            sx += points[idx].x;
            sy += points[idx].y;
            sz += points[idx].z;
            sr += points[idx].r;
            sg += points[idx].g;
            sb += points[idx].b;
        }
        double n = static_cast<double>(indices.size());
        result.push_back({sx / n, sy / n, sz / n, sr / n, sg / n, sb / n});
    }

    std::cout << "[dense3d] Voxel downsample: " << points.size()
              << " → " << result.size() << " (voxel=" << voxel_size << ")"
              << std::endl;

    return result;
}

}  // namespace dense3d
