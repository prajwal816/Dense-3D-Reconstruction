/**
 * Point cloud filter — statistical outlier removal (C++ accelerator stub).
 *
 * This provides a skeleton for C++ acceleration of point cloud operations.
 * The actual filtering logic is handled by Open3D in the Python pipeline;
 * this file exists to demonstrate the hybrid C++/Python project structure.
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace dense3d {

struct Point3D {
    double x, y, z;
};

/**
 * Compute pairwise Euclidean distance between two points.
 */
inline double distance(const Point3D& a, const Point3D& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Statistical outlier removal.
 *
 * For each point, compute the mean distance to its k nearest neighbours.
 * Points whose mean distance exceeds (global_mean + std_multiplier * global_std)
 * are classified as outliers.
 *
 * @param points      Input point cloud.
 * @param nb_neighbors Number of neighbours to consider.
 * @param std_ratio    Standard deviation multiplier threshold.
 * @return             Indices of inlier points.
 */
std::vector<size_t> statistical_outlier_removal(
    const std::vector<Point3D>& points,
    int nb_neighbors = 20,
    double std_ratio = 2.0
) {
    const size_t n = points.size();
    if (n == 0) return {};

    // Compute mean distance to k-NN for each point (brute-force for stub)
    std::vector<double> mean_dists(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        std::vector<double> dists;
        dists.reserve(n - 1);

        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            dists.push_back(distance(points[i], points[j]));
        }

        std::partial_sort(
            dists.begin(),
            dists.begin() + std::min<size_t>(nb_neighbors, dists.size()),
            dists.end()
        );

        size_t k = std::min<size_t>(nb_neighbors, dists.size());
        double sum = std::accumulate(dists.begin(), dists.begin() + k, 0.0);
        mean_dists[i] = sum / static_cast<double>(k);
    }

    // Global statistics
    double global_mean = std::accumulate(mean_dists.begin(), mean_dists.end(), 0.0)
                         / static_cast<double>(n);

    double sq_sum = 0.0;
    for (double d : mean_dists) {
        sq_sum += (d - global_mean) * (d - global_mean);
    }
    double global_std = std::sqrt(sq_sum / static_cast<double>(n));

    double threshold = global_mean + std_ratio * global_std;

    // Filter
    std::vector<size_t> inliers;
    for (size_t i = 0; i < n; ++i) {
        if (mean_dists[i] <= threshold) {
            inliers.push_back(i);
        }
    }

    std::cout << "[dense3d] SOR: " << n << " points → " << inliers.size()
              << " inliers (threshold=" << threshold << ")" << std::endl;

    return inliers;
}

}  // namespace dense3d
