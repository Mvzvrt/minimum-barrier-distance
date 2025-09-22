/**
 * @file mbd_core.cpp
 * @brief C++ implementation of Minimum Barrier Distance (MBD) algorithm with pybind11 bindings
 * @version 0.1.0
 * @date 2025
 *
 * High-performance implementation of the Minimum Barrier Distance algorithm for seeded
 * image segmentation. This implementation follows the exact propagation approach described
 * in "The Minimum Barrier Distance – Stability to Seed Point Position" (Strand et al., 2014).
 *
 * Key Features:
 * - Exact MBD computation using priority queue propagation
 * - Support for both 4 and 8 connectivity
 * - Efficient numpy array integration via pybind11
 * - Multi-label segmentation with hard constraints
 *
 * Implementation Notes:
 * - Priority queue ordering: bmax (ascending), bmin (descending), label (ascending)
 * - Seeds with labels > 0 are treated as hard constraints
 * - Memory efficient using contiguous array storage
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <queue>
#include <vector>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

/**
 * @brief Node structure for the priority queue in MBD propagation
 *
 * Implements the lexicographic ordering required for exact MBD computation:
 * 1. bmax: ascending order (minimize maximum value along path)
 * 2. bmin: descending order (maximize minimum value along path)
 * 3. label: ascending order (deterministic tie-breaking by seed label)
 */
struct Node
{
    float bmax; ///< Maximum value along the path
    float bmin; ///< Minimum value along the path
    int index;  ///< Linear index in the image array
    int label;  ///< Propagating label from seed

    bool operator<(const Node &other) const
    {
        if (bmax != other.bmax)
            return bmax > other.bmax; // smaller bmax first
        if (bmin != other.bmin)
            return bmin < other.bmin; // larger bmin first
        return label > other.label;   // smaller label first
    }
};

/**
 * @brief Get 4-connected neighbors for a pixel index
 * @param idx Linear index in the image array
 * @param H Image height
 * @param W Image width
 * @param out Vector to store neighbor indices (cleared before use)
 */
static inline void neighbors4(int idx, int H, int W, std::vector<int> &out)
{
    out.clear();
    int y = idx / W, x = idx % W;
    if (y > 0)
        out.push_back((y - 1) * W + x); // North
    if (y < H - 1)
        out.push_back((y + 1) * W + x); // South
    if (x > 0)
        out.push_back(y * W + (x - 1)); // West
    if (x < W - 1)
        out.push_back(y * W + (x + 1)); // East
}

/**
 * @brief Get 8-connected neighbors for a pixel index
 * @param idx Linear index in the image array
 * @param H Image height
 * @param W Image width
 * @param out Vector to store neighbor indices (cleared before use)
 */
static inline void neighbors8(int idx, int H, int W, std::vector<int> &out)
{
    out.clear();
    int y = idx / W, x = idx % W;
    if (y > 0)
        out.push_back((y - 1) * W + x);
    if (y < H - 1)
        out.push_back((y + 1) * W + x);
    if (x > 0)
        out.push_back(y * W + (x - 1));
    if (x < W - 1)
        out.push_back(y * W + (x + 1));
    if (y > 0 && x > 0)
        out.push_back((y - 1) * W + (x - 1));
    if (y > 0 && x < W - 1)
        out.push_back((y - 1) * W + (x + 1));
    if (y < H - 1 && x > 0)
        out.push_back((y + 1) * W + (x - 1));
    if (y < H - 1 && x < W - 1)
        out.push_back((y + 1) * W + (x + 1));
}

/**
 * @brief Run MBD label propagation with exact distance computation
 *
 * @param weights Input image as float32 array [H,W] with values in [0,1]
 * @param seeds Seed mask as int32 array [H,W], 0=unlabeled, >0=seed labels
 * @param conn Connectivity, either 4 or 8 (default=4)
 * @return tuple(labels, distances, pop_count)
 *         - labels: int32 array [H,W] with propagated labels
 *         - distances: float32 array [H,W] with barrier distances
 *         - pop_count: int, number of priority queue operations
 *
 * @throws std::runtime_error if inputs are invalid
 */
py::tuple run_mbd_label_propagation(py::array_t<float, py::array::c_style | py::array::forcecast> weights,
                                    py::array_t<int, py::array::c_style | py::array::forcecast> seeds,
                                    int conn)
{
    // Validate inputs
    if (conn != 4 && conn != 8)
    {
        throw std::runtime_error("conn must be 4 or 8");
    }

    auto bw = weights.request();
    auto bs = seeds.request();

    if (bw.ndim != 2 || bs.ndim != 2)
    {
        throw std::runtime_error("weights and seeds must be 2D arrays");
    }
    if (bw.shape[0] != bs.shape[0] || bw.shape[1] != bs.shape[1])
    {
        throw std::runtime_error("weights and seeds arrays must have the same shape");
    }

    const int H = static_cast<int>(bw.shape[0]);
    const int W = static_cast<int>(bw.shape[1]);
    const int N = H * W;

    const float *w = static_cast<float *>(bw.ptr);
    const int *s = static_cast<int *>(bs.ptr);

    std::vector<float> bmin(N, std::numeric_limits<float>::infinity());
    std::vector<float> bmax(N, -std::numeric_limits<float>::infinity());
    std::vector<float> dist(N, std::numeric_limits<float>::infinity());
    std::vector<int> label(N, 0);
    std::vector<char> locked(N, 0);

    std::priority_queue<Node> pq;

    // Initialize seeds, labels greater than 0 are hard constraints
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            int idx = y * W + x;
            int lab = s[idx];
            if (lab > 0)
            {
                float val = w[idx];
                bmin[idx] = val;
                bmax[idx] = val;
                dist[idx] = 0.0f;
                label[idx] = lab;
                locked[idx] = 1;
                pq.push(Node{bmax[idx], bmin[idx], idx, lab});
            }
        }
    }

    std::vector<int> neigh;
    int pops = 0;
    while (!pq.empty())
    {
        Node node = pq.top();
        pq.pop();
        ++pops;

        int idx = node.index;
        if (node.bmax != bmax[idx] || node.bmin != bmin[idx] || node.label != label[idx])
            continue;

        if (conn == 8)
            neighbors8(idx, H, W, neigh);
        else
            neighbors4(idx, H, W, neigh);

        for (int j : neigh)
        {
            if (locked[j] && label[j] != node.label)
                continue;

            float cand_bmin = std::min(bmin[idx], w[j]);
            float cand_bmax = std::max(bmax[idx], w[j]);
            float cand_bw = cand_bmax - cand_bmin;

            bool upd = false;
            if (cand_bw < dist[j])
            {
                upd = true;
            }
            else if (cand_bw == dist[j])
            {
                if (cand_bmax < bmax[j])
                    upd = true;
                else if (cand_bmax == bmax[j] && cand_bmin > bmin[j])
                    upd = true;
                else if (cand_bmax == bmax[j] && cand_bmin == bmin[j] && node.label < label[j])
                    upd = true;
            }

            if (upd)
            {
                bmin[j] = cand_bmin;
                bmax[j] = cand_bmax;
                dist[j] = cand_bw;
                label[j] = node.label;
                pq.push(Node{bmax[j], bmin[j], j, label[j]});
            }
        }
    }

    // Prepare outputs
    py::array_t<int> out_label({H, W});
    py::array_t<float> out_dist({H, W});
    auto bl = out_label.request();
    auto bd = out_dist.request();
    int *pl = static_cast<int *>(bl.ptr);
    float *pd = static_cast<float *>(bd.ptr);
    for (int i = 0; i < N; ++i)
    {
        pl[i] = label[i];
        pd[i] = dist[i];
    }
    return py::make_tuple(out_label, out_dist, pops);
}

PYBIND11_MODULE(mbd_core, m)
{
    m.doc() = "Minimum Barrier Distance, C++ core";
    m.def("run_mbd_label_propagation", &run_mbd_label_propagation,
          "Run seeded MBD propagation, returns (labels HxW int, dist HxW float, pops int)",
          py::arg("weights"), py::arg("seeds"), py::arg("conn") = 4);
}
