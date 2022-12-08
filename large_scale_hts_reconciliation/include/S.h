#include <vector>
#include <set>
#include <tuple>
#include <stdexcept>

#include <Eigen/LU>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<float, Eigen::ColMajor> SpMat;
typedef Eigen::Triplet<float> T;

namespace lhts {
    namespace S {
        SpMat build_sparse(const Eigen::MatrixXi S_compact, int num_base, int num_total, int num_levels)
    }
}