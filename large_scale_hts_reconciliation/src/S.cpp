#include "S.h"

namespace lhts {
    namespace S {
        SpMat build_sparse(const Eigen::MatrixXi S_compact, int num_base, int num_total, int num_levels) {
            SpMat S(num_total, num_base);
            
            std::vector<T> tripletList;

            assert(S_compact.rows() == num_total);
            assert(S_compact.cols() == num_levels);
            assert(num_levels > 1);

            for (int i = 0; i < num_base; i++) {
                int co = S_compact(i, 0);
                if (co >= num_base) {
                    throw std::invalid_argument("Make sure that the frist num_base rows of S_compact contain only leaf-level nodes.");
                }
                tripletList.push_back(T(co, co, 1));
                for (int j = 1; j < num_levels; j++) {
                    int ro = S_compact(i, j);
                    if (ro == -1) {
                        if (i < num_base) {
                            throw std::invalid_argument("Make sure that the frist num_base rows of S_compact contain only leaf-level nodes.");
                        }
                        break;
                    } else {
                        if (co >= num_base) {
                            throw std::invalid_argument("Make sure that the all leaf-level nodes have index < num_base.");
                        }
                        tripletList.push_back(T(ro, co, 1));
                    }
                }
            }

            S.setFromTriplets(tripletList.begin(), tripletList.end());

            return S;
        }
    }
}