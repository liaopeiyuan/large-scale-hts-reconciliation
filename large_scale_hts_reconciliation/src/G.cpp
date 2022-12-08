#include "G.h"
namespace lhts {
    namespace G {
        
MatrixXf build_sparse_OLS(SpMat Sp) {
    SpMat St = Sp.transpose().eval();
    MatrixXf M = MatrixXf(St * Sp);
    FullPivLU<MatrixXf> lu(M);
    return lu.matrixLU() * St;
}

MatrixXf build_sparse_WLS(const MatrixXi S, float w) {
    MatrixXf W = MatrixXf::Zero(S.rows(), S.cols());
    #pragma omp parallel for 
    for (int i = 0; i < S.rows(); i++) {
        W(i, i) = w;
    }
    MatrixXf Sp = S.cast<float>();
    MatrixXf St = Sp.transpose();
    MatrixXf M = St * W * Sp;
    FullPivLU<MatrixXf> lu(M);
    return lu.matrixLU() * St * W;
}


SpMat build_sparse_top_down(const MatrixXi S_compact, 
                const MatrixXf P, 
                int num_base, int num_total, int num_levels) {
    SpMat G(num_base, num_total);
    
    assert(S_compact.rows() == num_total);
    assert(S_compact.cols() == num_levels);
    assert(num_levels > 1);

    std::vector<T> tripletList;

    // #pragma omp parallel for 
    for (int i = 0; i < num_total; i++) {
        int co = S_compact(i, 0);
        int root = -1;
        bool is_base = true;
        for (int j = 1; j < num_levels; j++) {
            int ro = S_compact(i, j);
            if (ro == -1) {
                is_base = false;
                break;
            }
            root = ro;
        }
        if (is_base) {
            tripletList.push_back(T(co, root, P(co, 0)));
        }
    }

    G.setFromTriplets(tripletList.begin(), tripletList.end());

    return G;
}


SpMat build_sparse_middle_out(const MatrixXi S_compact, 
                const MatrixXf P, int level, 
                int num_base, int num_total, int num_levels) {
    SpMat G(num_base, num_total);
    
    assert(S_compact.rows() == num_total);
    assert(S_compact.cols() == num_levels);
    assert(num_levels > 1);

    std::vector<T> tripletList;

    // #pragma omp parallel for 
    for (int i = 0; i < num_total; i++) {
        int co = S_compact(i, 0);
        int root = co;
        int lvl = num_levels - level;
        bool is_base = true;
        for (int j = 1; j < num_levels; j++) {
            int ro = S_compact(i, j);
            if (ro == -1) {
                is_base = false;
                break;
            }
            if (lvl > 0) {
                root = ro;
                lvl--;
            }
        }
        if (is_base) {
            tripletList.push_back(T(co, root, P(co, 0)));
        }
    }

    G.setFromTriplets(tripletList.begin(), tripletList.end());

    return G;
}


SpMat build_sparse_bottom_up(const MatrixXi S_compact, int num_base, int num_total, int num_levels) {
    SpMat G(num_base, num_total);
    
    assert(S_compact.rows() == num_total);
    assert(S_compact.cols() == num_levels);
    assert(num_levels > 1);

    std::vector<T> tripletList;

    // #pragma omp parallel for 
    for (int i = 0; i < num_total; i++) {
        int co = S_compact(i, 0);
        bool is_base = true;
        for (int j = 1; j < num_levels; j++) {
            int ro = S_compact(i, j);
            if (ro == -1) {
                is_base = false;
                break;
            }
        }
        if (is_base) {
            tripletList.push_back(T(co, i, 1.0));
        }
    }

    G.setFromTriplets(tripletList.begin(), tripletList.end());

    return G;
}

    }

}