#include <iostream>
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <random>
#include "Eigen"
#include "cnpy.h"

using namespace std;


void cnpy2eigenInt(string data_fname, Eigen::MatrixXi& mat_out){
    cnpy::NpyArray npy_data = cnpy::npy_load(data_fname);
    int data_row = npy_data.shape[0];
    int data_col = npy_data.shape[1];
    int* ptr = static_cast<int *>(malloc(data_row * data_col * sizeof(int)));
    memcpy(ptr, npy_data.data<int>(), data_row * data_col * sizeof(int));
    new (&mat_out) Eigen::Map<Eigen::MatrixXi>(reinterpret_cast<int *>(dmat.data), data_col, data_row);
}

void cnpy2eigen(string data_fname, Eigen::MatrixXf& mat_out){
    cnpy::NpyArray npy_data = cnpy::npy_load(data_fname);
    int data_row = npy_data.shape[0];
    int data_col = npy_data.shape[1];
    float* ptr = static_cast<float *>(malloc(data_row * data_col * sizeof(float)));
    memcpy(ptr, npy_data.data<float>(), data_row * data_col * sizeof(float));
    new (&mat_out) Eigen::Map<Eigen::MatrixXf>(reinterpret_cast<float *>(dmat.data), data_col, data_row);
}

bool hasChild(Eigen::MatrixXi hierachy, int i) {
    for (int j = 0; j < hierachy.rows(); j ++) {
        if (i == hierachy(j, 0)) {
            return true;
        }
    }
    return false;
}

Eigen::MatrixXf getS(Eigen::MatrixXi hierachy) {
    int n = hierachy.coeffs().maxCoeff();
    vector<int> bottoms;
    for (int i = 0; i < n; i ++) {
        if(!hasChild(hierachy, i)) {
            bottoms.push_back(i);
        }
    }
    int m = bottoms.size();
    Eigen::MatrixXf S = MatrixXd::Zero(n, m);
    for (int i = 0; i < m; i ++) {
        S(bottoms[i], i) = 1;
    }
    for (int i = hierachy.rows() - 1; i >= 0; i ++) {
        int parent = hierachy(i, 0);
        int child = hierachy(i, 1);
        S(parent) += S(child);
    }
    return S;
}

// Bottom-Up method function that inputs 
// S, the summing matrix (tree structure) , a matrix
// and bt, the bottom prediction values, a matrix with size of all bottom leaves,
// and outputs the series of the whole tree
Eigen::MatrixXf BottomUp(Eigen::MatrixXi hierachy, Eigen::MatrixXf yt) {
    int n = hierachy.coeffs().maxCoeff();
    Eigen::MatrixXf S = getS(hierachy, n);
    vector<int> bottoms;
    for (int i = 0; i < n; i ++) {
        if(!hasChild(hierachy, i)) {
            bottoms.push_back(i);
        }
    }
    int m = bottoms.size();
    Eigen::MatrixXf G = MatrixXd::Zero(n, m);
    for (int i = 0; i < m; i ++) {
        G(bottoms[i], i) = 1;
    }
    G.transposeInPlace();
    Eigen::MatrixXf res = S * G;
    res = res * yt;
    return res;
}

// Top-Down method function that inputs
// S, the summing matrix, yt, the total series,
// and method in ["average_proportions", "proportion_averages", "forecast_proportions"]
// and outputs the series of the whole tree
/*xt::array<float> TopDown(int n, Eigen::MatrixXf<int> hierachy, Eigen::MatrixXf<float> yt, string method) {
    //get p
    //get G
    Eigen::MatrixXf<int> S = getS(hierachy, n);
    Eigen::MatrixXf<float> G = xt::zeros_like(S);
    Eigen::MatrixXf<float> prop;
    if (method == "forecast_proportions") {

    }
    if (method == "average_proportions") {
        prop = xt::mean(yt(n - 1) / yt(0), 1);
    } else if (method == "proportion_averages") {
        prop = xt::mean(yt(n - 1), 1) / xt::mean(yt(0));
    }
    G(0) = prop;
    G = xt::transpose(G);
    Eigen::MatrixXf<float> res = xt::linalg::dot(S, G);
    res = xt::linalg::dot(res, yt);
    return res;
}*/


Eigen::MatrixXf TopDown(Eigen::MatrixXi hierachy, Eigen::MatrixXf yt, 
                            Eigen::MatrixXf pnodes) {
    //get p
    //get G
    int n = hierachy.coeffs().maxCoeff();
    Eigen::MatrixXf S = getS(hierachy, n);
    vector<int> bottoms;
    for (int i = 0; i < n; i ++) {
        if(!hasChild(hierachy, i)) {
            bottoms.push_back(i);
        }
    }
    int m = bottoms.size();
    Eigen::MatrixXf G = MatrixXd::Zero(n, m);
    Eigen::MatrixXf phelper = pnodes;
    Eigen::MatrixXf prop = MatrixXd::Zero(1, m);
    for (int i = 0; i < hierachy.rows(); i ++) {
        int parent = hierachy(i, 0);
        int child = hierachy(i, 1);
        pnodes(child) *= pnodes(parent);
    }
    for (int i = 0; i < m; i ++) {
        int index = bottom[i];
        prop(i) = pnodes(index);
    }
    G(0) = prop;
    G.transposeInPlace();
    Eigen::MatrixXf res = S * G;
    res = res * yt;
    return res;
}

Eigen::MatrixXf MiddleOut(Eigen::MatrixXi hierachy, Eigen::MatrixXf yt, 
                            Eigen::MatrixXf pnodes, int level_start, int level_end) {
    int n = hierachy.coeffs().maxCoeff();
    int i = 0;
    while (hierachy(i, 0) != level_start) {
        i += 1;
    }
    Eigen::MatrixXi buH = hierachy.block(0, 0, i, 2);
    Eigen::MatrixXf buYt = yt.block(0, 0, level_start, yt.cols());
    Eigen::MatrixXf buRes = BottomUp(i, buH, buYt);

    Eigen::MatrixXi tdH = hierachy.block(i, 0, hierachy.rows(), 2);
    Eigen::MatrixXf tdYt = yt.block(level_start, 0, yt.rows(), yt.cols());
    Eigen::MatrixXf tdP = pnodes.block(level_end, 0, pnodes.rows(), pnodes.cols());
    Eigen::MatrixXf tdRes = TopDown(n - i, tdH, tdYt, tdP);
    Eigen::MatrixXf res(buRes.rows() + tdRes.rows(), buRes.cols());
    res.topRows(buRes.rows()) = buRes;
    res.bottomRows(tdRes.rows()) = tdRes;
    return res;
}

Eigen::MatrixXf OLS(Eigen::MatrixXi hierachy, Eigen::MatrixXf yt) {
    //get G
    int n = hierachy.coeffs().maxCoeff();
    Eigen::MatrixXf S = getS(hierachy, n);
    Eigen::MatrixXf ST = S.transpose();
    Eigen::MatrixXf G = ST * S;
    G = G.inverse();
    G = G * ST;
    Eigen::MatrixXf res = S * G;
    res = res * yt;
    return res;
}

Eigen::MatrixXf WLS(Eigen::MatrixXi hierachy, Eigen::MatrixXf yt, Eigen::MatrixXf W) {
    //get G
    //get W? diagonal matrix of S?
    int n = hierachy.coeffs().maxCoeff();
    Eigen::MatrixXf S = getS(hierachy, n);
    Eigen::MatrixXf ST = S.transpose();
    Eigen::MatrixXf G = ST * W;
    G = G * S;
    G = G.inverse();
    G = G * ST;
    G = G * W;
    Eigen::MatrixXf res = S * G;
    res = res * yt;
    return res;
}

int main() {
    // input in node format? parse node?
    // input in array format? array to eigen matrix
    string hierachy_filename;
    cout << "hierachy filename: ";
    cin >> hierachy_filename;
    Eigen::MatrixXi hierachy;
    cnpy2eigenInt(hierachy_filename, hierachy);

    string yt_filename;
    cout << "yt filename: ";
    cin >> yt_filename;
    Eigen::MatrixXf yt;
    cnpy2eigenf(yt_filename, yt);

    string method;
    cout << "method: ";
    cin >> method;
    
    Eigen::MatrixXf res;

    if (method == "bottom-up") {
        res = BottomUp(hierachy, yt);
    } 
    if (method == "top-down") {
        //get pnodes? as file?
        res = TopDown(hierachy, yt, pnodes);
    }
    if (method == "middle-out") {
        res = MiddleOut(hierachy, yt, pnodes, level_start, level_end);
    }
    if (method == "OLS") {
        res = OLS(hierachy, yt);
    }
    if (method == "WLS") {
        //W as file?
        res = WLS(hierachy, yt, W);
    }

}
