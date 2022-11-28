#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <random>
#include "Eigen"
#include "cnpy.h"

using namespace std;

bool hasChild(Eigen::MatrixXf hierachy, int i) {
    for (int j = 0; j < hierachy.rows(); j ++) {
        if (i == hierachy(j, 0)) {
            return true;
        }
    }
    return false;
}

Eigen::MatrixXf getS(Eigen::MatrixXf hierachy, int n) {
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
        float parent = hierachy(i, 0);
        float child = hierachy(i, 1);
        S(parent) += S(child);
    }
    return S;
}

// Bottom-Up method function that inputs 
// S, the summing matrix (tree structure) , a matrix
// and bt, the bottom prediction values, a matrix with size of all bottom leaves,
// and outputs the series of the whole tree
Eigen::MatrixXf BottomUp(int n, Eigen::MatrixXf hierachy, Eigen::MatrixXf yt) {
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


Eigen::MatrixXf TopDown(int n, Eigen::MatrixXf hierachy, Eigen::MatrixXf yt, 
                            Eigen::MatrixXf pnodes) {
    //get p
    //get G
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
        float parent = hierachy(i, 0);
        float child = hierachy(i, 1);
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

Eigen::MatrixXf MiddleOut(int n, Eigen::MatrixXf hierachy, Eigen::MatrixXf yt, 
                            Eigen::MatrixXf pnodes, int level_start, int level_end) {
    int i = 0;
    while (hierachy(i, 0) != level_start) {
        i += 1;
    }
    Eigen::MatrixXf buH = hierachy.block(0, 0, i, 2);
    Eigen::MatrixXf buYt = yt.block(0, 0, level_start, yt.cols());
    Eigen::MatrixXf buRes = BottomUp(i, buH, buYt);

    Eigen::MatrixXf tdH = hierachy.block(i, 0, hierachy.rows(), 2);
    Eigen::MatrixXf tdYt = yt.block(level_start, 0, yt.rows(), yt.cols());
    Eigen::MatrixXf tdP = pnodes.block(level_end, 0, pnodes.rows(), pnodes.cols());
    Eigen::MatrixXf tdRes = TopDown(n - i, tdH, tdYt, tdP);
    Eigen::MatrixXf res(buRes.rows() + tdRes.rows(), buRes.cols());
    res.topRows(buRes.rows()) = buRes;
    res.bottomRows(tdRes.rows()) = tdRes;
    return res;
}

Eigen::MatrixXf OLS(Eigen::MatrixXf hierachy, Eigen::MatrixXf yt) {
    //get G
    Eigen::MatrixXf S = getS(hierachy, n);
    Eigen::MatrixXf ST = S.transpose();
    Eigen::MatrixXf G = ST * S;
    G = G.inverse();
    G = G * ST;
    Eigen::MatrixXf res = S * G;
    res = res * yt;
    return res;
}

Eigen::MatrixXf<float> WLS(Eigen::MatrixXf<int> hierachy, Eigen::MatrixXf<float> yt, Eigen::MatrixXf<float> W) {
    //get G
    //get W? diagonal matrix of S?
    Eigen::MatrixXf<int> S = getS(hierachy, n);
    Eigen::MatrixXf<int> ST = S.transpose();
    Eigen::MatrixXf<int> G = ST * W;
    G = G * S;
    G = G.inverse();
    G = G * ST;
    G = G * W;
    Eigen::MatrixXf res = S * G;
    res = res * yt;
    return res;
}
