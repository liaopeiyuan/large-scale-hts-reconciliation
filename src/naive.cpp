#include <iostream>
#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

using namespace std;

bool hasChild(xt::xarray<int> hierachy, int i) {
    for (int j = 0; j < hierachy.size(); j ++) {
        if (i == hierachy(j, 0)) {
            return true;
        }
    }
    return false;
}

xt::xarray<int> getS(xt::xarray<int> hierachy, int n) {
    vector<int> bottoms;
    for (int i = 0; i < n; i ++) {
        if(!hasChild(hierachy, i)) {
            bottoms.push_back(i);
        }
    }
    int m = bottoms.size();
    xt::xarray<int> S = xt::zeros({n, m});
    for (int i = 0; i < m; i ++) {
        S(bottoms[i], i) = 1;
    }
    for (int i = hierachy.size() - 1; i >= 0; i ++) {
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
xt::xarray<float> BottomUp(int n, xt::xarray<int> hierachy, xt::xarray<float> yt) {
    xt::xarray<int> S = getS(hierachy, n);
    vector<int> bottoms;
    for (int i = 0; i < n; i ++) {
        if(!hasChild(hierachy, i)) {
            bottoms.push_back(i);
        }
    }
    int m = bottoms.size();
    xt::xarray<int> G = xt::zeros_like(S);
    for (int i = 0; i < m; i ++) {
        G(bottoms[i], i) = 1;
    }
    G = xt::transpose(G);
    xt::xarray<float> res = xt::linalg::dot(S, G);
    res = xt::linalg::dot(res, yt);
    return res;
}

// Top-Down method function that inputs
// S, the summing matrix, yt, the total series,
// and method in ["average_proportions", "proportion_averages", "forecast_proportions"]
// and outputs the series of the whole tree
/*xt::array<float> TopDown(int n, xt::xarray<int> hierachy, xt::xarray<float> yt, string method) {
    //get p
    //get G
    xt::xarray<int> S = getS(hierachy, n);
    xt::xarray<float> G = xt::zeros_like(S);
    xt::xarray<float> prop;
    if (method == "forecast_proportions") {

    }
    if (method == "average_proportions") {
        prop = xt::mean(yt(n - 1) / yt(0), 1);
    } else if (method == "proportion_averages") {
        prop = xt::mean(yt(n - 1), 1) / xt::mean(yt(0));
    }
    G(0) = prop;
    G = xt::transpose(G);
    xt::xarray<float> res = xt::linalg::dot(S, G);
    res = xt::linalg::dot(res, yt);
    return res;
}*/


xt::xarray<float> TopDown(int n, xt::xarray<int> hierachy, xt::xarray<float> yt, 
                            xt::xarray<float> pnodes) {
    //get p
    //get G
    xt::xarray<int> S = getS(hierachy, n);
    vector<int> bottoms;
    for (int i = 0; i < n; i ++) {
        if(!hasChild(hierachy, i)) {
            bottoms.push_back(i);
        }
    }
    int m = bottoms.size();
    xt::xarray<float> G = xt::zeros_like(S);
    xt::xarray<float> phelper = pnodes;
    xt::xarray<float> prop = xt::zeros({1, m});
    for (int i = 0; i < hierachy.size(); i ++) {
        int parent = hierachy(i, 0);
        int child = hierachy(i, 1);
        pnodes(child) *= pnodes(parent);
    }
    for (int i = 0; i < m; i ++) {
        int index = bottom[i];
        prop(i) = pnodes(index);
    }
    G(0) = prop;
    G = xt::transpose(G);
    xt::xarray<float> res = xt::linalg::dot(S, G);
    res = xt::linalg::dot(res, yt);
    return res;
}

xt::xarray<float> MiddleOut(int n, xt::xarray<int> hierachy, xt::xarray<float> yt, 
                            xt::xarray<float> pnodes, int level_start, int level_end) {
    int i = 0;
    while (hierachy(i, 0) != level_start) {
        i += 1;
    }
    xt::xarray buH = xt::view(hierachy, xt::range(_, i));
    xt::xarray buYt = xt::view(yt, xt::range(_, level_start));
    xt::xarray buRes = BottomUp(i, buH, buYt);

    xt::xarray tdH = xt::view(hierachy, xt::range(i, _));
    xt::xarray tdYt = xt::view(yt, xt::range(level_start, _));
    xt::xarray tdP = xt::view(pnodes, xt::range(level_end, _));
    xt::xarray tdRes = TopDown(n - i, tdH, tdYt, tdP);

    xt::xarray res = xt::stack(xt::xtuple(buRes, tdRes));
    return res;
}

xt::xarray<float> OLS(xt::xarray<int> hierachy, xt::xarray<float> yt) {
    //get G
    xt::xarray<int> S = getS(hierachy, n);
    xt::xarray<int> ST = xt::transpose(S);
    xt::xarray<int> G = xt::linalg::dot(ST, S);
    G = xt::linalg::inv(G);
    G = xt::linalg::dot(G, ST);
    xt::xarray<float> res = xt::linalg::dot(S, G);
    res = xt::linalg::dot(res, yt);
    return res;
}

xt::xarray<float> WLS(xt::xarray<int> hierachy, xt::xarray<float> yt, xt::xarray<float> W) {
    //get G
    //get W? diagonal matrix of S?
    xt::xarray<int> S = getS(hierachy, n);
    xt::xarray<int> ST = xt::transpose(S);
    xt::xarray<int> G = xt::linalg::dot(ST, W);
    G = xt::linalg::dot(G, S);
    G = xt::linalg::inv(G);
    G = xt::linalg::dot(G, ST);
    G = xt::linalg::dot(G, W);
    xt::xarray<float> res = xt::linalg::dot(S, G);
    res = xt::linalg::dot(res, yt);
    return res;
}