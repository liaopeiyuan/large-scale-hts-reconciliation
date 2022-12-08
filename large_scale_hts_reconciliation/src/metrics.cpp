#include "metrics.h"

namespace lhts {
namespace metrics {
double rmse(const MatrixXd res, const MatrixXd gt) {
  double sum = 0;
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      sum += pow(abs(res(i, j) - gt(i, j)), 2);
    }
  }
  double rmse = sqrt(sum / (res.rows() * res.cols()));
  return rmse;
}

double mae(const MatrixXd res, const MatrixXd gt) {
  double sum = 0;
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      sum += abs(res(i, j) - gt(i, j));
    }
  }
  double mae = sum / (res.rows() * res.cols());
  return mae;
}

double smape(const MatrixXd res, const MatrixXd gt) {
  double sum = 0;
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      double pd = res(i, j);
      double gtr = gt(i, j);
      double abse = abs(pd - gtr);
      double mean = (abs(pd) + abs(gtr)) / 2;
      if (mean == 0) {
        sum += 0;
      } else {
        double val = abse / mean;
        sum += val;
      }
    }
  }
  double smape = sum / (res.rows() * res.cols());
  return smape;
}

}  // namespace metrics
}  // namespace lhts