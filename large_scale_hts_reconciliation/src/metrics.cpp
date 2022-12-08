#include "metrics.h"

namespace lhts {
namespace metrics {
float rmse(const MatrixXf res, const MatrixXf gt) {
  float sum = 0;
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      sum += pow(abs(res(i, j) - gt(i, j)), 2);
    }
  }
  float rmse = sqrt(sum / (res.rows() * res.cols()));
  return rmse;
}

float mae(const MatrixXf res, const MatrixXf gt) {
  float sum = 0;
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      sum += abs(res(i, j) - gt(i, j));
    }
  }
  float mae = sum / (res.rows() * res.cols());
  return mae;
}

float smape(const MatrixXf res, const MatrixXf gt) {
  float sum = 0;
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      float pd = res(i, j);
      float gtr = gt(i, j);
      float abse = abs(pd - gtr);
      float mean = (abs(pd) + abs(gtr)) / 2;
      if (mean == 0) {
        sum += 0;
      } else {
        float val = abse / mean;
        sum += val;
      }
    }
  }
  float smape = sum / (res.rows() * res.cols());
  return smape;
}

}  // namespace metrics
}  // namespace lhts