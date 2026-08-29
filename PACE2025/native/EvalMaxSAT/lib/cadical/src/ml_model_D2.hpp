#ifndef ML_MODEL_HPP
#define ML_MODEL_HPP

namespace CaDiCaL {

static inline double ml_predict(const double *x) {
  double score = -6.876082e-08;

  score += (x[0] < 3 ? (x[4] < 0.00102673 ? 0.00586901 : -0.0042054164000000002) : (x[1] < 7 ? -0.018433201999999999 : -0.00044933970000000001));

  return score;
}

}
#endif
