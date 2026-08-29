#ifndef Formula_H_
#define Formula_H_
#include <cstdint>
#include <cstdio>
#include <optional>
#include <vector>

namespace Eval
{
  class Formula
  {
  private:
    void *monMaxSat;

  public:
    Formula();
    ~Formula();


    int addClause(const int ps[], size_t length, long w = 0);
    void setVertexMapping(
    const unsigned int* mapping,
    size_t length,
    const unsigned int* edge_data,
    const size_t* edge_offsets,
    size_t num_edges,
    const double* features,
    size_t num_features);
    bool getValue(int lit);
    void setMLBranch(bool enabled);
    bool solve();
    void printStats();
  };
};
#endif
