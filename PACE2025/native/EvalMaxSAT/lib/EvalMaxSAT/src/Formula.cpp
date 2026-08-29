#include "Formula.hpp"
#include "cadicalinterface.h"
#include "EvalMaxSAT.h"
namespace Eval
{
  
  Formula::Formula() {
    monMaxSat = new EvalMaxSAT<Solver_cadical>();
  }
  
  Formula::~Formula() {
    EvalMaxSAT<Solver_cadical> *monMaxSat = (EvalMaxSAT<Solver_cadical> *)this->monMaxSat;
    delete monMaxSat;
    this->monMaxSat = nullptr;
  }

  int Formula::addClause(const int ps[], size_t length, long w) {
    EvalMaxSAT<Solver_cadical> *monMaxSat = (EvalMaxSAT<Solver_cadical> *)this->monMaxSat;
    std::vector<int> clause;
    for (size_t i = 0; i < length; i++) {
      int lit = ps[i];
      while (abs(lit) > monMaxSat->nVars()) monMaxSat->newVar();
      clause.push_back(lit);
    }
    
    std::optional<long long> ww = {};
    if (w > 0) { ww = w; }
    return monMaxSat->addClause(clause, ww);
  }

  void Formula::setVertexMapping(
    const unsigned int* mapping,
    size_t length,
    const unsigned int* edge_data,
    const size_t* edge_offsets,
    size_t num_edges,
    const double* features,
    size_t num_features) {

    EvalMaxSAT<Solver_cadical> *monMaxSat =
        (EvalMaxSAT<Solver_cadical> *)this->monMaxSat;

    std::vector<unsigned int> vertex_mapping(
        mapping,
        mapping + length);

    std::vector<std::vector<unsigned int>> edges(num_edges);

    for (size_t i = 0; i < num_edges; ++i) {
        for (size_t j = edge_offsets[i];
             j < edge_offsets[i + 1];
             ++j) {
            edges[i].push_back(edge_data[j]);
        }
    }


    monMaxSat->setVertexMapping(vertex_mapping, edges);
    std::vector<double> static_features(
        features,
        features + num_features);

    monMaxSat->setVertexFeatures(static_features);
    }

  void Formula::setMLBranch(bool enabled) {
    EvalMaxSAT<Solver_cadical> *monMaxSat =
        (EvalMaxSAT<Solver_cadical> *)this->monMaxSat;

    monMaxSat->setMLBranch(enabled);
}

  bool Formula::getValue(int lit) {
    EvalMaxSAT<Solver_cadical> *monMaxSat = (EvalMaxSAT<Solver_cadical> *)this->monMaxSat;
    return monMaxSat->getValue(lit);
  }

  void Formula::printStats() {
    EvalMaxSAT<Solver_cadical> *monMaxSat =
        (EvalMaxSAT<Solver_cadical> *)this->monMaxSat;

    monMaxSat->printStats();
  }

  bool Formula::solve() {
    EvalMaxSAT<Solver_cadical> *monMaxSat = (EvalMaxSAT<Solver_cadical> *)this->monMaxSat;
    return monMaxSat->solve();
  }

  
};
