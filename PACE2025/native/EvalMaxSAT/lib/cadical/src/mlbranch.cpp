#include "internal.hpp"
#include <limits>
#include "ml_model.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace CaDiCaL {

void Internal::collect_branch_sample (int decision_variable) {

  static std::ofstream out ("ml_branch_samples.csv", std::ios::app);
  static uint64_t decision_id = 0;
  if (decision_id >= 25000)
    return;

  if (!out) return;

  const int chosen = abs (decision_variable);

  if (chosen <= 0 || chosen > max_var) return;

  if (chosen >= (int) vertex_mapping.size ()) return;

  const unsigned int chosen_original = vertex_mapping[chosen];

  if (chosen_original * 12 + 11 >= vertex_features.size ())
    return;

  const size_t target = 32;

  std::vector<int> selected;
  selected.reserve (target);

  selected.push_back (chosen);

  auto valid_candidate = [this] (int v) {

    if (v <= 0 || v > max_var) return false;
    if (val (v)) return false;
    if (v >= (int) vertex_mapping.size ()) return false;

    const unsigned int original_vertex = vertex_mapping[v];

    if (original_vertex * 12 + 11 >= vertex_features.size ())
      return false;

    return true;
  };

  if (use_scores ()) {

    ScoreSchedule tmp = scores;

    while (!tmp.empty () && selected.size () < target) {

      const int v = tmp.pop_front ();

      if (!valid_candidate (v))
        continue;

      if (v == chosen)
        continue;

      selected.push_back (v);
    }

  } else {

    for (int v = queue.unassigned;
         v && selected.size () < target;
         v = link (v).prev) {

      if (!valid_candidate (v))
        continue;

      if (v == chosen)
        continue;

      selected.push_back (v);
    }
  }

  if (selected.size () != target)
    return;

  const uint64_t group = decision_id++;

  for (const int v : selected) {

    const unsigned int original_vertex = vertex_mapping[v];

    double x[19];

    for (int j = 0; j < 12; ++j)
      x[j] = vertex_features[original_vertex * 12 + j];

    x[12] = level;
    x[13] = trail.size ();
    x[14] = propagated;
    x[15] = stats.conflicts;
    x[16] = std::log1p (stab[v]);
    x[17] = val (v) != 0;
    x[18] = val (v) ? var(v).level : 0;

    const double ml_score = ml_predict (x);

    const double evsids = stab[v];

    const int assigned = val (v) != 0;

    int assignment_level = 0;

    if (assigned)
      assignment_level = var(v).level;

    const int label = (v == chosen) ? 1 : 0;

    std::ostringstream row;

    row << group << ","
        << v << ","
        << original_vertex << ",";

    for (int j = 0; j < 12; ++j) {

      row << vertex_features[original_vertex * 12 + j];

      if (j < 11)
        row << ",";
    }

    row << ","
        << level << ","
        << trail.size () << ","
        << propagated << ","
        << stats.conflicts << ","
        << evsids << ","
        << assigned << ","
        << assignment_level << ","
        << ml_score << ","
        << label
        << "\n";

    out << row.str ();
  }
}

int Internal::next_decision_variable_ml () {

  const int native = next_decision_variable ();
  const char *env = std::getenv("ML_DEPTH");
  const int ml_depth = env ? std::atoi(env) : 2;
  if (level >= ml_depth)
    return native;
  int best = native;
  double best_score =
      -std::numeric_limits<double>::infinity ();

  if (use_scores ()) {

    for (auto it = scores.begin ();
         it != scores.end ();
         ++it) {

      const int v = *it;

      if (v <= 0 || v > max_var) continue;
      if (val (v)) continue;
      if (!active (v)) continue;
      if (v >= (int) vertex_mapping.size ()) continue;

      const unsigned int original_vertex =
          vertex_mapping[v];

      if (original_vertex * 12 + 11 >= vertex_features.size ())
        continue;

      double x[19];

      for (int j = 0; j < 12; ++j)
        x[j] = vertex_features[original_vertex * 12 + j];

      x[12] = level;
      x[13] = trail.size ();
      x[14] = propagated;
      x[15] = stats.conflicts;
      x[16] = std::log1p (stab[v]);
      x[17] = 0;
      x[18] = 0;

      const double score = ml_predict (x);

      if (score > best_score) {
        best_score = score;
        best = v;
      }
    }

  } else {

    for (int v = queue.last;
         v;
         v = link (v).prev) {

      if (v <= 0 || v > max_var) continue;
      if (val (v)) continue;
      if (!active (v)) continue;
      if (v >= (int) vertex_mapping.size ()) continue;

      const unsigned int original_vertex =
          vertex_mapping[v];

      if (original_vertex * 12 + 11 >= vertex_features.size ())
        continue;

      double x[19];

      for (int j = 0; j < 12; ++j)
        x[j] = vertex_features[original_vertex * 12 + j];

      x[12] = level;
      x[13] = trail.size ();
      x[14] = propagated;
      x[15] = stats.conflicts;
      x[16] = std::log1p (stab[v]);
      x[17] = 0;
      x[18] = 0;

      const double score = ml_predict (x);

      if (score > best_score) {
        best_score = score;
        best = v;
      }
    }
  }

  return best;
}
}
