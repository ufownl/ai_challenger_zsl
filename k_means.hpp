#ifndef AI_CHALLENGER_ZSL_K_MEANS_HPP
#define AI_CHALLENGER_ZSL_K_MEANS_HPP

#include <dlib/matrix.h>
#include <vector>
#include <numeric>
#include <utility>
#include <cassert>

namespace k_means {

template <class T>
struct point {
  using value_type = dlib::matrix<T, 0, 1>;

  value_type value;
  size_t cluster;

  point() = default;

  point(long d, size_t c)
    : value(d)
    , cluster(c) {
    for (auto i = 0l; i < value.nr(); ++i) {
      value(i) = T{};
    }
  }

  point(value_type v, size_t c)
    : value(std::move(v))
    , cluster(c) {
    // nop
  }
};

template <class T>
using point_cloud = std::vector<point<T>>;

namespace {

template <class T>
point_cloud<T> calc_centers(size_t kv, const point_cloud<T>& points) {
  assert(!points.empty());
  point_cloud<T> centers;
  centers.reserve(kv);
  for (auto i = 0ul; i < kv; ++i) {
    centers.emplace_back(points.front().value.nr(), i);
  }
  std::vector<size_t> pt_cnt(kv);
  for (auto& pt: points) {
    centers[pt.cluster].value += pt.value;
    ++pt_cnt[pt.cluster];
  }
  for (auto i = 0ul; i < kv; ++i) {
    centers[i].value /= pt_cnt[i];
  }
  return centers;
}

template <class T>
bool is_constrict(const point_cloud<T>& lhs, const point_cloud<T>& rhs, T threshold) {
  assert(lhs.size() == rhs.size());
  for (auto i = 0ul; i < lhs.size(); ++i) {
    if (dlib::length_squared(lhs[i].value - rhs[i].value) > threshold * threshold) {
      return false;
    }
  }
  return true;
}

}

template <class T>
point_cloud<T> cluster(size_t kv, point_cloud<T>& points, point_cloud<T> seeds, T threshold) {
  assert(!points.empty());
  for (;;) {
    for (auto& pt: points) {
      auto min_dist = std::numeric_limits<T>::max();
      for (auto& sd: seeds) {
        auto dist = dlib::length_squared(pt.value - sd.value);
        if (dist < min_dist) {
          pt.cluster = sd.cluster;
          min_dist = dist;
        }
      }
    }
    auto centers = calc_centers(kv, points);
    if (is_constrict(centers, seeds, threshold)) {
      return centers;
    }
    using std::swap;
    swap(seeds, centers);
  }
}

}

#endif  // AI_CHALLENGER_ZSL_K_MEANS_HPP
