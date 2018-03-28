#ifndef AI_CHALLENGER_ZSL_DATASET_HPP
#define AI_CHALLENGER_ZSL_DATASET_HPP

#include <istream>
#include <string>
#include <map>
#include <algorithm>

inline std::map<std::string, size_t> load_labels(
  std::istream& in
) {
  std::map<std::string, size_t> ret;
  auto label = 0ul;
  for (std::string line; std::getline(in, line); ) {
    auto it = std::find(line.begin(), line.end(), ',');
    if (it != line.end()) {
      ret.emplace(std::string{line.begin(), it}, label++);
    }
  }
  return ret;
}

inline std::vector<std::pair<std::string, size_t>> load_image_labels(
  std::istream& in,
  const std::map<std::string, size_t> labels
) {
  std::vector<std::pair<std::string, size_t>> ret;
  for (std::string line; std::getline(in, line); ) {
    auto p = std::find(line.begin(), line.end(), ',');
    if (p == line.end()) {
      continue;
    }
    while (*++p == ' ') {
      // nop
    }
    auto label_l = p;
    p = std::find(p, line.end(), ',');
    if (p == line.end()) {
      continue;
    }
    auto label_r = p;
    p = std::find(p, line.end(), ']');
    if (p == line.end()) {
      continue;
    }
    p = std::find(p, line.end(), ',');
    if (p == line.end()) {
      continue;
    }
    while (*++p == ' ') {
      // nop
    }
    auto it = labels.find(std::string{label_l, label_r});
    if (it == labels.end()) {
      continue;
    }
    ret.emplace_back(std::string{p, line.end()}, it->second);
  }
  return ret;
}

#endif  // AI_CHALLENGER_ZSL_DATASET_HPP
