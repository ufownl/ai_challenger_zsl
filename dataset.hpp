#ifndef AI_CHALLENGER_ZSL_DATASET_HPP
#define AI_CHALLENGER_ZSL_DATASET_HPP

#include <dlib/matrix.h>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

inline std::vector<std::string> load_labels(
  std::istream& in
) {
  std::vector<std::string> ret;
  for (std::string line; std::getline(in, line); ) {
    auto it = std::find(line.begin(), line.end(), ',');
    if (it != line.end()) {
      ret.emplace_back(line.begin(), it);
    }
  }
  std::sort(ret.begin(), ret.end());
  return ret;
}

inline std::vector<std::pair<std::string, size_t>> load_image_labels(
  std::istream& in,
  const std::vector<std::string> labels
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
    auto it = std::lower_bound(labels.begin(), labels.end(), std::string{label_l, label_r});
    if (it == labels.end()) {
      continue;
    }
    ret.emplace_back(std::string{p, line.end()}, std::distance(labels.begin(), it));
  }
  return ret;
}

inline std::vector<std::pair<std::string, dlib::matrix<float, 0, 1>>> load_image_attributes(
  std::istream& in,
  long attr_num
) {
  std::vector<std::pair<std::string, dlib::matrix<float, 0, 1>>> ret;
  for (std::string line; std::getline(in, line); ) {
    auto p = std::find(line.begin(), line.end(), ',');
    if (p == line.end()) {
      continue;
    }
    while (*++p == ' ') {
      // nop
    }
    auto file_l = p;
    p = std::find(p, line.end(), ',');
    if (p == line.end()) {
      continue;
    }
    auto file_r = p;
    p = std::find(p, line.end(), '[');
    if (p == line.end()) {
      continue;
    }
    auto attr_l = ++p;
    p = std::find(p, line.end(), ']');
    if (p == line.end()) {
      continue;
    }
    dlib::matrix<float, 0, 1> attr(attr_num);
    std::stringstream ss{std::string{attr_l, p}, std::ios_base::in};
    for (auto i = 0l; i < attr_num; ++i) {
      ss >> attr(i);
    }
    ret.emplace_back(std::string{file_l, file_r}, std::move(attr));
  }
  return ret;
}

inline std::vector<std::pair<std::string, dlib::matrix<float, 0, 1>>> load_label_attributes(
  std::istream& in,
  long attr_num
) {
  std::vector<std::pair<std::string, dlib::matrix<float, 0, 1>>> ret;
  for (std::string line; std::getline(in, line); ) {
    auto p = std::find(line.begin(), line.end(), ',');
    if (p == line.end()) {
      continue;
    }
    auto label_r = p;
    p = std::find(p, line.end(), '[');
    if (p == line.end()) {
      continue;
    }
    auto attr_l = ++p;
    p = std::find(p, line.end(), ']');
    if (p == line.end()) {
      continue;
    }
    dlib::matrix<float, 0, 1> attr(attr_num);
    std::stringstream ss{std::string{attr_l, p}, std::ios_base::in};
    for (auto i = 0l; i < attr_num; ++i) {
      ss >> attr(i);
    }
    ret.emplace_back(std::string{line.begin(), label_r}, std::move(attr));
  }
  return ret;
}

#endif  // AI_CHALLENGER_ZSL_DATASET_HPP
