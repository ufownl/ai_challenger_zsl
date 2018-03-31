#ifndef AI_CHALLENGER_ZSL_VALIDATOR_HPP
#define AI_CHALLENGER_ZSL_VALIDATOR_HPP

#include <dlib/data_io.h>
#include <limits>
#include <numeric>

template <size_t N>
class validator {
public:
  using net_type = loss_multiattr<dlib::sig<typename base_network<N>::testing_type>>;

  template <class T>
  explicit validator(const T& net)
    : net_(net) {
    // nop
  } 

  void load_dataset(
    const std::string& labels_file,
    const std::string& images_file,
    const std::string& attributes_file
  ) {
    load_labels_(labels_file);
    load_images_(images_file);
    load_attributes_(attributes_file);
  } 

  void run(const std::string& images_root) {
    dlib::rand rnd{std::time(nullptr)};
    for (auto& inf: images_) {
      dlib::matrix<dlib::rgb_pixel> img;
      dlib::load_image(img, images_root + inf.first);
      dlib::array<dlib::matrix<dlib::rgb_pixel>> crops;
      randomly_crop_images(img, crops, rnd, 16);
      auto ps = net_(crops);
      auto p = std::accumulate(ps.begin() + 1, ps.end(), ps.front()) / ps.size();
      auto dis_sqr = std::numeric_limits<float>::max();
      std::string label;
      for (auto& attr: attributes_) {
        auto t = dlib::length_squared(p - attr.second);
        if (t < dis_sqr) {
          dis_sqr = t;
          label = attr.first;
        }
      }
      auto it = std::lower_bound(labels_.begin(), labels_.end(), label);
      if (static_cast<size_t>(std::distance(labels_.begin(), it)) == inf.second) {
        ++right_num_;
      } else {
        ++wrong_num_;
      }
    }
  } 

  size_t right_num() const { return right_num_; }
  size_t wrong_num() const { return wrong_num_; }

private:
  void load_labels_(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
      std::cerr << "Could not load the validating labels." << std::endl;
      return;
    }
    labels_ = load_labels(in);
  }

  void load_images_(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
      std::cerr << "Could not load the validating images." << std::endl;
      return;
    }
    images_ = load_image_labels(in, labels_);
  }

  void load_attributes_(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
      std::cerr << "Could not load the validating attributes." << std::endl;
      return;
    }
    attributes_ = load_label_attributes(in, N);
  }

  net_type net_;
  std::vector<std::string> labels_;
  std::vector<std::pair<std::string, size_t>> images_;
  std::vector<std::pair<std::string, dlib::matrix<float, 0, 1>>> attributes_;
  size_t right_num_ {0};
  size_t wrong_num_ {0};
};

#endif  // AI_CHALLENGER_ZSL_VALIDATOR_HPP
