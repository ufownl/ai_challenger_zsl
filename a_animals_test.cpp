#include "base_network.hpp"
#include "loss_multiattr.hpp"
#include "dataset.hpp"
#include "randomly_crop.hpp"
#include <dlib/dir_nav.h>
#include <dlib/matrix.h>
#include <dlib/data_io.h>
#include <iostream>
#include <fstream>
#include <thread>

using testing_net = loss_multiattr<dlib::sig<base_network<123>::testing_type>>;

int main(int argc, char* argv[]) {
  dlib::set_dnn_prefer_smallest_algorithms();
  auto images = dlib::directory{"ai_challenger_zsl2018_train_test_a_20180321/zsl_a_animals_test_20180321"}.get_files();
  std::ifstream attr_in("ai_challenger_zsl2018_train_test_a_20180321/zsl_a_animals_train_20180321/zsl_a_animals_train_annotations_attributes_per_class_20180321.txt");
  if (!attr_in) {
    std::cerr << "Could not load the testing attributes." << std::endl;
    return 1;
  }
  auto attributes = load_label_attributes(attr_in, 123);
  std::ifstream labels_in("ai_challenger_zsl2018_train_test_a_20180321/zsl_a_animals_train_20180321/zsl_a_animals_train_annotations_label_list_20180321.txt");
  if (!labels_in) {
    std::cerr << "Could not load the training labels." << std::endl;
    return 1;
  }
  auto labels = load_labels(labels_in);
  attributes.erase(std::remove_if(attributes.begin(), attributes.end(), [&](const std::pair<std::string, dlib::matrix<float, 0, 1>>& v) {
    auto it = std::lower_bound(labels.begin(), labels.end(), v.first);
    return it != labels.end() && *it == v.first;
  }), attributes.end());
  testing_net net;
  dlib::deserialize("a_animals_train.resnet34") >> net;
  dlib::rand rnd{std::time(nullptr)};
  for (auto& f: images) {
    dlib::matrix<dlib::rgb_pixel> img;
    dlib::load_image(img, f.full_name());
    dlib::array<dlib::matrix<dlib::rgb_pixel>> crops;
    randomly_crop_images(img, crops, rnd, 16);
    auto ps = net(crops);
    for (auto i = ps.begin() + 1; i != ps.end(); ++i) {
      ps.front() += *i;
    }
    ps.front() /= ps.size();
    auto dis_sqr = std::numeric_limits<float>::max();
    std::string label;
    for (auto& attr: attributes) {
      auto t = dlib::length_squared(ps.front() - attr.second);
      if (t < dis_sqr) {
        dis_sqr = t;
        label = attr.first;
      }
    }
    std::cout << f.name() << " " << label << std::endl;
  }
  return 0;
}
