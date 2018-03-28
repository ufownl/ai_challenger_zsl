#include "base_network.hpp"
#include "dataset.hpp"
#include "randomly_crop.hpp"
#include <dlib/data_io.h>
#include <iostream>
#include <fstream>
#include <thread>

using training_net = dlib::loss_multiclass_log<base_network<40>::training_type>;

int main(int argc, char* argv[]) {
  dlib::set_dnn_prefer_smallest_algorithms();

  size_t threshold = 8000;
  if (argc >= 2) {
    threshold = atol(argv[1]);
  }

  std::cout << "Loading dataset..." << std::endl;
  std::ifstream labels_in("ai_challenger_zsl2018_train_test_a_20180321/zsl_a_animals_train_20180321/zsl_a_animals_train_annotations_label_list_20180321.txt");
  std::ifstream images_in("ai_challenger_zsl2018_train_test_a_20180321/zsl_a_animals_train_20180321/zsl_a_animals_train_annotations_labels_20180321.txt");
  if (!labels_in || !images_in) {
    std::cerr << "Could not load the dataset." << std::endl;
    return 1;
  }
  auto images = load_image_labels(images_in, load_labels(labels_in));
  std::cout << "images in dataset: " << images.size() << std::endl;

  constexpr auto initial_learning_rate = 0.1;
  training_net net;
  dlib::dnn_trainer<training_net> trainer(net, dlib::sgd{0.0001f, 0.9f});
  trainer.be_verbose();
  trainer.set_learning_rate(initial_learning_rate);
  trainer.set_synchronization_file("a_animals_pre_train.state", std::chrono::minutes{10});
  trainer.set_iterations_without_progress_threshold(threshold);
  set_all_bn_running_stats_window_sizes(net, 1000);

  dlib::pipe<std::pair<dlib::matrix<dlib::rgb_pixel>, size_t>> data{200};
  auto load_data = [&data, &images](time_t seed) {
    dlib::rand rnd(std::time(nullptr) + seed);
    while(data.is_enabled()) {
      auto inf = images[rnd.get_random_32bit_number() % images.size()];
      dlib::matrix<dlib::rgb_pixel> img;
      load_image(img, "ai_challenger_zsl2018_train_test_a_20180321/zsl_a_animals_train_20180321/zsl_a_animals_train_images_20180321/" + inf.first);
      dlib::matrix<dlib::rgb_pixel> crop;
      randomly_crop_image(img, crop, rnd);
      data.enqueue(std::make_pair(std::move(crop), inf.second));
    }
  };
  auto loader_num = std::thread::hardware_concurrency();
  std::vector<std::thread> data_loaders;
  data_loaders.reserve(loader_num);
  for (auto i = 0u; i < loader_num; ++i) {
    data_loaders.emplace_back([load_data, i] {
      load_data(i);
    });
  }
  std::vector<dlib::matrix<dlib::rgb_pixel>> samples;
  std::vector<unsigned long> labels;
  while(trainer.get_learning_rate() >= initial_learning_rate * 1e-3) {
    while(samples.size() < 64) {
      std::pair<dlib::matrix<dlib::rgb_pixel>, size_t> img;
      data.dequeue(img);
      samples.emplace_back(std::move(img.first));
      labels.emplace_back(img.second);
    }
    trainer.train_one_step(samples, labels);
    samples.clear();
    labels.clear();
  }
  for (auto& t: data_loaders) {
    t.join();
  }
  trainer.get_net();
  net.clean();

  std::cout << "Saving network..." << std::endl;
  dlib::serialize("a_animals_pre_train.resnet34") << net;
  return 0;
}
