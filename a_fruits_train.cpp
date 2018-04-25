#include "base_network.hpp"
#include "loss_multiattr.hpp"
#include "dataset.hpp"
#include "randomly_crop.hpp"
#include "validator.hpp"
#include <iostream>
#include <fstream>
#include <thread>

using pre_trained_net = dlib::loss_multiclass_log<base_network<40>::training_type>;
using training_net = loss_multiattr<dlib::sig<base_network<58>::training_type>>;

int main(int argc, char* argv[]) {
  dlib::set_dnn_prefer_smallest_algorithms();

  size_t batch_size = 64;
  if (argc >= 2) {
    batch_size = atol(argv[1]);
  }

  std::cout << "Loading dataset..." << std::endl;
  std::ifstream images_in("ai_challenger_zsl2018_train_test_a_20180321/zsl_a_fruits_train_20180321/zsl_a_fruits_train_annotations_attributes_20180321.txt");
  if (!images_in) {
    std::cerr << "Could not load the dataset." << std::endl;
    return 1;
  }
  auto images = load_image_attributes(images_in, 58);
  std::cout << "images in dataset: " << images.size() << std::endl;

  constexpr auto initial_learning_rate = 0.1;
  pre_trained_net pnet;
  dlib::deserialize("models/a_fruits_pre_train.resnet34") >> pnet;
  dlib::visit_layers_range<2, pre_trained_net::num_layers>(pnet, zero_learning_rate{});
  training_net net;
  dlib::layer<3>(net) = dlib::layer<2>(pnet);
  dlib::dnn_trainer<training_net> trainer(net, dlib::sgd{0.0001f, 0.9f});
  trainer.be_verbose();
  trainer.set_learning_rate(initial_learning_rate);
  trainer.set_synchronization_file("states/a_fruits_train.state", std::chrono::minutes{10});
  trainer.set_iterations_without_progress_threshold(8000);
  set_all_bn_running_stats_window_sizes(net, 1000);

  dlib::pipe<std::pair<dlib::matrix<dlib::rgb_pixel>, dlib::matrix<float, 0, 1>>> data{200};
  auto load_data = [&data, &images](time_t seed) {
    dlib::rand rnd{std::time(nullptr) + seed};
    while(data.is_enabled()) {
      auto& inf = images[rnd.get_random_32bit_number() % images.size()];
      dlib::matrix<dlib::rgb_pixel> img;
      dlib::load_image(img, "ai_challenger_zsl2018_train_test_a_20180321/zsl_a_fruits_train_20180321/zsl_a_fruits_train_images_20180321/" + inf.first);
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
  std::vector<dlib::matrix<float, 0, 1>> labels;
  while(trainer.get_learning_rate() >= initial_learning_rate * 1e-3) {
    while(samples.size() < batch_size) {
      std::pair<dlib::matrix<dlib::rgb_pixel>, dlib::matrix<float, 0, 1>> img;
      data.dequeue(img);
      samples.emplace_back(std::move(img.first));
      labels.emplace_back(std::move(img.second));
    }
    trainer.train_one_step(samples, labels);
    samples.clear();
    labels.clear();
  }
  data.disable();
  for (auto& t: data_loaders) {
    t.join();
  }
  trainer.get_net();
  net.clean();

  std::cout << "Saving network..." << std::endl;
  dlib::serialize("models/a_fruits_train.resnet34") << net;

  std::cout << "Validating..." << std::endl;
  validator<58> v{net};
  v.load_dataset("ai_challenger_zsl2018_train_test_a_20180321/zsl_a_fruits_train_20180321/zsl_a_fruits_train_annotations_label_list_20180321.txt",
                 "ai_challenger_zsl2018_train_test_a_20180321/zsl_a_fruits_train_20180321/zsl_a_fruits_train_annotations_labels_20180321.txt",
                 "ai_challenger_zsl2018_train_test_a_20180321/zsl_a_fruits_train_20180321/zsl_a_fruits_train_annotations_attributes_per_class_20180321.txt");
  v.run("ai_challenger_zsl2018_train_test_a_20180321/zsl_a_fruits_train_20180321/zsl_a_fruits_train_images_20180321/");
  std::cout << "Right: " << v.right_num() << std::endl;
  std::cout << "Wrong: " << v.wrong_num() << std::endl;
  std::cout << "Accuracy: " << static_cast<double>(v.right_num()) / (v.right_num() + v.wrong_num()) << std::endl;
  return 0;
}
