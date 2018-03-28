#ifndef AI_CHALLENGER_ZSL_BASE_NETWORK_HPP
#define AI_CHALLENGER_ZSL_BASE_NETWORK_HPP

#include <dlib/dnn.h>

template <size_t Size>
class base_network {
private:
  template <
    template <
      int,
      template <class> class,
      int,
      class
    > class Block,
    int N,
    template <class> class BatchNormal,
    class Subnet
  > using residual = dlib::add_prev1<Block<N, BatchNormal, 1, dlib::tag1<Subnet>>>;

  template <
    template <
      int,
      template <class> class,
      int,
      class
    > class Block,
    int N,
    template <class> class BatchNormal,
    typename Subnet
  > using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<Block<N,BatchNormal,2,dlib::tag1<Subnet>>>>>>;

  template <
    int N,
    template <class> class BatchNormal,
    int Stride,
    typename Subnet
  > using block = BatchNormal<dlib::con<N,3,3,1,1,dlib::relu<BatchNormal<dlib::con<N,3,3,Stride,Stride,Subnet>>>>>;

  template <int N, class Subnet> using res       = dlib::relu<residual<block,N,dlib::bn_con,Subnet>>;
  template <int N, class Subnet> using ares      = dlib::relu<residual<block,N,dlib::affine,Subnet>>;
  template <int N, class Subnet> using res_down  = dlib::relu<residual_down<block,N,dlib::bn_con,Subnet>>;
  template <int N, class Subnet> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,Subnet>>;

  template <class Subnet> using level1 = res<512,res<512,res_down<512,Subnet>>>;
  template <class Subnet> using level2 = res<256,res<256,res<256,res<256,res<256,res_down<256,Subnet>>>>>>;
  template <class Subnet> using level3 = res<128,res<128,res<128,res_down<128,Subnet>>>>;
  template <class Subnet> using level4 = res<64,res<64,res<64,Subnet>>>;

  template <class Subnet> using alevel1 = ares<512,ares<512,ares_down<512,Subnet>>>;
  template <class Subnet> using alevel2 = ares<256,ares<256,ares<256,ares<256,ares<256,ares_down<256,Subnet>>>>>>;
  template <class Subnet> using alevel3 = ares<128,ares<128,ares<128,ares_down<128,Subnet>>>>;
  template <class Subnet> using alevel4 = ares<64,ares<64,ares<64,Subnet>>>;

public:
  using training_type = dlib::fc<Size,dlib::avg_pool_everything<level1<level2<level3<level4<dlib::max_pool<3,3,2,2,dlib::relu<dlib::bn_con<dlib::con<64,7,7,2,2,dlib::input_rgb_image_sized<227>>>>>>>>>>>;

  using testing_type = dlib::fc<Size,dlib::avg_pool_everything<alevel1<alevel2<alevel3<alevel4<dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<64,7,7,2,2,dlib::input_rgb_image_sized<227>>>>>>>>>>>;
};

#endif  // AI_CHALLENGER_ZSL_BASE_NETWORK_HPP
