#ifndef AI_CHALLENGER_ZSL_RANDOMLY_CROP_HPP
#define AI_CHALLENGER_ZSL_RANDOMLY_CROP_HPP

#include <dlib/image_transforms.h>

inline dlib::rectangle make_random_cropping_rect(
  const dlib::matrix<dlib::rgb_pixel>& img,
  dlib::rand& rnd
) {
  // figure out what rectangle we want to crop from the image
  auto mins = 0.466666666;
  auto maxs = 0.875;
  auto scale = rnd.get_double_in_range(mins, maxs);
  auto size = scale * std::min(img.nr(), img.nc());
  dlib::rectangle rect(size, size);
  // randomly shift the box around
  dlib::point offset(rnd.get_random_32bit_number() % (img.nc() - rect.width()),
                     rnd.get_random_32bit_number() % (img.nr() - rect.height()));
  return dlib::move_rect(rect, offset);
}

inline void randomly_crop_image(
  const dlib::matrix<dlib::rgb_pixel>& img,
  dlib::matrix<dlib::rgb_pixel>& crop,
  dlib::rand& rnd
) {
  auto rect = make_random_cropping_rect(img, rnd);
  // now crop it out as a 227x227 image.
  dlib::extract_image_chip(img, dlib::chip_details(rect, dlib::chip_dims(227,227)), crop);
  // Also randomly flip the image
  if (rnd.get_random_double() > 0.5) {
    crop = fliplr(crop);
  }
  // And then randomly adjust the colors.
  dlib::apply_random_color_offset(crop, rnd);
}

inline void randomly_crop_images(
  const dlib::matrix<dlib::rgb_pixel>& img,
  dlib::array<dlib::matrix<dlib::rgb_pixel>>& crops,
  dlib::rand& rnd,
  long num_crops
) {
  std::vector<dlib::chip_details> dets;
  for (long i = 0; i < num_crops; ++i) {
    auto rect = make_random_cropping_rect(img, rnd);
    dets.push_back(dlib::chip_details(rect, dlib::chip_dims(227,227)));
  }
  dlib::extract_image_chips(img, dets, crops);
  for (auto&& img : crops) {
    // Also randomly flip the image
    if (rnd.get_random_double() > 0.5) {
      img = dlib::fliplr(img);
    }
    // And then randomly adjust the colors.
    apply_random_color_offset(img, rnd);
  }
}

#endif  // AI_CHALLENGER_ZSL_RANDOMLY_CROP_HPP
