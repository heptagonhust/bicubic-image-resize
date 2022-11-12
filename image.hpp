
#ifndef IMAGE_H_
#define IMAGE_H_

#define STB_IMAGE_IMPLEMENTATION

#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include "utils.hpp"
#include <string>

RGBImage LoadImage(const std::string &filename) {
  int cols, rows, img_channels;
  int expected_channels = 3;
  // expected 3 channels loaded from image
  auto data = stbi_load(filename.c_str(), &cols, &rows, &img_channels,
                        expected_channels);
  printf("image height: %d, width: %d\n", rows, cols);
  return RGBImage{cols, rows, expected_channels, data};
}

void StoreImage(RGBImage img, const std::string &filename) {
  std::cerr << "save image " << filename << std::endl;
  auto succ = stbi_write_jpg(filename.c_str(), img.cols, img.rows, img.channels,
                             img.data, 95);
  if (!succ) {
    std::cerr << "error saving image " << std::endl;
  }
}

#endif