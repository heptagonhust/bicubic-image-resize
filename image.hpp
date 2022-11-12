
#ifndef IMAGE_H_
#define IMAGE_H_

#define STB_IMAGE_IMPLEMENTATION

#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include "utils.hpp"
#include <string>

RGBImage LoadImage(const std::string &filename) {
  int cols, rows, channels;
  auto data = stbi_load(filename.c_str(), &cols, &rows, &channels, 3);
  if (channels != 3) {
    std::cerr << filename << " has " << channels << " channels " << std::endl;
    std::cerr << "only supports rgb image" << std::endl;
    exit(0);
  }
  // for (int i = 0; i < rows; i++) {
  //   for (int j = 0; j < cols; j++) {
  //     printf("%u %u %u\n", data[((i * cols) + j) * channels],
  //            data[((i * cols) + j) * channels + 1],
  //            data[((i * cols) + j) * channels + 2]);
  //   }
  // }
  printf("image height: %d, width: %d\n", rows, cols);
  return RGBImage{cols, rows, data};
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