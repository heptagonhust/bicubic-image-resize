
#ifndef IMAGE_H_
#define IMAGE_H_

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

Mat LoadImage(const std::string &filename) {
  auto mat = imread(filename, IMREAD_COLOR);
  return mat;
}

void StoreImage(const cv::Mat &mat, const std::string &filename) {
  std::cerr << "save image " << filename << std::endl;
  auto succ = imwrite(filename, mat);
  if(!succ) {
    std::cerr << "error saving image " << std::endl;
  }
}

#endif