#ifndef RESIZE_H_
#define RESIZE_H_

#include "utils.hpp"
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

float WeightCoeff(float x, float a) {
  if (x <= 1) {
    return 1 - (a + 3) * x * x + (a + 2) * x * x * x;
  } else if (x < 2) {
    return -4 * a + 8 * a * x - 5 * a * x * x + a * x * x * x;
  }
  return 0.0;
}

void CalcCoeff4x4(float x, float y, float *coeff) {
  const float a = -0.5f;

  float u = x - floor(x);
  float v = y - floor(y);

  u += 1;
  v += 1;

  float A[4];
  A[0] = WeightCoeff(abs(u), a);
  A[1] = WeightCoeff(abs(u - 1), a);
  A[2] = WeightCoeff(abs(u - 2), a);
  A[3] = WeightCoeff(abs(u - 3), a);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      coeff[i * 4 + j] =
          WeightCoeff(fabs(u - i), a) * WeightCoeff(fabs(v - j), a);
    }
  }
}

Vec3b BGRAfterBiCubic(Mat src, float x_float, float y_float) {
  float coeff[16];

  float sum[] = {.0f, .0f, .0f};
  int x0 = floor(x_float) - 1;
  int y0 = floor(y_float) - 1;
  CalcCoeff4x4(x_float, y_float, coeff);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      auto old_bgr = src.ptr<Vec3b>(x0 + i, y0 + j);
      for (int d = 0; d < 3; d++) {
        sum[d] += coeff[i * 4 + j] * (*old_bgr)[d];
      }
    }
  }
  return Vec3b(sum[0], sum[1], sum[2]);
}

Mat ResizeImage(Mat src, float ratio) {
  Timer timer("resize image by 5x");
  int resize_rows = src.rows * ratio;
  int resize_cols = src.cols * ratio;

  auto check_perimeter = [src](float x, float y) -> bool {
    return x < src.rows - 2 && x > 1 && y < src.cols - 2 && y > 1;
  };

  Mat res(resize_rows, resize_cols, CV_8UC3);
  for (int i = 0; i < resize_rows; i++) {
    for (int j = 0; j < resize_cols; j++) {
      float src_x = i / ratio;
      float src_y = j / ratio;
      if (check_perimeter(src_x, src_y)) {
        res.at<Vec3b>(i, j) = BGRAfterBiCubic(src, src_x, src_y);
      }
    }
  }
  return res;
}

#endif