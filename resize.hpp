#ifndef RESIZE_H_
#define RESIZE_H_

#include "utils.hpp"
#include <cmath>
#include <utility>
#include <thread>
#include <chrono>
#include "utils.hpp"
#include <cmath>
#define N 20
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

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      coeff[i * 4 + j] =
          WeightCoeff(fabs(u - i), a) * WeightCoeff(fabs(v - j), a);
    }
  }
}

unsigned char BGRAfterBiCubic(RGBImage src, float x_float, float y_float,
                              int channels, int d) {
  float coeff[16];

  int x0 = floor(x_float) - 1;
  int y0 = floor(y_float) - 1;
  CalcCoeff4x4(x_float, y_float, coeff);

  float sum = .0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      sum += coeff[i * 4 + j] *
             src.data[((x0 + i) * src.cols + y0 + j) * channels + d];
    }
  }
  return static_cast<unsigned char>(sum);
}
int SubResize(RGBImage src,unsigned char *res,int channels,int resize_rows,int resize_cols,float ratio,int block_x,int block_y,int channel){
  auto check_perimeter = [src](float x, float y) -> bool {
    return x < src.rows - 2 && x > 1 && y < src.cols - 2 && y > 1;
  };
  int step_x = ratio * src.cols / N + 1;
  int step_y = ratio * src.rows / N + 1;
  for (register int i = step_x * block_x; i < resize_rows && i < step_x * (block_x + 1); i++) {
    for (register int j = step_y * block_y; j < resize_cols && j < step_y * (block_y + 1); j++) {
      float src_x = i / ratio;
      float src_y = j / ratio;
      if (check_perimeter(src_x, src_y)) {
        //for (int d = 0; d < channels; d++) {
          res[((i * resize_cols) + j) * channels + channel] =
              BGRAfterBiCubic(src, src_x, src_y, channels, channel);
        //}
      }
    }
  }
  return 0;
}
RGBImage ResizeImage(RGBImage src, float ratio){
  const int channels = src.channels;
  Timer timer("resize image by 5x");
  int resize_rows = src.rows * ratio;
  int resize_cols = src.cols * ratio;
  auto res = new unsigned char[channels * resize_rows * resize_cols];
  std::fill(res, res + channels * resize_rows * resize_cols, 0);
  printf("resize to: %d x %d\n", resize_rows, resize_cols);
  int i = 0;
  std::thread MyThread[N * N * channels];
  for(int block_x=0;block_x < N;block_x++){
    for(int block_y=0;block_y < N;block_y++){
      for(int channel = 0;channel<channels;channel++)
        MyThread[i++]=std::thread(SubResize,src,res,channels,resize_rows,resize_cols,ratio,block_x,block_y,channel);
    }
  }
  for(int i = 0;i < N * N * channels;i++){
    MyThread[i].join();
  }
  return RGBImage{resize_cols, resize_rows, channels, res};
}

#endif