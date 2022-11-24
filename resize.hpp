#ifndef RESIZE_H_
#define RESIZE_H_

#include "utils.hpp"
#include <cmath>
#include <utility>
#include <thread>
#include <chrono>
#include "utils.hpp"
#include <cmath>
#define N 4
float WeightCoeff(float x, float a) {
  float temp = x * x;//not int temp!!
  if (x <= 1) {
    return 1 - (a + 3) * temp + (a + 2) * x * temp;
  } else if (x < 2) {
    float a4 = (a + a) + (a + a);
    return - a4 + 2 * a4 * x - (a4 + a) * temp + a * x * temp;
  }
  return 0.0;
}

void CalcCoeff4x4(float x, float y, float *coeff) {
  const float a = -0.5f;

  float u = x - floor(x);   
  float v = y - floor(y);

  u += 1;
  v += 1;

  float WeightCoeff_u[4];
  float WeightCoeff_v[4];
  //for (int k = 0;;k < 4;k++){
    WeightCoeff_v[0] = WeightCoeff(fabs(v), a);
    WeightCoeff_v[1] = WeightCoeff(fabs(v - 1), a);
    WeightCoeff_v[2] = WeightCoeff(fabs(v - 2), a);
    WeightCoeff_v[3] = WeightCoeff(fabs(v - 3), a);

    WeightCoeff_u[0] = WeightCoeff(fabs(u), a);
    WeightCoeff_u[1] = WeightCoeff(fabs(u - 1), a);
    WeightCoeff_u[2] = WeightCoeff(fabs(u - 2), a);
    WeightCoeff_u[3] = WeightCoeff(fabs(u - 3), a);

  //}
  //for (int i = 0; i < 4; i++) {
    //for (int j = 0; j < 4; j++) {//循环展开
      coeff[0] = WeightCoeff_u[0] * WeightCoeff_v[0];
      coeff[1] = WeightCoeff_u[0] * WeightCoeff_v[1];      
      coeff[2] = WeightCoeff_u[0] * WeightCoeff_v[2];
      coeff[3] = WeightCoeff_u[0] * WeightCoeff_v[3];// make use of cache

      coeff[4] = WeightCoeff_u[1] * WeightCoeff_v[0];
      coeff[5] = WeightCoeff_u[1] * WeightCoeff_v[1];      
      coeff[6] = WeightCoeff_u[1] * WeightCoeff_v[2];
      coeff[7] = WeightCoeff_u[1] * WeightCoeff_v[3];

      coeff[8] = WeightCoeff_u[2] * WeightCoeff_v[0];
      coeff[9] = WeightCoeff_u[2] * WeightCoeff_v[1];      
      coeff[10] = WeightCoeff_u[2] * WeightCoeff_v[2];
      coeff[11] = WeightCoeff_u[2] * WeightCoeff_v[3];

      coeff[12] = WeightCoeff_u[3] * WeightCoeff_v[0];
      coeff[13] = WeightCoeff_u[3] * WeightCoeff_v[1];      
      coeff[14] = WeightCoeff_u[3] * WeightCoeff_v[2];
      coeff[15] = WeightCoeff_u[3] * WeightCoeff_v[3];

 
    //}
  //}
}

unsigned char BGRAfterBiCubic(RGBImage src, float x_float, float y_float,int channels, int d) {
  float coeff[16];

  int x0 = floor(x_float) - 1;
  int y0 = floor(y_float) - 1;
  CalcCoeff4x4(x_float, y_float, coeff);

  float sum = .0f;
//  for (int i = 0; i < 4; i++) {
//  for (int j = 0; j < 4; j++) {
//     sum += coeff[i * 4 + j] * src.data[((x0 + i) * src.cols + y0 + j) * channels + d];
//   }
// }
  sum += coeff[0] * src.data[((x0) * src.cols + y0 + 0) * channels + d];
  sum += coeff[1] * src.data[((x0) * src.cols + y0 + 1) * channels + d];
  sum += coeff[2] * src.data[((x0) * src.cols + y0 + 2) * channels + d];
  sum += coeff[3] * src.data[((x0) * src.cols + y0 + 3) * channels + d];

  sum += coeff[4] * src.data[((x0+1) * src.cols + y0 + 0) * channels + d];
  sum += coeff[5] * src.data[((x0+1) * src.cols + y0 + 1) * channels + d];
  sum += coeff[6] * src.data[((x0+1) * src.cols + y0 + 2) * channels + d];
  sum += coeff[7] * src.data[((x0+1) * src.cols + y0 + 3) * channels + d];

  sum += coeff[8] * src.data[((x0+2) * src.cols + y0 + 0) * channels + d];
  sum += coeff[9] * src.data[((x0+2) * src.cols + y0 + 1) * channels + d];
  sum += coeff[10] * src.data[((x0+2) * src.cols + y0 + 2) * channels + d];
  sum += coeff[11] * src.data[((x0+2) * src.cols + y0 + 3) * channels + d];

  sum += coeff[12] * src.data[((x0+3) * src.cols + y0 + 0) * channels + d];
  sum += coeff[13] * src.data[((x0+3) * src.cols + y0 + 1) * channels + d];
  sum += coeff[14] * src.data[((x0+3) * src.cols + y0 + 2) * channels + d];
  sum += coeff[15] * src.data[((x0+3) * src.cols + y0 + 3) * channels + d];



  return static_cast<unsigned char>(sum);
}
int SubResize(RGBImage src,unsigned char *res,int channels,int resize_rows,int resize_cols,float ratio,int block_x,int block_y){
  auto check_perimeter = [src](float x, float y) -> bool {
    return x < src.rows - 2 && x > 1 && y < src.cols - 2 && y > 1;
  };
  int step_x = ratio * src.cols / N + 1;//步长的计算，+1 消去截断误差的影响（最后一个块会小一点）
  int step_y = ratio * src.rows / N + 1;
  for (int i = step_x * block_x; i < resize_rows && i < step_x * (block_x + 1); i++) {
    for (int j = step_y * block_y; j < resize_cols && j < step_y * (block_y + 1); j++) {
      float src_x = i / ratio;
      float src_y = j / ratio;
      if (check_perimeter(src_x, src_y)) {
        //for (int d = 0; d < channels; d++) {
          res[((i * resize_cols) + j) * channels] = BGRAfterBiCubic(src, src_x, src_y, channels, 0);
          res[((i * resize_cols) + j) * channels+1] = BGRAfterBiCubic(src, src_x, src_y, channels, 1);
          res[((i * resize_cols) + j) * channels+2] = BGRAfterBiCubic(src, src_x, src_y, channels, 2);
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
  std::thread MyThread[N * N];
  for(int block_x=0;block_x < N;block_x++){
    for(int block_y=0;block_y < N;block_y++){
     // for(int channel = 0;channel<channels;channel++)//best 4*4 not 4*4*3
        MyThread[i++]=std::thread(SubResize,src,res,channels,resize_rows,resize_cols,ratio,block_x,block_y);//分配计算任务给多个线程
      }
  }
  for(int i = 0;i < N * N;i++){
    MyThread[i].join();
  }
  return RGBImage{resize_cols, resize_rows, channels, res};
}

#endif