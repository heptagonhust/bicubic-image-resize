#ifndef RESIZE_H_
#define RESIZE_H_

#include <utility>
#include <thread>
#include <chrono>
#include "utils.hpp"
#include <cmath>
#include <immintrin.h> //AVX(include wmmintrin.h)
//#include <intrin.h>    //(include immintrin.h)

#define N 8
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

  float WeightCoeff_u[16];
  float WeightCoeff_v[16];

    WeightCoeff_v[0] = WeightCoeff_v[1] = WeightCoeff_v[2] =  WeightCoeff_v[3] = WeightCoeff(fabs(v), a);
    WeightCoeff_v[4] = WeightCoeff_v[5] = WeightCoeff_v[6] =  WeightCoeff_v[7] = WeightCoeff(fabs(v-1), a);
    WeightCoeff_v[8] = WeightCoeff_v[9] = WeightCoeff_v[10] = WeightCoeff_v[11] = WeightCoeff(fabs(v-2), a);
    WeightCoeff_v[12] = WeightCoeff_v[13] = WeightCoeff_v[14] = WeightCoeff_v[15] = WeightCoeff(fabs(v-3), a);


    WeightCoeff_u[0] = WeightCoeff_u[4] = WeightCoeff_u[8] = WeightCoeff_u[12] = WeightCoeff(fabs(u), a);
    WeightCoeff_u[1] = WeightCoeff_u[5] = WeightCoeff_u[9] = WeightCoeff_u[13] = WeightCoeff(fabs(u - 1), a);
    WeightCoeff_u[2] = WeightCoeff_u[6] = WeightCoeff_u[10] = WeightCoeff_u[14] = WeightCoeff(fabs(u - 2), a);
    WeightCoeff_u[3] = WeightCoeff_u[7] = WeightCoeff_u[11] = WeightCoeff_u[15] = WeightCoeff(fabs(u - 3), a);

  __m256 s = _mm256_loadu_ps (WeightCoeff_u);
  __m256 t = _mm256_loadu_ps (WeightCoeff_v);
  __m256 rst1 = _mm256_mul_ps (s,t);
  _mm256_storeu_ps(coeff,rst1);

  __m256 c = _mm256_loadu_ps (WeightCoeff_u+8);
  __m256 d = _mm256_loadu_ps (WeightCoeff_v+8);
  __m256 rst2 = _mm256_mul_ps (c,d);
  _mm256_storeu_ps(coeff+8,rst2);


}

unsigned char BGRAfterBiCubic(RGBImage src, float x_float, float y_float,int channels, int d,float coeff[16]) {
  

  int x0 = floor(x_float) - 1;
  int y0 = floor(y_float) - 1;
  

  float sum = .0f;
  //#pragma simd 负优化。
  for (int i = 0; i < 4; i++) {
  for (int j = 0; j < 4; j++) {
     sum += coeff[i * 4 + j] * src.data[((x0 + i) * src.cols + y0 + j) * channels + d];
   }
 }
/*
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
*/


  return static_cast<unsigned char>(sum);
}
int SubResize(RGBImage src,unsigned char *res,int channels,int resize_rows,int resize_cols,float ratio,int block_x,int block_y){
  Timer part("part");
  auto check_perimeter = [src](float x, float y) -> bool {
    return x < src.rows - 2 && x > 1 && y < src.cols - 2 && y > 1;
  };
  //int step_x = ratio * src.cols / N + 1;//步长的计算，+1 消去截断误差的影响（最后一个块会小一点）
  //int step_y = ratio * src.rows / N + 1;
  int step_x = resize_cols / N + 1;//步长的计算，+1 消去截断误差的影响（最后一个块会小一点）
  int step_y = resize_rows / N + 1;
/*
<---x(i)-->
*********|
*********y,j
*********|
3 rows,9colunms,
*/

  float coeff[16];
  for (int i = step_y * block_y; i < resize_rows && i < step_y * (block_y + 1); i++) {
    for (int j = step_x * block_x; j < resize_cols +1 && j < step_x * (block_x + 1); j++) {
      float src_x = i / ratio;
      float src_y = j / ratio;
      if (check_perimeter(src_x, src_y)) {
        //for (int d = 0; d < channels; d++) {
          CalcCoeff4x4(src_x, src_y, coeff);
          res[((i * resize_cols) + j) * channels] = BGRAfterBiCubic(src, src_x, src_y, channels, 0,coeff);//(before)>>> resize image by 5x: 67ms save image ../images/CS_5x.jpg(after)
          res[((i * resize_cols) + j) * channels+1] = BGRAfterBiCubic(src, src_x, src_y, channels, 1,coeff);
          res[((i * resize_cols) + j) * channels+2] = BGRAfterBiCubic(src, src_x, src_y, channels, 2,coeff);
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