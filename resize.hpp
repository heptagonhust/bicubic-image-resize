#ifndef RESIZE_H_
#define RESIZE_H_

#include "utils.hpp"
#include <cmath>
#include <thread>
#include "immintrin.h"
#include <vector>

#define channels 3

inline float WeightCoeff(float x,const float a) {
  if (x <= 1) {
    float x_2 = x*x;
    return 1 - (a + 3) * x_2 + (a + 2) * x * x_2;
  } else if (x < 2) {
    float x_2 = x*x;
    float a4 = 4 * a;
    return -1 * a4 + 2 * a4 * x - 5 * a * x_2 + a * x * x_2;
  }
  return 0.0;
}

static void CalcCoeff4x4(float x, float y, float *coeff) {
  const float a = -0.5f;

  float u = x - floor(x);
  float v = y - floor(y);

  u += 1;
  v += 1;

  float WCoeffU[16];
  WCoeffU[0] = WCoeffU[1] = WCoeffU[2] = WCoeffU[3] = WeightCoeff(fabs(u - 0), a);
  WCoeffU[4] = WCoeffU[5] = WCoeffU[6] = WCoeffU[7] = WeightCoeff(fabs(u - 1), a);
  WCoeffU[8] = WCoeffU[9] = WCoeffU[10] = WCoeffU[11] = WeightCoeff(fabs(u - 2), a);
  WCoeffU[12] = WCoeffU[13] = WCoeffU[14] = WCoeffU[15] = WeightCoeff(fabs(u - 3), a);
  float WCoeffV[16];
  WCoeffV[0] = WCoeffV[4] = WCoeffV[8] = WCoeffV[12] = WeightCoeff(fabs(v - 0), a);
  WCoeffV[1] = WCoeffV[5] = WCoeffV[9] = WCoeffV[13] = WeightCoeff(fabs(v - 1), a);
  WCoeffV[2] = WCoeffV[6] = WCoeffV[10] = WCoeffV[14] = WeightCoeff(fabs(v - 2), a);
  WCoeffV[3] = WCoeffV[7] = WCoeffV[11] = WCoeffV[15] = WeightCoeff(fabs(v - 3), a);

  /*//#pragma simd
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      coeff[i * 4 + j] = WCoeffU[i] * WCoeffV[j];
    }
  }*/
  
  //__m512 __WCoeffU = _mm512_set_ps(WCoeffU[0], WCoeffU[0], WCoeffU[0], WCoeffU[0], WCoeffU[1], WCoeffU[1], WCoeffU[1], WCoeffU[1], WCoeffU[2], WCoeffU[2], WCoeffU[2], WCoeffU[2], WCoeffU[3], WCoeffU[3], WCoeffU[3], WCoeffU[3]);
  //__m512 __WCoeffV = _mm512_set_ps(WCoeffV[0], WCoeffV[0], WCoeffV[0], WCoeffV[0], WCoeffV[1], WCoeffV[1], WCoeffV[1], WCoeffV[1], WCoeffV[2], WCoeffV[2], WCoeffV[2], WCoeffV[2], WCoeffV[3], WCoeffV[3], WCoeffV[3], WCoeffV[3]);

  __m512 __WCoeffU = _mm512_load_ps(WCoeffU);
  __m512 __WCoeffV = _mm512_load_ps(WCoeffV);

  _mm512_store_ps(coeff,_mm512_mul_ps(__WCoeffU,__WCoeffV));

}

static void BGRAfterBiCubic(RGBImage *src, float x_float, float y_float, unsigned char *sum, float *coeff) {

  int x0 = floor(x_float) - 1;
  int y0 = floor(y_float) - 1;

  float sumf[3] = {.0f};
  #pragma simd
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      float temp = coeff[i * 4 + j];
      int position = ((x0 + i) * src->cols + y0 + j) * channels;
      sumf[0] += temp * src->data[position + 0];
      sumf[1] += temp * src->data[position + 1];
      sumf[2] += temp * src->data[position + 2];
    }
  }

  /*__m128 row_0_coeff = _mm_load_ss(coeff);
  __m128 row_1_coeff = _mm_load_ss(coeff+4);
  __m128 row_2_coeff = _mm_load_ss(coeff+8);
  __m128 row_3_coeff = _mm_load_ss(coeff+12);
  int row0 = (x0 + 0) * src.cols + y0 ;
  __m128 row_0_data_0 = _mm_set_ps((float)src.data[(row0 + 3) * channels + 0],(float)src.data[(row0 + 2) * channels + 0],(float)src.data[(row0 + 1) * channels + 0],(float)src.data[(row0 + 0) * channels + 0]);
  __m128 row_0_data_1 = _mm_set_ps((float)src.data[(row0 + 3) * channels + 1],(float)src.data[(row0 + 2) * channels + 1],(float)src.data[(row0 + 1) * channels + 1],(float)src.data[(row0 + 0) * channels + 1]);
  __m128 row_0_data_2 = _mm_set_ps((float)src.data[(row0 + 3) * channels + 2],(float)src.data[(row0 + 2) * channels + 2],(float)src.data[(row0 + 1) * channels + 2],(float)src.data[(row0 + 0) * channels + 2]);
  int row1 = (x0 + 1) * src.cols + y0 ;
  __m128 row_1_data_0 = _mm_set_ps((float)src.data[(row1 + 3) * channels + 0],(float)src.data[(row1 + 2) * channels + 0],(float)src.data[(row1 + 1) * channels + 0],(float)src.data[(row1 + 0) * channels + 0]);
  __m128 row_1_data_1 = _mm_set_ps((float)src.data[(row1 + 3) * channels + 1],(float)src.data[(row1 + 2) * channels + 1],(float)src.data[(row1 + 1) * channels + 1],(float)src.data[(row1 + 0) * channels + 1]);
  __m128 row_1_data_2 = _mm_set_ps((float)src.data[(row1 + 3) * channels + 2],(float)src.data[(row1 + 2) * channels + 2],(float)src.data[(row1 + 1) * channels + 2],(float)src.data[(row1 + 0) * channels + 2]);
  int row2 = (x0 + 2) * src.cols + y0 ;
  __m128 row_2_data_0 = _mm_set_ps((float)src.data[(row2 + 3) * channels + 0],(float)src.data[(row2 + 2) * channels + 0],(float)src.data[(row2 + 1) * channels + 0],(float)src.data[(row2 + 0) * channels + 0]);
  __m128 row_2_data_1 = _mm_set_ps((float)src.data[(row2 + 3) * channels + 1],(float)src.data[(row2 + 2) * channels + 1],(float)src.data[(row2 + 1) * channels + 1],(float)src.data[(row2 + 0) * channels + 1]);
  __m128 row_2_data_2 = _mm_set_ps((float)src.data[(row2 + 3) * channels + 2],(float)src.data[(row2 + 2) * channels + 2],(float)src.data[(row2 + 1) * channels + 2],(float)src.data[(row2 + 0) * channels + 2]);
  int row3 = (x0 + 3) * src.cols + y0 ;
  __m128 row_3_data_0 = _mm_set_ps((float)src.data[(row3 + 3) * channels + 0],(float)src.data[(row3 + 2) * channels + 0],(float)src.data[(row2 + 1) * channels + 0],(float)src.data[(row3 + 0) * channels + 0]);
  __m128 row_3_data_1 = _mm_set_ps((float)src.data[(row3 + 3) * channels + 1],(float)src.data[(row3 + 2) * channels + 1],(float)src.data[(row2 + 1) * channels + 1],(float)src.data[(row3 + 0) * channels + 1]);
  __m128 row_3_data_2 = _mm_set_ps((float)src.data[(row3 + 3) * channels + 2],(float)src.data[(row3 + 2) * channels + 2],(float)src.data[(row2 + 1) * channels + 2],(float)src.data[(row3 + 0) * channels + 2]);  
  */
  
  /*__m512 __coeff = _mm512_load_ps(coeff);

  int row0 = (x0 + 0) * src->cols + y0 ;
  int row1 = (x0 + 1) * src->cols + y0 ;
  int row2 = (x0 + 2) * src->cols + y0 ;
  int row3 = (x0 + 3) * src->cols + y0 ;

  __m512 __data_0 = _mm512_set_ps(src->data[(row3 + 3) * channels + 0],src->data[(row3 + 2) * channels + 0],src->data[(row2 + 1) * channels + 0],src->data[(row3 + 0) * channels + 0],(float)src->data[(row2 + 3) * channels + 0],src->data[(row2 + 2) * channels + 0],src->data[(row2 + 1) * channels + 0],src->data[(row2 + 0) * channels + 0],src->data[(row1 + 3) * channels + 0],src->data[(row1 + 2) * channels + 0],src->data[(row1 + 1) * channels + 0],src->data[(row1 + 0) * channels + 0],src->data[(row0 + 3) * channels + 0],src->data[(row0 + 2) * channels + 0],src->data[(row0 + 1) * channels + 0],src->data[(row0 + 0) * channels + 0]);
  __m512 __data_1 = _mm512_set_ps(src->data[(row3 + 3) * channels + 1],src->data[(row3 + 2) * channels + 1],src->data[(row2 + 1) * channels + 1],src->data[(row3 + 0) * channels + 1],(float)src->data[(row2 + 3) * channels + 1],src->data[(row2 + 2) * channels + 1],src->data[(row2 + 1) * channels + 1],src->data[(row2 + 0) * channels + 1],src->data[(row1 + 3) * channels + 1],src->data[(row1 + 2) * channels + 1],src->data[(row1 + 1) * channels + 1],src->data[(row1 + 0) * channels + 1],src->data[(row0 + 3) * channels + 1],src->data[(row0 + 2) * channels + 1],src->data[(row0 + 1) * channels + 1],src->data[(row0 + 0) * channels + 1]);
  __m512 __data_2 = _mm512_set_ps(src->data[(row3 + 3) * channels + 2],src->data[(row3 + 2) * channels + 2],src->data[(row2 + 1) * channels + 2],src->data[(row3 + 0) * channels + 2],(float)src->data[(row2 + 3) * channels + 2],src->data[(row2 + 2) * channels + 2],src->data[(row2 + 1) * channels + 2],src->data[(row2 + 0) * channels + 2],src->data[(row1 + 3) * channels + 2],src->data[(row1 + 2) * channels + 2],src->data[(row1 + 1) * channels + 2],src->data[(row1 + 0) * channels + 2],src->data[(row0 + 3) * channels + 2],src->data[(row0 + 2) * channels + 2],src->data[(row0 + 1) * channels + 2],src->data[(row0 + 0) * channels + 2]);*/
  
  /*//_mm_mul_ss
  __m128 row_0_sum_0 = _mm_mul_ss(row_0_coeff, row_0_data_0);
  __m128 row_0_sum_1 = _mm_mul_ss(row_0_coeff, row_0_data_1);
  __m128 row_0_sum_2 = _mm_mul_ss(row_0_coeff, row_0_data_2);
  __m128 row_1_sum_0 = _mm_mul_ss(row_1_coeff, row_1_data_0);
  __m128 row_1_sum_1 = _mm_mul_ss(row_1_coeff, row_1_data_1);
  __m128 row_1_sum_2 = _mm_mul_ss(row_1_coeff, row_1_data_2);
  __m128 row_2_sum_0 = _mm_mul_ss(row_2_coeff, row_2_data_0);
  __m128 row_2_sum_1 = _mm_mul_ss(row_2_coeff, row_2_data_1);
  __m128 row_2_sum_2 = _mm_mul_ss(row_2_coeff, row_2_data_2);
  __m128 row_3_sum_0 = _mm_mul_ss(row_3_coeff, row_3_data_0);
  __m128 row_3_sum_1 = _mm_mul_ss(row_3_coeff, row_3_data_1);
  __m128 row_3_sum_2 = _mm_mul_ss(row_3_coeff, row_3_data_2);*/
  //_mm512_mul_ps

  /*__m512 __sum_0 = _mm512_mul_ps(__coeff,__data_0);
  __m512 __sum_1 = _mm512_mul_ps(__coeff,__data_1);
  __m512 __sum_2 = _mm512_mul_ps(__coeff,__data_2);*/

  /*sum[0] = static_cast<unsigned char>(row_0_sum_0[0]+row_0_sum_0[1]+row_0_sum_0[2]+row_0_sum_0[3]+row_1_sum_0[0]+row_1_sum_0[1]+row_1_sum_0[2]+row_1_sum_0[3]+row_2_sum_0[0]+row_2_sum_0[1]+row_2_sum_0[2]+row_2_sum_0[3]+row_3_sum_0[0]+row_3_sum_0[1]+row_3_sum_0[2]+row_3_sum_0[3]);
  sum[1] = static_cast<unsigned char>(row_0_sum_1[0]+row_0_sum_1[1]+row_0_sum_1[2]+row_0_sum_1[3]+row_1_sum_1[0]+row_1_sum_1[1]+row_1_sum_1[2]+row_1_sum_1[3]+row_2_sum_1[0]+row_2_sum_1[1]+row_2_sum_1[2]+row_2_sum_1[3]+row_3_sum_1[0]+row_3_sum_1[1]+row_3_sum_1[2]+row_3_sum_1[3]);
  sum[2] = static_cast<unsigned char>(row_0_sum_2[0]+row_0_sum_2[1]+row_0_sum_2[2]+row_0_sum_2[3]+row_1_sum_2[0]+row_1_sum_2[1]+row_1_sum_2[2]+row_1_sum_2[3]+row_2_sum_2[0]+row_2_sum_2[1]+row_2_sum_2[2]+row_2_sum_2[3]+row_3_sum_2[0]+row_3_sum_2[1]+row_3_sum_2[2]+row_3_sum_2[3]);*/
  sum[0] = static_cast<unsigned char>(sumf[0]);
  sum[1] = static_cast<unsigned char>(sumf[1]);
  sum[2] = static_cast<unsigned char>(sumf[2]);

  //sum[0] = static_cast<unsigned char>(__sum_0[0] + __sum_0[1] +__sum_0[2] +__sum_0[3] +__sum_0[4] +__sum_0[5] +__sum_0[6] +__sum_0[7] +__sum_0[8] +__sum_0[9] +__sum_0[10] +__sum_0[11] +__sum_0[12] +__sum_0[13] +__sum_0[14] +__sum_0[15]);
  //sum[1] = static_cast<unsigned char>(__sum_1[0] + __sum_1[1] +__sum_1[2] +__sum_1[3] +__sum_1[4] +__sum_1[5] +__sum_1[6] +__sum_1[7] +__sum_1[8] +__sum_1[9] +__sum_1[10] +__sum_1[11] +__sum_1[12] +__sum_1[13] +__sum_1[14] +__sum_1[15]);
  //sum[2] = static_cast<unsigned char>(__sum_2[0] + __sum_2[1] +__sum_2[2] +__sum_2[3] +__sum_2[4] +__sum_2[5] +__sum_2[6] +__sum_2[7] +__sum_2[8] +__sum_2[9] +__sum_2[10] +__sum_2[11] +__sum_2[12] +__sum_2[13] +__sum_2[14] +__sum_2[15]);

  /*sum[0] = static_cast<unsigned char>(_mm512_reduce_add_ps(__sum_0));
  sum[1] = static_cast<unsigned char>(_mm512_reduce_add_ps(__sum_1));
  sum[2] = static_cast<unsigned char>(_mm512_reduce_add_ps(__sum_2));*/

  return ;
}


//void ResizeImageThread(RGBImage src, int i, int j, int src_x, int src_y, int channels, int thischannel, unsigned char *res, float *coeff){
//  int resize_cols = src.cols * 5.f;
// res[((i * resize_cols) + j) * channels + thischannel] = BGRAfterBiCubic(src, src_x, src_y, channels, thischannel, coeff);
//  return;
//}

void ResizeImagePart(RGBImage *src, float ratio, int x_left, int x_right, int y_up, int y_down, unsigned char *res) {
//  static const int resize_rows = src.rows * ratio;
//  static const int resize_cols = src.cols * ratio;
  const int resize_cols = src->cols * ratio;
  //auto check_perimeter = [src](float x, float y) -> bool {
  //  return x < src.rows - 2 && x > 1 && y < src.cols - 2 && y > 1;
  //};
//  Timer part("Worker");
  float coeff[16];
  unsigned char sum[3];
  if(x_left == 0) x_left += 2;
  if(y_up == 0) y_up += 2;
  if(x_right/ratio == src->rows) x_right -= 2;
  if(y_down/ratio == src->cols) y_down -= 2;
  for (int i = x_left; i < x_right; i++) {
    for (int j = y_up; j < y_down; j++) {
      float src_x = i / ratio;
      float src_y = j / ratio;
      //if (check_perimeter(src_x, src_y)) {
        CalcCoeff4x4(src_x, src_y, coeff);
        /*std::thread worker0(ResizeImageThread, src, i, j, src_x, src_y, channels, 0, res, coeff);
        std::thread worker1(ResizeImageThread, src, i, j, src_x, src_y, channels, 1, res, coeff);
        std::thread worker2(ResizeImageThread, src, i, j, src_x, src_y, channels, 2, res, coeff);
        worker0.join();
        worker1.join();
        worker2.join();*/
        BGRAfterBiCubic(src, src_x, src_y, sum, coeff);
        res[((i * resize_cols) + j) * channels + 0] = sum[0];
        res[((i * resize_cols) + j) * channels + 1] = sum[1];
        res[((i * resize_cols) + j) * channels + 2] = sum[2];
      //}
    }
  }
  return ;
}

RGBImage ResizeImage(RGBImage src, float ratio) {
  Timer timer("resize image by 5x");
  const int resize_rows = src.rows * ratio ;
  const int resize_cols = src.cols * ratio ;

  printf("resize to: %d x %d\n", resize_rows, resize_cols);

  auto res = new unsigned char[channels * resize_rows * resize_cols];
  std::fill(res, res + channels * resize_rows * resize_cols, 0);

  const int n=8;
  std::thread PartThread[n][n];

  for(int x = 0 ; x < n ; x++)
    for(int y = 0 ; y < n ; y++){
      PartThread[x][y] = std::thread(ResizeImagePart, &src, ratio, (int)((resize_rows/n)*x), (int)((resize_rows/n)*(x+1)), (int)((resize_cols/n)*y), (int)((resize_cols/n)*(y+1)), res);
    }
  for(int x = 0 ; x < n ; x++)
    for(int y = 0 ; y < n ; y++){
      PartThread[x][y].join();
    }

  return RGBImage{resize_cols, resize_rows, channels, res};
}

#endif
