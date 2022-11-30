#ifndef RESIZE_H_
#define RESIZE_H_

#include "utils.hpp"
#include <cmath>
#include "omp.h"


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

unsigned char cubic_inner_SSE(RGBImage src, float x_float, float y_float, int channels, int d)
{
    const float a = -0.5f;
    //计算权重系数
    float u = x_float - floor(x_float);
    float v = y_float - floor(y_float);
    float a_mul_4 = (a + a) + (a + a);
    float a_mul_5 = a_mul_4 + a;
    float a_mul_8 = a_mul_4 + a_mul_4;
    float a_add_3 = a + 3;
    float a_add_2 = a + 2;
    __m128 a_m = _mm_set1_ps(a);
    __m128 m_1 = _mm_set1_ps(1.0);
    __m128 a_mul_4_m = _mm_set1_ps(a_mul_4);
    __m128 a_mul_5_m = _mm_set1_ps(a_mul_5);
    __m128 a_mul_8_m = _mm_set1_ps(a_mul_8);
    __m128 a_add_3_m = _mm_set1_ps(a_add_3);
    __m128 a_add_2_m = _mm_set1_ps(a_add_2);


    __m128 C30_A30 = _mm_set_ps(2 - v, 1 + v, 2 - u, 1 + u);
    __m128 C21_A21 = _mm_set_ps(1 - v, v, 1 - u, u);


    __m128 tmp0 = _mm_sub_ps(_mm_mul_ps(a_m, C30_A30), a_mul_5_m);
    tmp0 = _mm_add_ps(a_mul_8_m, _mm_mul_ps(C30_A30, tmp0));
    tmp0 = _mm_sub_ps(_mm_mul_ps(C30_A30, tmp0), a_mul_4_m);


    __m128 tmp1 = _mm_sub_ps(_mm_mul_ps(a_add_2_m, C21_A21), a_add_3_m);
    tmp1 = _mm_mul_ps(_mm_mul_ps(C21_A21, C21_A21), tmp1);
    tmp1 = _mm_add_ps(m_1, tmp1);


    __m128 A_m = _mm_unpacklo_ps(tmp0, tmp1);
    __m128 C_m = _mm_unpackhi_ps(tmp0, tmp1);
    A_m = _mm_shuffle_ps(A_m, A_m, _MM_SHUFFLE(2, 3, 1, 0));
    C_m = _mm_shuffle_ps(C_m, C_m, _MM_SHUFFLE(2, 3, 1, 0));


    __declspec(align(16)) float C[4];
    _mm_store_ps(C, C_m);


    __m128 m128_C = _mm_set1_ps(C[0]);
    __m128 coeff0 = _mm_mul_ps(A_m, m128_C);


    m128_C = _mm_set1_ps(C[1]);
    __m128 coeff1 = _mm_mul_ps(A_m, m128_C);


    m128_C = _mm_set1_ps(C[2]);
    __m128 coeff2 = _mm_mul_ps(A_m, m128_C);


    m128_C = _mm_set1_ps(C[3]);
    __m128 coeff3 = _mm_mul_ps(A_m, m128_C);

    ///


    int x0 = floor(x_float) - 1;
    int y0 = floor(y_float) - 1;


    __m128 sum_m = _mm_setzero_ps();

    __m128 src_m = _mm_set_ps(src.data[(x0 * src.cols + y0 + 3) * channels + d], src.data[(x0 * src.cols + y0 + 2) * channels + d], src.data[(x0 * src.cols + y0 + 1) * channels + d], src.data[(x0 * src.cols + y0) * channels + d]);
    sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff0));


    src_m = _mm_set_ps(src.data[((x0 + 1) * src.cols + y0 + 3) * channels + d], src.data[((x0 + 1) * src.cols + y0 + 2) * channels + d], src.data[((x0 + 1) * src.cols + y0 + 1) * channels + d], src.data[((x0 + 1) * src.cols + y0) * channels + d]);
    sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff1));

    src_m = _mm_set_ps(src.data[((x0 + 2) * src.cols + y0 + 3) * channels + d], src.data[((x0 + 2) * src.cols + y0 + 2) * channels + d], src.data[((x0 + 2) * src.cols + y0 + 1) * channels + d], src.data[((x0 + 2) * src.cols + y0) * channels + d]);
    sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff2));

    src_m = _mm_set_ps(src.data[((x0 + 3) * src.cols + y0 + 3) * channels + d], src.data[((x0 + 3) * src.cols + y0 + 2) * channels + d], src.data[((x0 + 3) * src.cols + y0 + 1) * channels + d], src.data[((x0 + 3) * src.cols + y0) * channels + d]);
    sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff3));


    float *p = (float *)&sum_m;
    unsigned char sum = (unsigned char)(p[0] + p[1] + p[2] + p[3]);


    return static_cast<unsigned char>(sum);
}

RGBImage ResizeImage(RGBImage src, float ratio) {
  const int channels = src.channels;
  Timer timer("resize image by 5x");
  int resize_rows = src.rows * ratio;
  int resize_cols = src.cols * ratio;

  printf("resize to: %d x %d\n", resize_rows, resize_cols);

  auto check_perimeter = [src](float x, float y) -> bool {
    return x < src.rows - 2 && x > 1 && y < src.cols - 2 && y > 1;
  };

  auto res = new unsigned char[channels * resize_rows * resize_cols];
  std::fill(res, res + channels * resize_rows * resize_cols, 0);

#  pragma omp parallel for num_threads(128) //设置线程的个数
  for (int i = 0; i < resize_rows; i++) {
    for (int j = 0; j < resize_cols; j++) {
      float src_x = i / ratio;
      float src_y = j / ratio;
      if (check_perimeter(src_x, src_y)) {
        for (int d = 0; d < channels; d++) {
          res[((i * resize_cols) + j) * channels + d] =
                  cubic_inner_SSE(src, src_x, src_y, channels, d);
        }
      }
    }
  }
  return RGBImage{resize_cols, resize_rows, channels, res};
}

#endif