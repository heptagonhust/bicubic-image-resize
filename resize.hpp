#ifndef RESIZE_H_
#define RESIZE_H_

#include "utils.hpp"
#include <cmath>
#include <vector>
#include <thread>
#include <immintrin.h>

#define channels 3
#define ratio 5.0f
#define _ratio 0.2f

#define a -0.5f
#define a_mul_4 -2.0f
#define a_mul_5 -2.5f
#define a_mul_8 -4.0f
#define a_add_3 2.5f
#define a_add_2 1.5f

#define a -0.5f

unsigned char BGRAfterBiCubic(RGBImage src, float x_float, float y_float, int d)
{

  float u = x_float - floor(x_float);
  float v = y_float - floor(y_float);

  __m128 a_m = _mm_set1_ps(a);
  __m128 m_1 = _mm_set1_ps(1.0);
  __m128 a_mul_4_m = _mm_set1_ps(a_mul_4);
  __m128 a_mul_5_m = _mm_set1_ps(a_mul_5);
  __m128 a_mul_8_m = _mm_set1_ps(a_mul_8);
  __m128 a_add_3_m = _mm_set1_ps(a_add_3);
  __m128 a_add_2_m = _mm_set1_ps(a_add_2);

  __m128 C30_A30 = _mm_set_ps(2 - v, 1 + v, 2 - u, 1 + u); // C3 C0 A3 A0
  __m128 C21_A21 = _mm_set_ps(1 - v, v, 1 - u, u);         // C2 C1 A2 A1

  __m128 tmp0 = _mm_sub_ps(_mm_mul_ps(a_m, C30_A30), a_mul_5_m); // a*xx - a_mul_5
  tmp0 = _mm_add_ps(a_mul_8_m, _mm_mul_ps(C30_A30, tmp0));       // a_mul_8 + xx*(a*xx - a_mul_5)
  tmp0 = _mm_sub_ps(_mm_mul_ps(C30_A30, tmp0), a_mul_4_m);       // xx*(a_mul_8 + xx*(a*xx - a_mul_5)) - a_mul_4  = C3 C0 A3 A0

  __m128 tmp1 = _mm_sub_ps(_mm_mul_ps(a_add_2_m, C21_A21), a_add_3_m); // a_add_2*xx - a_add_3
  tmp1 = _mm_mul_ps(_mm_mul_ps(C21_A21, C21_A21), tmp1);               // xx*xx*(a_add_2*xx - a_add_3)
  tmp1 = _mm_add_ps(m_1, tmp1);                                        // 1 + xx*xx*(a_add_2*xx - a_add_3) = C2 C1 A2 A1

  __m128 A_m = _mm_unpacklo_ps(tmp0, tmp1);                // tmp1[1] tmp0[1] tmp1[0] tmp0[0] = A2 A3 A1 A0
  __m128 C_m = _mm_unpackhi_ps(tmp0, tmp1);                // tmp1[3] tmp0[3] tmp1[2] tmp0[2] = C2 C3 C1 C0
  A_m = _mm_shuffle_ps(A_m, A_m, _MM_SHUFFLE(2, 3, 1, 0)); // A3 A2 A1 A0
  C_m = _mm_shuffle_ps(C_m, C_m, _MM_SHUFFLE(2, 3, 1, 0)); // C3 C2 C1 C0

  float C[4];
  float *q = (float *)&C_m;
  for (int i = 0; i != 4; ++i)
  {
    C[i] = *(q + i);
  }

  __m128 coeff0[4];
  for (int i = 0; i != 4; ++i)
  {
    __m128 m128_C = _mm_set1_ps(C[i]);
    coeff0[i] = _mm_mul_ps(A_m, m128_C);
  }

  int x0 = floor(x_float) - 1;
  int y0 = floor(y_float) - 1;

  register __m128 sum_m = _mm_setzero_ps();

  for (int i = 0; i != 4; ++i)
  {
    __m128 src_m = _mm_set_ps(src.data[((x0)*src.cols + y0 + i) * channels + d], src.data[((x0 + 1) * src.cols + y0 + i) * channels + d], src.data[((x0 + 2) * src.cols + y0 + i) * channels + d], src.data[((x0 + 3) * src.cols + y0 + i) * channels + d]);
    sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff0[i]));
  }

  float *p = (float *)&sum_m;
  float sum = (float)(p[0] + p[1] + p[2] + p[3]);
  return static_cast<unsigned char>(sum);
}

RGBImage ResizeImage(RGBImage src)
{
  // const int channels = src.channels;
  Timer timer("resize image by 5x");
  int resize_rows = src.rows * ratio;
  int resize_cols = src.cols * ratio;

  printf("resize to: %d x %d\n", resize_rows, resize_cols);

  auto check_perimeter = [src](float x, float y) -> bool
  {
    return x < src.rows - 2 && x > 1 && y < src.cols - 2 && y > 1;
  };

  auto res = new unsigned char[channels * resize_rows * resize_cols];
  std::fill(res, res + channels * resize_rows * resize_cols, 0);
  std::vector<std::thread> pool1;
  // std::thread *pool=new std::thread[resize_rows+5];
  // int index=0;
  /////////////////////////////////////
  // float _ratio=1/ratio;
  for (register int i = 0; i != resize_rows; ++i)
  {
    std::thread t = std::thread([=]
                                {
      for (register int j = 0; j != resize_cols; ++j) {
        float src_x = i * _ratio;
        float src_y = j * _ratio;
        if (check_perimeter(src_x, src_y)) {
          for(int d=0;d!=channels;++d){
            res[((i * resize_cols) + j) * channels + d] =
                    BGRAfterBiCubic(src, src_x, src_y,  d);
                }
                /*
                res[((i * resize_cols) + j) * channels + 0] =
                    BGRAfterBiCubic(src, src_x, src_y,  0);
                res[((i * resize_cols) + j) * channels + 1] =
                    BGRAfterBiCubic(src, src_x, src_y, 1);
                res[((i * resize_cols) + j) * channels + 2] =
                    BGRAfterBiCubic(src, src_x, src_y, 2);
                    */
        }
      } });
    pool1.push_back(std::move(t));
    //*(pool+index)=std::move(t);
    // index++;
  }
  for (auto &t : pool1)
    t.join();
  // for(int i=0;i<index;i++) (*(pool+i)).join();
  // delete[] pool;

  return RGBImage{resize_cols, resize_rows, channels, res};
}

#endif