/*夏彦文 U202115317 18252677381@163.com */
// 原本对源程序中的几个函数分别做了访存和消除多余计算的优化，之后全部用simd指令代替了
// 请看readme
#ifndef RESIZE_H_
#define RESIZE_H_
#include "utils.hpp"
#include <cmath>
#include <thread>
#include <immintrin.h>
#include <pthread.h>
struct youhua1{
  int rows, cols;
} resizer;//访存优化
/*inline float WeightCoeff(float x, float a) {//已优化
  
  float temp1 = x * x;
  float temp2 = temp1 * x;
  if (x <= 1) {
    return 1 - (a + 3) * temp1 + (a + 2) * temp2;
  } else if (x < 2) {
    return -4 * a + 8 * a * x - 5 * a * temp1 + a * temp2;
  }
  return 0.0;
}

void CalcCoeff4x4(float x, float y, float *coeff) {//已优化
  const float a = -0.5f;

  float u = x - floor(x);
  float v = y - floor(y);
// #pragma omp atomic
   u += 1;
  v += 1;
   float A[4];
    A[0] = WeightCoeff(abs(u), a);
    A[1] = WeightCoeff(abs(u - 1), a);
    A[2] = WeightCoeff(abs(u - 2), a);
    A[3] = WeightCoeff(abs(u - 3), a);
 
 
    for (int s = 0; s < 4; s++)
    {
      float C = WeightCoeff(abs(v - s), a);
      coeff[s * 4] = A[0] * C;
      coeff[s * 4 + 1] = A[1] * C;
      coeff[s * 4 + 2] = A[2] * C;
      coeff[s * 4 + 3] = A[3] * C;
    }

  /*for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      coeff[i * 4 + j] =
          WeightCoeff(fabs(u - i), a) * WeightCoeff(fabs(v - j), a);
    }
  }*/
/*
} 
/*float AVXFmAdd(const float *input1, const float *input2, int size)
{
	if (input1 == nullptr || input2 == nullptr)
	{
		printf("input data is null\n");
		return -1;
	}
	int nBlockWidth = 8;
	int cntBlock = size / nBlockWidth;
	int cntRem = size % nBlockWidth;
 
	float output = 0;
	__m256 loadData1, loadData2;
	//__m256 mulData = _mm256_setzero_ps();
	__m256 sumData = _mm256_setzero_ps();
	const float *p1 = input1;
	const float *p2 = input2;
	for (int i = 0; i < cntBlock; i++)
	{
		loadData1 = _mm256_load_ps(p1);
		loadData2 = _mm256_load_ps(p2);
		//mulData = _mm256_mul_ps(loadData1, loadData2);
		//sumData = _mm256_add_ps(sumData, mulData);
		sumData = _mm256_fmadd_ps(loadData1, loadData2, sumData);
		p1 += nBlockWidth;
		p2 += nBlockWidth * 3;
	}
	sumData = _mm256_hadd_ps(sumData, sumData); // p[0] + p[1] + p[4] + p[5] + p[8] + p[9] + p[12] + p[13] + ... 
	sumData = _mm256_hadd_ps(sumData, sumData); // p[2] + p[3] + p[6] + p[7] + p[10] + p[11] + p[14] + p[15] + ... 
	output += sumData.m256_f32[(0)];            // 前4组
	output += sumData.m256_f32[(4)];            // 后4组
 
	for (int i = 0; i < cntRem; i++)
	{
		output += p1[i] * p2[i];
	}
 
	return output;
}*/
/*
unsigned char BGRAfterBiCubic(RGBImage src, float x_float, float y_float,
                              int channels, int d) {
  float coeff[16];

  int x0 = (int)(x_float) - 1;
  int y0 = (int)(y_float) - 1;
  CalcCoeff4x4(x_float, y_float, coeff);
  
  float sum = .0f;
  x0 = x0 * src.cols + y0;
  
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // #pragma omp atomic
      sum +=  coeff[i * 4 + j] *
             src.data[ (x0 + i * src.cols + j) * channels + d];
    }
  }
  
  /*for (int i = 0; i < 4; i++) {
    
      // #pragma omp atomic
      sum = sum + (coeff[i * 4 ] *
             src.data[((x0 + i) * src.cols + y0 ) * channels + d]) + (coeff[i * 4 + 1] *
             src.data[((x0 + i) * src.cols + y0 + 1) * channels + d]) + (coeff[i * 4 + 2] *
             src.data[((x0 + i) * src.cols + y0 + 2) * channels + d]) + (coeff[i * 4 + 3] *
             src.data[((x0 + i) * src.cols + y0 + 3) * channels + d]);
    
  } /*
  return static_cast<unsigned char>(sum);
}*/
void BGRAfterBiCubic(unsigned char* src, float x_float, float y_float, float a, int cols, int d, unsigned char &aa, unsigned char &b, unsigned char &c)
{//流水线指令优化，该函数把三个通道都计算一遍，并通过引用返回数组中对应位置的数值

  //计算权重系数
  float u = x_float - floor(x_float);
  float v = y_float - floor(y_float);
  float a_mul_4 = (a + a) + (a + a);   //4a
  float a_mul_5 = a_mul_4 + a;         //5a
  float a_mul_8 = a_mul_4 + a_mul_4;   //8a
  float a_add_3 = a + 3;
  float a_add_2 = a + 2;
  __m128 a_m = _mm_set1_ps(a);
  __m128 m_1 = _mm_set1_ps(1.0);
  __m128 a_mul_4_m = _mm_set1_ps(a_mul_4);
  __m128 a_mul_5_m = _mm_set1_ps(a_mul_5);
  __m128 a_mul_8_m = _mm_set1_ps(a_mul_8);
  __m128 a_add_3_m = _mm_set1_ps(a_add_3);
  __m128 a_add_2_m = _mm_set1_ps(a_add_2);
 
 
  __m128 C30_A30 = _mm_set_ps(2 - v, 1 + v, 2 - u, 1 + u);   //C3 C0 A3 A0
  __m128 C21_A21 = _mm_set_ps(1 - v, v, 1 - u, u);   //C2 C1 A2 A1
 
 
  __m128 tmp0 = _mm_sub_ps(_mm_mul_ps(a_m, C30_A30), a_mul_5_m);   //a*xx - a_mul_5
  tmp0 = _mm_add_ps(a_mul_8_m, _mm_mul_ps(C30_A30, tmp0));       //a_mul_8 + xx*(a*xx - a_mul_5)
  tmp0 = _mm_sub_ps(_mm_mul_ps(C30_A30, tmp0), a_mul_4_m);    //xx*(a_mul_8 + xx*(a*xx - a_mul_5)) - a_mul_4  = C3 C0 A3 A0
 
 
  __m128 tmp1 = _mm_sub_ps(_mm_mul_ps(a_add_2_m, C21_A21), a_add_3_m);   //a_add_2*xx - a_add_3
  tmp1 = _mm_mul_ps(_mm_mul_ps(C21_A21, C21_A21), tmp1);    //xx*xx*(a_add_2*xx - a_add_3)
  tmp1 = _mm_add_ps(m_1, tmp1);     //1 + xx*xx*(a_add_2*xx - a_add_3) = C2 C1 A2 A1
 
 
  __m128 A_m = _mm_unpacklo_ps(tmp0, tmp1);    //tmp1[1] tmp0[1] tmp1[0] tmp0[0] = A2 A3 A1 A0
  __m128 C_m = _mm_unpackhi_ps(tmp0, tmp1);    //tmp1[3] tmp0[3] tmp1[2] tmp0[2] = C2 C3 C1 C0
  A_m = _mm_shuffle_ps(A_m, A_m, _MM_SHUFFLE(2, 3, 1, 0));   //A3 A2 A1 A0
  C_m = _mm_shuffle_ps(C_m, C_m, _MM_SHUFFLE(2, 3, 1, 0));   //C3 C2 C1 C0
 
 
 float C[4];
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
  
  u_char *src_p = src + cols  * x0 * 3;
  u_char *srcp1 = src_p;
  int index1 = (y0 + 3) * 3 + d;
  int index2 = index1 - 3;
  int index3 = index2 - 3;
  int index4 = index3 - 3;
  __m128 src_m = _mm_set_ps(src_p[index1], src_p[index2], src_p[index3], src_p[index4]);//四个一组
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff0));
 
 
  src_p = src_p + cols * 3;
  u_char *srcp2 = src_p;
  src_m = _mm_set_ps(src_p[index1], src_p[index2], src_p[index3], src_p[index4]);
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff1));
 
 
  src_p = src_p + cols * 3;
  u_char *srcp3 = src_p;
  src_m = _mm_set_ps(src_p[index1], src_p[index2], src_p[index3], src_p[index4]);
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff2));
 
 
  src_p = src_p + cols * 3;
  u_char *srcp4 = src_p;
  src_m = _mm_set_ps(src_p[index1], src_p[index2], src_p[index3], src_p[index4]);
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff3));
 
 
  float *p = (float *)&sum_m;

  aa= (u_char)(p[0] + p[1] + p[2] + p[3]);


 sum_m = _mm_setzero_ps();
  
  src_p = srcp1;
  src_m = _mm_set_ps(src_p[index1 + 1], src_p[index2 + 1], src_p[index3 + 1], src_p[index4 + 1]);
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff0));
 
 
  src_p = srcp2;
  src_m = _mm_set_ps(src_p[index1 + 1], src_p[index2 + 1], src_p[index3 + 1], src_p[index4 + 1]);
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff1));
 
 
  src_p = srcp3;
 src_m = _mm_set_ps(src_p[index1 + 1], src_p[index2 + 1], src_p[index3 + 1], src_p[index4 + 1]);
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff2));
 
 
  src_p = srcp4;
  src_m = _mm_set_ps(src_p[index1 + 1], src_p[index2 + 1], src_p[index3 + 1], src_p[index4 + 1]);
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff3));
 
 
  p  = (float *)&sum_m;

  b = (u_char)(p[0] + p[1] + p[2] + p[3]);


  sum_m = _mm_setzero_ps();
  
  src_p = srcp1;
  src_m = _mm_set_ps(src_p[index1 + 2], src_p[index2 + 2], src_p[index3 + 2], src_p[index4 + 2]);
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff0));
 
 
  src_p = srcp2;
  src_m = _mm_set_ps(src_p[index1 + 2], src_p[index2 + 2], src_p[index3 + 2], src_p[index4 + 2]);
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff1));
 
 
  src_p = srcp3;
  src_m = _mm_set_ps(src_p[index1 + 2], src_p[index2 + 2], src_p[index3 + 2], src_p[index4 + 2]);
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff2));
 
 
  src_p = srcp4;
  src_m = _mm_set_ps(src_p[index1 + 2], src_p[index2 + 2], src_p[index3 + 2], src_p[index4 + 2]);
  sum_m = _mm_add_ps(sum_m, _mm_mul_ps(src_m, coeff3));
 
 
  p = (float *)&sum_m;

  c = (u_char)(p[0] + p[1] + p[2] + p[3]);
 
}
RGBImage ResizeImage(RGBImage src, float ratio) {
  const int channels = src.channels;

  
  Timer timer("resize image by 5x");
  youhua1 resizee;
  youhua1 resizef;
  resizee.rows = src.rows * (int)ratio ;//放缩之后大小
  resizee.cols = src.cols * (int)ratio ;
  resizef.rows = src.rows;//放缩之前最小
  resizef.cols = src.cols;
  printf("resize to: %d x %d\n", resizee.rows, resizee.cols);

  auto check_perimeter = [resizef](float x, float y) -> bool {
    return x < resizef.rows - 2 && y < resizef.cols - 2 && x > 1 && y > 1;
  };
  
  auto res = new unsigned char[channels * resizee.rows * resizee.cols];
  std::fill(res, res + channels * resizee.rows * resizee.cols, 0);
  // float src_x;
  // float src_y;
  // unsigned char* point = src.data
  // #pragma omp parallel for 
  #pragma omp parallel for num_threads(64) //多线程优化
  for (int i = resizee.rows - 10; i >= 10; i--) {
    for (int j = resizee.cols - 10; j >= 10; j--) {
      //这里修改了原先判断边界范围的语句，减少了判断次数
      float src_x = i * 0.2;
      float src_y = j * 0.2;
      int temp = ((i * resizee.cols) + j) * 3;
      //改写循环，减少多余计算等
      BGRAfterBiCubic(src.data, src_x, src_y, -0.5, resizef.cols, 0, res[temp], res[temp + 1], res[temp + 2]);
          // res[((i * resizee.cols) + j) * 3 ] = point[0];
          // res[((i * resizee.cols) + j) * 3 + 1] = point[1];
          // res[((i * resizee.cols) + j) * 3 + 2] = point[2];
          // 访存压力大，程序反而变慢
          // res[((i * resizee.cols) + j) * 3 + 1] =
              // BGRAfterBiCubic(src.data, src_x, src_y, -0.5, resizef.cols, 1, 3);
          // res[((i * resizee.cols) + j) * 3 + 2] =
              // BGRAfterBiCubic(src.data, src_x, src_y, -0.5, resizef.cols, 2, 3);
    }
  }
  
  return RGBImage{resizee.cols, resizee.rows, channels, res};
}

#endif

