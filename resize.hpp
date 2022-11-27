#ifndef RESIZE_H_
#define RESIZE_H_

#include "utils.hpp"
#include <cmath>

#include <immintrin.h>

constexpr float WeightCoeff(float x, float a)
{
    if (x <= 1)
        return 1 - (a + 3) * x * x + (a + 2) * x * x * x;
    else if (x < 2)
        return -4 * a + 8 * a * x - 5 * a * x * x + a * x * x * x;
    return 0.0;
}

constexpr void CalcCoeff4x4(float u, float v, float outCoeff[4][4])
{
    constexpr float a = -0.5f;

    u += 1.0f;
    v += 1.0f;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            outCoeff[i][j] = WeightCoeff(__builtin_fabs(u - i), a) * WeightCoeff(__builtin_fabs(v - j), a);
}

constexpr auto kNChannel = 3;
constexpr auto kRatio = 5;
constexpr auto kRatioFloat = static_cast<float>(kRatio);

struct CoeffTable
{
    constexpr CoeffTable()
    {
        for (auto ir = 0; ir < kRatio; ++ir)
        {
            const auto u = static_cast<float>(ir) / kRatioFloat;
            for (auto ic = 0; ic < kRatio; ++ic)
            {
                const auto v = static_cast<float>(ic) / kRatioFloat;
                CalcCoeff4x4(u, v, m_Coeffs[ir][ic]);
            }
        }
    }

    alignas(64) float m_Coeffs[kRatio][kRatio][4][4]{};
};

static constexpr CoeffTable kCoeffs;

RGBImage ResizeImage(RGBImage src, float ratio) {
    if (kNChannel != src.channels || kRatio != ratio)
        return {};

    Timer timer("resize image by 5x");

    const auto nRow = src.rows;
    const auto nCol = src.cols;
    const auto nResRow = nRow * kRatio;
    const auto nResCol = nCol * kRatio;

    printf("resize to: %d x %d\n", nResRow, nResCol);

    const auto pRes = new unsigned char[kNChannel * nResRow * nResCol]{};

    // Analysis of check_perimeter() in vanilla code:
    // srcRow = r + ir / kRatio
    // resRow = r * kRatio + ir
    // * srcRow <= 1
    //   * r + ir / kRatio <= 1
    //   * (r == 0) || (r == 1 && ir == 0)
    //   * resRow <= kRatio
    // * srcRow >= nRow - 2
    //   * r + ir / kRatio >= nRow - 2
    //   * (r >= nRow - 2)
    //   * resRow >= (nRow - 2) * kRatio
    // * 1 < srcRow < nRow - 2
    //   * (r == 1 && ir > 0) || (2 <= r < nRow - 2)
    //   * kRatio < resRow < nResRow - 2 * kRatio
    // For the sake of simplicity, we change the limit to
    // * 1 <= srcRow < nRow - 2
    //  * 1 <= r < nRow - 2
    //  * kRatio <= resRow < nResRow - 2 * kRatio
    // The above also holds for columns

#define PRECOMPUTE_COEFFS 1
#define SIMPLIFY_START 0
#define LOAD_IN_YMM 0
#define LOAD_IN_XMM 1
#define LOAD_IN_INTRIN 1

    for (auto r = 1; r < nRow - 2; ++r)
    {
        for (auto c = 1; c < nCol - 2; ++c)
        {
            alignas(32) float in[4][4][kNChannel];
        #if LOAD_IN_INTRIN
            {
            #if LOAD_IN_YMM
                const auto in00 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel]);
                const auto in01 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel + 8]);
                const auto in10 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[(r * nCol + (c - 1)) * kNChannel]);
                const auto in11 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[(r * nCol + (c - 1)) * kNChannel + 4]);
                const auto in20 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel]);
                const auto in21 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel + 8]);
                const auto in30 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel]);
                const auto in31 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel + 4]);
                const auto in0110 = _mm256_permute2x128_si256(_mm256_castsi128_si256(in01), _mm256_castsi128_si256(in10), 0b0010'0000);
                const auto in2130 = _mm256_permute2x128_si256(_mm256_castsi128_si256(in21), _mm256_castsi128_si256(in30), 0b0010'0000);
                const auto fin00 = _mm256_cvtepi32_ps(in00);
                const auto fin0110 = _mm256_cvtepi32_ps(in0110);
                const auto fin11 = _mm256_cvtepi32_ps(in11);
                const auto fin20 = _mm256_cvtepi32_ps(in20);
                const auto fin2130 = _mm256_cvtepi32_ps(in2130);
                const auto fin31 = _mm256_cvtepi32_ps(in31);
                _mm256_store_ps((float*)&in + 0, fin00);
                _mm256_store_ps((float*)&in + 8, fin0110);
                _mm256_store_ps((float*)&in + 16, fin11);
                _mm256_store_ps((float*)&in + 24, fin20);
                _mm256_store_ps((float*)&in + 32, fin2130);
                _mm256_store_ps((float*)&in + 40, fin31);
            #elif LOAD_IN_XMM
                const auto in00 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel + 0]);
                const auto in01 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel + 4]);
                const auto in02 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel + 8]);
                const auto in10 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 0) * nCol + (c - 1)) * kNChannel + 0]);
                const auto in11 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 0) * nCol + (c - 1)) * kNChannel + 4]);
                const auto in12 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 0) * nCol + (c - 1)) * kNChannel + 8]);
                const auto in20 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel + 0]);
                const auto in21 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel + 4]);
                const auto in22 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel + 8]);
                const auto in30 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel + 0]);
                const auto in31 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel + 4]);
                const auto in32 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel + 8]);
                const auto fin00 = _mm_cvtepi32_ps(in00);
                const auto fin01 = _mm_cvtepi32_ps(in01);
                const auto fin02 = _mm_cvtepi32_ps(in02);
                const auto fin10 = _mm_cvtepi32_ps(in10);
                const auto fin11 = _mm_cvtepi32_ps(in11);
                const auto fin12 = _mm_cvtepi32_ps(in12);
                const auto fin20 = _mm_cvtepi32_ps(in20);
                const auto fin21 = _mm_cvtepi32_ps(in21);
                const auto fin22 = _mm_cvtepi32_ps(in22);
                const auto fin30 = _mm_cvtepi32_ps(in30);
                const auto fin31 = _mm_cvtepi32_ps(in31);
                const auto fin32 = _mm_cvtepi32_ps(in32);
                _mm_store_ps((float*)&in[0] + 0, fin00);
                _mm_store_ps((float*)&in[0] + 4, fin01);
                _mm_store_ps((float*)&in[0] + 8, fin02);
                _mm_store_ps((float*)&in[1] + 0, fin10);
                _mm_store_ps((float*)&in[1] + 4, fin11);
                _mm_store_ps((float*)&in[1] + 8, fin12);
                _mm_store_ps((float*)&in[2] + 0, fin20);
                _mm_store_ps((float*)&in[2] + 4, fin21);
                _mm_store_ps((float*)&in[2] + 8, fin22);
                _mm_store_ps((float*)&in[3] + 0, fin30);
                _mm_store_ps((float*)&in[3] + 4, fin31);
                _mm_store_ps((float*)&in[3] + 8, fin32);
            #else
                const auto in00 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel]);
                const auto in01 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r - 1) * nCol + (c - 1)) * kNChannel + 8]);
                const auto in10 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[(r * nCol + (c - 1)) * kNChannel]);
                const auto in11 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[(r * nCol + (c - 1)) * kNChannel + 8]);
                const auto in20 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel]);
                const auto in21 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 1) * nCol + (c - 1)) * kNChannel + 8]);
                const auto in30 = _mm256_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel]);
                const auto in31 = _mm_cvtepu8_epi32(*(const __m128i*)&src.data[((r + 2) * nCol + (c - 1)) * kNChannel + 8]);
                const auto fin00 = _mm256_cvtepi32_ps(in00);
                const auto fin01 = _mm_cvtepi32_ps(in01);
                const auto fin10 = _mm256_cvtepi32_ps(in10);
                const auto fin11 = _mm_cvtepi32_ps(in11);
                const auto fin20 = _mm256_cvtepi32_ps(in20);
                const auto fin21 = _mm_cvtepi32_ps(in21);
                const auto fin30 = _mm256_cvtepi32_ps(in30);
                const auto fin31 = _mm_cvtepi32_ps(in31);
                _mm256_store_ps(&in[0][0][0], fin00);
                _mm_store_ps(&in[0][2][2], fin01);
                _mm256_store_ps(&in[1][0][0], fin10);
                _mm_store_ps(&in[1][2][2], fin11);
                _mm256_store_ps(&in[2][0][0], fin20);
                _mm_store_ps(&in[2][2][2], fin21);
                _mm256_store_ps(&in[3][0][0], fin30);
                _mm_store_ps(&in[3][2][2], fin31);
            #endif
            }
        #else
            for (auto i = 0; i < 4; ++i)
                for (auto j = 0; j < 4; ++j)
                    for (auto ch = 0; ch < kNChannel; ++ch)
                        in[i][j][ch] = src.data[((r + i - 1) * nCol + (c + j - 1)) * kNChannel + ch];
        #endif

        #if SIMPLIFY_START
            for (auto ir = 0; ir < kRatio; ++ir)
        #else
            for (auto ir = r == 1 ? 1 : 0; ir < kRatio; ++ir)
        #endif
            {
            #if SIMPLIFY_START
                for (auto ic = 0; ic < kRatio; ++ic)
            #else
                for (auto ic = c == 1 ? 1 : 0; ic < kRatio; ++ic)
            #endif
                {
                #if PRECOMPUTE_COEFFS
                    const auto& coeffs = kCoeffs.m_Coeffs[ir][ic];
                #else
                    float coeffs[4][4];
                    const auto x = float(r * kRatio + ir) / kRatioFloat;
                    const auto y = float(c * kRatio + ic) / kRatioFloat;
                    CalcCoeff4x4(x - floor(x), y - floor(y), coeffs);
                #endif
                    float sums[kNChannel]{};
                    for (auto i = 0; i < 4; ++i)
                        for (auto j = 0; j < 4; ++j)
                            for (auto ch = 0; ch < kNChannel; ++ch)
                                sums[ch] += coeffs[i][j] * in[i][j][ch];
                    for (int ch = 0; ch < kNChannel; ++ch)
                        pRes[((r * kRatio + ir) * nResCol + (c * kRatio + ic)) * kNChannel + ch] = static_cast<unsigned char>(sums[ch]);
                }
            }
        }
    }

    return RGBImage{nResCol, nResRow, kNChannel, pRes};
}

#endif
