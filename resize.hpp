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
static_assert(kRatio * kNChannel <= 16, "YMM");

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
                CalcCoeff4x4(u, v, m_Data[ir][ic]);
            }
        }
    }

    constexpr decltype(auto) operator[](int ir) const { return (m_Data[ir]); }

    alignas(32) float m_Data[kRatio][kRatio][4][4]{};
};

static constexpr CoeffTable kCoeffs;

struct CoeffTableSwizzled
{
    constexpr CoeffTableSwizzled()
    {
        for (auto ir = 0; ir < kRatio; ++ir)
            for (auto i = 0; i < 4; ++i)
                for (auto j = 0; j < 4; ++j)
                    for (auto ic = 0; ic < kRatio; ++ic)
                        for (auto ch = 0; ch < kNChannel; ++ch)
                            m_Data[ir][i][j][ic * kNChannel + ch] = kCoeffs[ir][ic][i][j];
    }

    constexpr decltype(auto) operator[](int ir) const { return (m_Data[ir]); }

    alignas(32) float m_Data[kRatio][4][4][16]{};
};

static constexpr CoeffTableSwizzled kCoeffsSwizzled;

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

    //#pragma omp parallel for
    for (auto r = 1; r < nRow - 2; ++r)
    {
        for (auto c = 1; c < nCol - 2; ++c)
        {
            alignas(32) float in012[4][4][16]{};
            for (int i = 0; i < 4; ++i)
                for (auto j = 0; j < 4; ++j)
                    for (auto ic = 0; ic < kRatio; ++ic)
                        for (auto ch = 0; ch < kNChannel; ++ch)
                            in012[i][j][ic * kNChannel + ch] = src.data[((r + i - 1) * nCol + (c + j - 1)) * kNChannel + ch];

            //alignas(32) float in[4][4][kNChannel];
            //for (auto i = 0; i < 4; ++i)
            //    for (auto j = 0; j < 4; ++j)
            //        for (auto ch = 0; ch < kNChannel; ++ch)
            //            in[i][j][ch] = src.data[((r + i - 1) * nCol + (c + j - 1)) * kNChannel + ch];

        #if SIMPLIFY_START
            for (auto ir = 0; ir < kRatio; ++ir)
        #else
            for (auto ir = r == 1 ? 1 : 0; ir < kRatio; ++ir)
        #endif
            {
                //float sums[16]{};
                //const auto& coeffs = kCoeffsSwizzled[ir];
                //for (auto i = 0; i < 4; ++i)
                //    for (auto j = 0; j < 4; ++j)
                //        for (auto k = 0; k < 16; ++k)
                //            sums[k] += coeffs[i][j][k] * in012[i][j][k];
                //for (auto k = 0; k < 16; ++k)
                //    sums[k] = __builtin_fmin(__builtin_fmaxf(sums[k], 0.0f), 255.0f);
                //for (auto i = 0; i < 4; ++i)
                //    for (auto j = 0; j < 4; ++j)
                //        for (auto ic = 0; ic < kRatio; ++ic)
                //            for (auto ch = 0; ch < kNChannel; ++ch)
                //                sums[ic * kNChannel + ch] += kCoeffs[ir][ic][i][j] * in[i][j][ch];
                //const auto yf0 = _mm256_load_ps(&sums[0]);
                //const auto yf1 = _mm256_load_ps(&sums[8]);
                //for (auto k = 0; k < 16; ++k)
                //    pRes[((r * kRatio + ir) * nResCol + c * kRatio) * kNChannel + k] = sums[k];
                //for (auto ic = 0; ic < kRatio; ++ic)
                //{
                //    for (auto ch = 0; ch < kNChannel; ++ch)
                //    {
                //        //pRes[((r * kRatio + ir) * nResCol + c * kRatio) * kNChannel + (ic * kNChannel + ch)] = sums[ic * kNChannel + ch];
                //        pRes[((r * kRatio + ir) * nResCol +  c * kRatio + ic ) * kNChannel + ch] = sums[ic * kNChannel + ch];
                //        //pRes[((r * kRatio + ir) * nResCol + (c * kRatio + ic)) * kNChannel + ch] = sums[ic * kNChannel + ch];
                //    }
                //}
                const auto& coeffs = kCoeffsSwizzled.m_Data[ir];
                auto yf0 = _mm256_setzero_ps();
                auto yf1 = _mm256_setzero_ps();

                for (auto i = 0; i < 4; ++i)
                    for (auto j = 0; j < 4; ++j)
                    {
                        yf0 = _mm256_fmadd_ps(_mm256_load_ps(&coeffs[i][j][0]), _mm256_load_ps(&in012[i][j][0]), yf0);
                        yf1 = _mm256_fmadd_ps(_mm256_load_ps(&coeffs[i][j][8]), _mm256_load_ps(&in012[i][j][8]), yf1);
                    }
                //_mm256_store_ps(&sums[0], yf0);
                //_mm256_store_ps(&sums[8], yf1);

                //for (auto k = 0; k < 16; ++k)
                //    pRes[((r * kRatio + ir) * nResCol + c * kRatio) * kNChannel + k] = sums[k];

                const auto ydw0 = _mm256_cvttps_epi32(yf0);
                const auto ydw1 = _mm256_cvttps_epi32(yf1);
                const auto xdw00 = _mm256_castsi256_si128(ydw0);
                const auto xdw10 = _mm256_castsi256_si128(ydw1);
                const auto xdw01 = _mm256_extracti128_si256(ydw0, 1);
                const auto xdw11 = _mm256_extracti128_si256(ydw1, 1);
                const auto xw0 = _mm_packus_epi32(xdw00, xdw01);
                const auto xw1 = _mm_packus_epi32(xdw10, xdw11);
                const auto xw = _mm_packus_epi16(xw0, xw1);
                _mm_storeu_si128((__m128i*)&pRes[((r * kRatio + ir) * nResCol + c * kRatio) * kNChannel], xw);

                //__m128 h0, h1;
                //asm(
                //    R"???(
                //        vcvttps2dq %[y0], %[y0]
                //        vcvttps2dq %[y1], %[y1]
                //        vextracti128 $1, %[y0], %[h0]
                //        vextracti128 $1, %[y1], %[h1]
                //        vpackssdw %[h0], %x[y0], %x[y0]
                //        vpackssdw %[h1], %x[y1], %x[y1]
                //        vpackuswb %x[y1], %x[y0], %x[y0]
                //        vmovdqu %x[y0], %[res]
                //    )???"
                //    : [y0]"+x"(yf0)
                //    , [y1]"+x"(yf1)
                //    , [h0]"=x"(h0)
                //    , [h1]"=x"(h1)
                //    , [res]"=m"((__m128&)pRes[((r * kRatio + ir) * nResCol + c * kRatio) * kNChannel])
                //);
            //#if SIMPLIFY_START
            //    for (auto ic = 0; ic < kRatio; ++ic)
            //#else
            //    for (auto ic = c == 1 ? 1 : 0; ic < kRatio; ++ic)
            //#endif
            //    {
            //    #if PRECOMPUTE_COEFFS
            //        const auto& coeffs = kCoeffs.m_Data[ir][ic];
            //    #else
            //        float coeffs[4][4];
            //        const auto x = float(r * kRatio + ir) / kRatioFloat;
            //        const auto y = float(c * kRatio + ic) / kRatioFloat;
            //        CalcCoeff4x4(x - floor(x), y - floor(y), coeffs);
            //    #endif
            //        float sums[kNChannel]{};
            //        for (auto i = 0; i < 4; ++i)
            //            for (auto j = 0; j < 4; ++j)
            //                for (auto ch = 0; ch < kNChannel; ++ch)
            //                    sums[ch] += coeffs[i][j] * in[i][j][ch];
            //        for (int ch = 0; ch < kNChannel; ++ch)
            //            pRes[((r * kRatio + ir) * nResCol + (c * kRatio + ic)) * kNChannel + ch] = static_cast<unsigned char>(sums[ch]);
            //    }
            }
        }
    }

    return RGBImage{nResCol, nResRow, kNChannel, pRes};
}

#endif
