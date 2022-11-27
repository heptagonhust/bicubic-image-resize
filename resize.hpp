#ifndef RESIZE_H_
#define RESIZE_H_

#include "utils.hpp"
#include <cmath>

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

    for (auto r = 1; r < nRow - 2; ++r)
    {
    #if SIMPLIFY_START
        for (auto ir = 0; ir < kRatio; ++ir)
    #else
        for (auto ir = r == 1 ? 1 : 0; ir < kRatio; ++ir)
    #endif
        {
            for (auto c = 1; c < nCol - 2; ++c)
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
                                sums[ch] += coeffs[i][j] * src.data[((r + i - 1) * nCol + (c + j - 1)) * kNChannel + ch];
                    for (int ch = 0; ch < kNChannel; ++ch)
                        pRes[((r * kRatio + ir) * nResCol + (c * kRatio + ic)) * kNChannel + ch] = static_cast<unsigned char>(sums[ch]);
                }
            }
        }
    }

    return RGBImage{nResCol, nResRow, kNChannel, pRes};
}

#endif
