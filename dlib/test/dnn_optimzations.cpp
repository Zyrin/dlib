// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "../dnn.h"

#include "tester.h"
#include <random>

namespace
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.dnn_optimizations");

    void img2col_unoptimized(
        matrix<float>& output,
        const tensor& data,
        long n,
        long filter_nr,
        long filter_nc,
        long stride_y,
        long stride_x,
        long padding_y,
        long padding_x
    )
    {
        const auto d = data.host() + data.k()*data.nr()*data.nc()*n;
        const rectangle boundary = get_rect(data);

        const long out_nr = 1 + (data.nr() + 2 * padding_y - filter_nr) / stride_y;
        const long out_nc = 1 + (data.nc() + 2 * padding_x - filter_nc) / stride_x;

        output.set_size(out_nr*out_nc,
            data.k()*filter_nr*filter_nc);
        DLIB_CASSERT(output.size() != 0);
        float* t = &output(0, 0);

        // now fill in the Toeplitz output matrix for the n-th sample in data.  
        size_t cnt = 0;
        const long max_r = data.nr() + padding_y - (filter_nr - 1);
        const long max_c = data.nc() + padding_x - (filter_nc - 1);
        for(long r = -padding_y; r < max_r; r += stride_y)
        {
            for(long c = -padding_x; c < max_c; c += stride_x)
            {
                for(long k = 0; k < data.k(); ++k)
                {
                    for(long y = 0; y < filter_nr; ++y)
                    {
                        for(long x = 0; x < filter_nc; ++x)
                        {
                            DLIB_ASSERT(cnt < output.size());
                            long xx = c + x;
                            long yy = r + y;
                            if(boundary.contains(xx, yy))
                                *t = d[(k*data.nr() + yy)*data.nc() + xx];
                            else
                                *t = 0;
                            ++t;
                            ++cnt;
                        }
                    }
                }
            }
        }
    }

    void test_cpu_img2col()
    {
        print_spinner();

        std::mt19937 gen(0);

        auto randomTensor = [&gen](long long n, long long k, long long nr, long long nc)
        {
            resizable_tensor t(n, k, nr, nc);
            float* valPtr = t.host();
            std::uniform_real_distribution<float> dist(-10, 10);
            for(size_t i = 0; i < t.size(); ++i) 
            { *valPtr = dist(gen); }
            return t;
        };
        auto testEqualResults = [](const tensor& data, long filter_nr, long filter_nc,
            long stride_y, long stride_x, long padding_y, long padding_x)
        {
            matrix<float> m1, m2;
            for(int n = 0; n < data.num_samples(); ++n)
            {
                img2col_unoptimized(m1, data, n, filter_nr, filter_nc, stride_y, stride_x, padding_y, padding_x);
                dlib::cpu::detail::img2col(m2, data, n, filter_nr, filter_nc, stride_y, stride_x, padding_y, padding_x);
                DLIB_TEST(m1 == m2);
            }
        };

        const auto t = randomTensor(3, 5, 15, 15);

        testEqualResults(t, 1, 1, 1, 1, 0, 0);
        testEqualResults(t, 1, 1, 2, 2, 0, 0);
        testEqualResults(t, 1, 1, 3, 3, 0, 0);
        testEqualResults(t, 1, 1, 5, 5, 0, 0);
        testEqualResults(t, 1, 1, 1, 1, 1, 1);
        testEqualResults(t, 1, 1, 2, 2, 1, 1);
        testEqualResults(t, 1, 1, 3, 3, 1, 1);
        testEqualResults(t, 1, 1, 5, 5, 1, 1);
        testEqualResults(t, 1, 1, 1, 1, 2, 2);
        testEqualResults(t, 1, 1, 2, 2, 2, 2);
        testEqualResults(t, 1, 1, 3, 3, 2, 2);
        testEqualResults(t, 1, 1, 5, 5, 2, 2);

        testEqualResults(t, 3, 3, 1, 1, 0, 0);
        testEqualResults(t, 3, 3, 2, 2, 0, 0);
        testEqualResults(t, 3, 3, 3, 3, 0, 0);
        testEqualResults(t, 3, 3, 5, 5, 0, 0);
        testEqualResults(t, 3, 3, 1, 1, 1, 1);
        testEqualResults(t, 3, 3, 2, 2, 1, 1);
        testEqualResults(t, 3, 3, 3, 3, 1, 1);
        testEqualResults(t, 3, 3, 5, 5, 1, 1);
        testEqualResults(t, 3, 3, 1, 1, 2, 2);
        testEqualResults(t, 3, 3, 2, 2, 2, 2);
        testEqualResults(t, 3, 3, 3, 3, 2, 2);
        testEqualResults(t, 3, 3, 5, 5, 2, 2);

        testEqualResults(t, 5, 5, 1, 1, 0, 0);
        testEqualResults(t, 5, 5, 2, 2, 0, 0);
        testEqualResults(t, 5, 5, 3, 3, 0, 0);
        testEqualResults(t, 5, 5, 5, 5, 0, 0);
        testEqualResults(t, 5, 5, 1, 1, 1, 1);
        testEqualResults(t, 5, 5, 2, 2, 1, 1);
        testEqualResults(t, 5, 5, 3, 3, 1, 1);
        testEqualResults(t, 5, 5, 5, 5, 1, 1);
        testEqualResults(t, 5, 5, 1, 1, 2, 2);
        testEqualResults(t, 5, 5, 2, 2, 2, 2);
        testEqualResults(t, 5, 5, 3, 3, 2, 2);
        testEqualResults(t, 5, 5, 5, 5, 2, 2);

        dlog << LERROR << "test";
    }

    // ----------------------------------------------------------------------------------------

    class dnn_optimizations_tester : public tester
    {
    public:
        dnn_optimizations_tester(
        ) :
            tester("test_dnn_optimizations",
                "Test optimized dnn functions agaoinst basic forward implementations")
        {}

        void perform_test()
        {
            // make the tests repeatable
            srand(1234);

            test_cpu_img2col();
        }
    } a;
}