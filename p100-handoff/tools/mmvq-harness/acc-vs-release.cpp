// Accuracy of the CUDA mul_mat dispatch per weight type and batch width, against an exact (double)
// product of the *dequantized* weights and the f32 activations. Same seed -> same data, so two
// libggml-cuda builds (picked with LD_LIBRARY_PATH) can be compared case by case.
// out: type M K n nmse maxrel checksum
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <string>

static double run(ggml_backend_t be, ggml_type t, int M, int K, int n, const std::vector<char> & wq,
                  const std::vector<float> & x, const std::vector<double> & ref, double & maxrel, double & chk) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, t, K, M);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
    ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, Y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(W, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(X, x.data(), 0, x.size()*sizeof(float));
    ggml_backend_graph_compute(be, g);
    std::vector<float> y((size_t) M*n);
    ggml_backend_tensor_get(Y, y.data(), 0, y.size()*sizeof(float));
    double se = 0, sr = 0; maxrel = 0; chk = 0;
    double rms = 0; for (double r : ref) rms += r*r; rms = sqrt(rms/ref.size());
    for (size_t i = 0; i < y.size(); ++i) {
        const double d = (double) y[i] - ref[i];
        se += d*d; sr += ref[i]*ref[i];
        maxrel = std::max(maxrel, fabs(d)/rms);
        chk += (double) y[i]*(double)((i % 7) + 1);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return se/sr;
}

int main(int argc, char ** argv) {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no cuda\n"); return 1; }
    std::vector<std::string> names = { "q4_0","q4_1","q5_0","q5_1","q8_0","q2_K","q3_K","q4_K","q5_K","q6_K",
        "iq2_xxs","iq2_xs","iq2_s","iq3_xxs","iq3_s","iq1_s","iq1_m","iq4_nl","iq4_xs","mxfp4" };
    if (argc > 1) { names.clear(); for (int i = 1; i < argc; ++i) names.push_back(argv[i]); }
    const int shapes[3][2] = { {8704, 5120}, {5120, 8704}, {512, 5120} };   // M (rows), K
    for (const auto & nm : names) {
        ggml_type t = GGML_TYPE_COUNT;
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) { const char * s = ggml_type_name((ggml_type) i); if (s && nm == s) t = (ggml_type) i; }
        if (t == GGML_TYPE_COUNT) { fprintf(stderr, "unknown type %s\n", nm.c_str()); continue; }
        for (auto & sh : shapes) {
            const int M = sh[0], K = sh[1];
            std::mt19937 rng(1234 + M + K + (int) t);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> w((size_t) M*K);
            for (auto & v : w) v = nd(rng)*0.02f;
            std::vector<float> imat(K, 1.0f);
            std::vector<char> wq(ggml_row_size(t, K)*M);
            ggml_quantize_init(t);
            ggml_quantize_chunk(t, w.data(), wq.data(), 0, M, K, imat.data());
            std::vector<float> wd((size_t) M*K);
            ggml_get_type_traits(t)->to_float(wq.data(), wd.data(), (int64_t) M*K);
            for (int n = 1; n <= 16; ++n) {
                std::vector<float> x((size_t) K*n);
                for (auto & v : x) v = nd(rng);
                std::vector<double> ref((size_t) M*n);
                #pragma omp parallel for collapse(2)
                for (int c = 0; c < n; ++c) for (int r = 0; r < M; ++r) {
                    double s = 0; const float * wr = wd.data() + (size_t) r*K; const float * xc = x.data() + (size_t) c*K;
                    for (int k = 0; k < K; ++k) s += (double) wr[k]*xc[k];
                    ref[(size_t) c*M + r] = s;
                }
                double mr, chk;
                const double e = run(be, t, M, K, n, wq, x, ref, mr, chk);
                printf("%s %d %d %d %.4e %.4e %.10e\n", nm.c_str(), M, K, n, e, mr, chk);
                fflush(stdout);
            }
        }
    }
    ggml_backend_free(be);
    return 0;
}
