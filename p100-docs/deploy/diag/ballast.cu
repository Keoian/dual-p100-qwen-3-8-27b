// hold GPU memory on one device until killed: ballast <device> <leave_free_MiB>
# build: nvcc -O2 -arch=sm_60 -o ballast ballast.cu   (simulates low VRAM headroom while testing)
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>
int main(int argc, char ** argv) {
    int dev = atoi(argv[1]); size_t leave = (size_t) atol(argv[2]) << 20;
    cudaSetDevice(dev);
    size_t fr, tot; cudaMemGetInfo(&fr, &tot);
    if (fr <= leave) { printf("already only %zu MiB free\n", fr >> 20); fflush(stdout); pause(); }
    size_t n = fr - leave; void * p = nullptr;
    while (cudaMalloc(&p, n) != cudaSuccess && n > (64u << 20)) n -= 16u << 20;
    cudaMemset(p, 0, n); cudaDeviceSynchronize();
    cudaMemGetInfo(&fr, &tot);
    printf("ballast: device %d holds %zu MiB, free now %zu MiB\n", dev, n >> 20, fr >> 20); fflush(stdout);
    pause();
}
