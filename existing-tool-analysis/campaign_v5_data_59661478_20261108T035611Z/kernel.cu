// Atomic race condition probe.
//
// One thread from each SM race to perform atomicAdd operations to increment
// a shared global counter. The return value of atomicAdd is the previous value
// of the counter, which tells us the thread order.
//
// i.e. first thread will receive 0 and sets the counter to 1,
// second thread will receive 1 and sets the counter to 2, and so on.
//
// This is repeated across multiple rounds, and the resulting order each round
// forms the fingerprint that is then stored in results.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

namespace cg = cooperative_groups;

// Standard cuda error checking macro
#define CHECK_CUDA(expr_to_check)                          \
    do                                                     \
    {                                                      \
        cudaError_t result = expr_to_check;                \
        if (result != cudaSuccess)                         \
        {                                                  \
            fprintf(stderr,                                \
                    "CUDA Runtime Error: %s:%i:%d = %s\n", \
                    __FILE__,                              \
                    __LINE__,                              \
                    result,                                \
                    cudaGetErrorString(result));           \
            exit(1);                                       \
        }                                                  \
    } while (0)

#define N_ROUNDS 250
#define N_WARMUP 50
#define THREADS_PER_BLOCK 64

__device__ int counter; // Global counter, target of atomicAdd

__global__ void atomic_race(int32_t *results, uint32_t seed)
{
    cg::grid_group grid = cg::this_grid();

    unsigned int smid; // Get id of current streaming multiprocessor
    asm("mov.u32 %0, %%smid;" : "=r"(smid));

    uint32_t lcg_state = seed ^ smid;       // linear congruential random number generator
    __shared__ int32_t local_buf[N_ROUNDS]; // Local buffer to store results

    for (int r = 0; r < N_ROUNDS; r++)
    {
        lcg_state = (lcg_state * 1103515245u + 12345u) & 0x7fffffffu;
        if (r < N_WARMUP)
        {
            grid.sync();
            continue;
        }
        else if (r % 70 <= 10) // Reset occasionally
        {
            grid.sync();
            for (int i = 0; i < 2; i++)
                __nanosleep(1000000);
            __nanosleep(smid * 10);
        }
        if (threadIdx.x == lcg_state % THREADS_PER_BLOCK) // Choose racing thread randomly
        {
            // Random delay
            float acc = 2.0f;
            for (int i = 0; i < 100000 + lcg_state % 17; i++)
            {
                acc = rsqrtf(acc);
            }
            __threadfence();

            // Race to write to counter, store order in local_buf
            if (smid % 2 == 0)                         // Reduce data collected
                local_buf[r] = atomicAdd(&counter, 1); // Race condition

            // This never happens, code here to prevent dead code elimination of delay loop above by complier
            if (acc == -1.0f)
                local_buf[r] = (int32_t)-10;
        }
    }

    // Save results in local buffer to global memory
    __syncthreads();
    int32_t *dst = results + (size_t)smid / 2 * N_ROUNDS;
    if (smid % 2 == 0)
    {
        for (int r = threadIdx.x; r < N_ROUNDS; r += blockDim.x)
            dst[r] = local_buf[r];
    }
}

int main(int argc, char **argv)
{
    // Batched: run every seed given on the command line in ONE process (one CUDA context).
    if (argc < 2) { fprintf(stderr, "usage: probe <seed> [seed ...]\n"); return 1; }

    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp prop; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int n_sms = prop.multiProcessorCount;

    char uuid[80];
    const unsigned char *b = (const unsigned char *)prop.uuid.bytes;
    snprintf(uuid, sizeof(uuid),
             "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]);
    printf("# device=%s\n# gpu_uuid=%s\n# n_sms=%d\n# compute_cap=%d.%d\n",
           prop.name, uuid, n_sms, prop.major, prop.minor);

    int can_coop = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&can_coop, cudaDevAttrCooperativeLaunch, 0));
    if (!can_coop) { fprintf(stderr, "Device does not support cooperative launch\n"); exit(1); }

    size_t results_size = (size_t)N_ROUNDS * n_sms / 2 * sizeof(int32_t);
    int32_t *d_results; CHECK_CUDA(cudaMalloc(&d_results, results_size));
    int32_t *h_results = (int32_t *)malloc(results_size);

    for (int a = 1; a < argc; a++) {
        uint32_t seed = (uint32_t)strtoul(argv[a], NULL, 10);
        int zero = 0; CHECK_CUDA(cudaMemcpyToSymbol(counter, &zero, sizeof(int)));
        void *args[] = {&d_results, &seed};
        CHECK_CUDA(cudaLaunchCooperativeKernel((void *)atomic_race, n_sms, THREADS_PER_BLOCK, args, 0, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost));
        printf("# seed=%u\n", seed);
        for (int sm_id = 0; sm_id < n_sms; sm_id += 2) {
            printf("# sm=%d\n", sm_id);
            for (int r = N_WARMUP; r < N_ROUNDS; r++)
                printf(r > N_WARMUP ? ",%d" : "%d", h_results[(size_t)sm_id / 2 * N_ROUNDS + r]);
            printf("\n");
        }
    }
    CHECK_CUDA(cudaFree(d_results)); free(h_results);
    return 0;
}
