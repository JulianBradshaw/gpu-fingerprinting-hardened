#pragma once
#include <stdint.h>

// RFC 8439 block function, 20 rounds. The GPU challenge is PUBLIC: this is
// cryptographic diffusion/domain separation, not a secret authentication key.
#ifdef __CUDACC__
#define FP_HD __host__ __device__ __forceinline__
#else
#define FP_HD inline
#endif
namespace fp {
constexpr uint32_t LOCAL_DOMAIN = 0x47505532u;
constexpr uint32_t GLOBAL_DOMAIN = 0x47505531u;
struct Key { uint32_t words[8]; };
FP_HD uint32_t rotl(uint32_t x, int n) { return (x << n) | (x >> (32-n)); }
FP_HD void quarter(uint32_t &a,uint32_t &b,uint32_t &c,uint32_t &d) {
    a+=b; d=rotl(d^a,16); c+=d; b=rotl(b^c,12);
    a+=b; d=rotl(d^a,8); c+=d; b=rotl(b^c,7);
}
FP_HD void block(const Key &key,uint32_t counter,uint32_t n0,uint32_t n1,uint32_t n2,uint32_t out[16]) {
    uint32_t state[16]={0x61707865u,0x3320646eu,0x79622d32u,0x6b206574u,
        key.words[0],key.words[1],key.words[2],key.words[3],
        key.words[4],key.words[5],key.words[6],key.words[7],counter,n0,n1,n2};
    uint32_t x[16];
    for(int i=0;i<16;i++) x[i]=state[i];
    for(int i=0;i<10;i++) {
        quarter(x[0],x[4],x[8],x[12]); quarter(x[1],x[5],x[9],x[13]);
        quarter(x[2],x[6],x[10],x[14]); quarter(x[3],x[7],x[11],x[15]);
        quarter(x[0],x[5],x[10],x[15]); quarter(x[1],x[6],x[11],x[12]);
        quarter(x[2],x[7],x[8],x[13]); quarter(x[3],x[4],x[9],x[14]);
    }
    for(int i=0;i<16;i++) out[i]=x[i]+state[i];
}
} // namespace fp
