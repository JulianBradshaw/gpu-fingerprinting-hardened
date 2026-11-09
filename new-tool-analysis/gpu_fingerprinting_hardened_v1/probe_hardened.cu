// Research probe: requires NEW enrollment; spoof resistance is unmeasured.
// Build: nvcc -std=c++17 -O2 -arch=sm_90 probe_hardened.cu -o probe_hardened
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "chacha20_core.h"

namespace cg=cooperative_groups;
constexpr int THREADS=64, WARMUP=32, BURST=3, BANKS=16;
constexpr int DELAY_BASE=2048, DELAY_JITTER=2048;
struct alignas(128) Counter { unsigned value; unsigned padding[31]; };
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

__global__ void probe(fp::Key key,int rounds,bool subset,int *results,
                      unsigned *smids,unsigned *priorities,Counter *counters,unsigned *delay_sink) {
    auto grid=cg::this_grid();
    __shared__ unsigned smid,lane,delay,active,bank,k;
    if(threadIdx.x==0) {
        unsigned actual_smid;
        asm("mov.u32 %0, %%smid;":"=r"(actual_smid));
        smid=actual_smid;
        smids[blockIdx.x]=smid;
    }
    grid.sync();
    for(int round=0;round<WARMUP+rounds;round++) {
        if(threadIdx.x==0) {
            uint32_t local[16],global[16];
            fp::block(key,0,smid,round,fp::LOCAL_DOMAIN,local);
            fp::block(key,0,0xffffffffu,round,fp::GLOBAL_DOMAIN,global);
            lane=local[0]%THREADS;
            delay=DELAY_BASE+local[1]%(DELAY_JITTER+1);
            priorities[blockIdx.x]=local[2];
            bank=global[1]%BANKS;
            k=subset ? 4+global[0]%5 : gridDim.x;
            if(blockIdx.x==0) counters[bank].value=0;
            if(round>=WARMUP) results[(round-WARMUP)*gridDim.x+blockIdx.x]=-1;
        }
        grid.sync();
        if(threadIdx.x==0) {
            unsigned rank=0,p=priorities[blockIdx.x];
            for(unsigned b=0;b<gridDim.x;b++)
                if(priorities[b]<p || (priorities[b]==p && smids[b]<smid)) rank++;
            active=rank<k;
        }
        __syncthreads();
        if(active && threadIdx.x==lane) {
            unsigned accumulator=smid^unsigned(round)^key.words[0];
            for(unsigned i=0;i<delay;i++) {
                // Volatile dependent instructions keep delay work observable.
                asm volatile("mad.lo.u32 %0, %0, 1664525, 1013904223;":"+r"(accumulator));
            }
            delay_sink[blockIdx.x]=accumulator;
            __threadfence();
            unsigned position=atomicAdd(&counters[bank].value,1u);
            if(round>=WARMUP) results[(round-WARMUP)*gridDim.x+blockIdx.x]=int(position);
        }
        grid.sync(); // Never conditionally skip a collective barrier.
    }
}

fp::Key parse_key(const std::string &s) {
    if(s.size()!=64) throw std::runtime_error("challenge must be exactly 64 hexadecimal characters");
    fp::Key key{};
    for(int i=0;i<32;i++) {
        if(!std::isxdigit(static_cast<unsigned char>(s[2*i])) ||
           !std::isxdigit(static_cast<unsigned char>(s[2*i+1]))) throw std::runtime_error("invalid challenge");
        unsigned b=std::stoul(s.substr(2*i,2),nullptr,16);
        key.words[i/4]|=b<<(8*(i%4));
    }
    return key;
}

std::string profile_json(bool subset,int rounds) {
    return std::string("{\"format\":\"gpu-fp-hardened-v1\",\"prng\":\"chacha20-public-challenge\",\"mode\":\"")+
        (subset?"subset":"all")+"\",\"rounds\":"+std::to_string(rounds)+
        ",\"warmup_rounds\":32,\"burst\":3,\"threads\":64,\"banks\":16,\"delay_base\":2048,\"delay_jitter\":2048}";
}

int main(int argc,char **argv) {
    try {
        std::string mode=argc>1?argv[1]:"subset";
        if(mode!="all" && mode!="subset") throw std::runtime_error("mode must be all or subset");
        int rounds=128;
        if(argc>2) { size_t used=0; rounds=std::stoi(argv[2],&used); if(used!=std::string(argv[2]).size()) throw std::runtime_error("invalid rounds"); }
        if(argc>3 || rounds<16 || rounds>512) throw std::runtime_error("usage: probe_hardened [all|subset] [rounds 16..512]");
        bool subset=mode=="subset";
        check(cudaSetDevice(0));
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,0));
        int sms=prop.multiProcessorCount,cooperative=0,occupancy=0;
        if(sms<8 || sms>1024) throw std::runtime_error("requires 8..1024 SMs");
        check(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,0));
        if(!cooperative) throw std::runtime_error("cooperative launch unsupported");
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,probe,THREADS,0));
        if(occupancy<1) throw std::runtime_error("kernel cannot reside on every SM");
        int *d_results=nullptr; unsigned *d_smids=nullptr,*d_priorities=nullptr,*d_sink=nullptr;
        Counter *d_counters=nullptr;
        check(cudaMalloc(&d_results,size_t(rounds)*sms*sizeof(int)));
        check(cudaMalloc(&d_smids,sms*sizeof(unsigned)));
        check(cudaMalloc(&d_priorities,sms*sizeof(unsigned)));
        check(cudaMalloc(&d_sink,sms*sizeof(unsigned)));
        check(cudaMalloc(&d_counters,BANKS*sizeof(Counter)));
        cudaEvent_t start,stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
        std::vector<int> raw(size_t(rounds)*sms);
        std::vector<unsigned> mapping(sms),expected_smids;
        auto launch=[&](fp::Key key,std::vector<int> *aligned)->float {
            void *args[]={&key,&rounds,&subset,&d_results,&d_smids,&d_priorities,&d_counters,&d_sink};
            check(cudaEventRecord(start));
            check(cudaLaunchCooperativeKernel((void*)probe,sms,THREADS,args,0,nullptr));
            check(cudaGetLastError()); check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
            float ms=0; check(cudaEventElapsedTime(&ms,start,stop));
            check(cudaMemcpy(mapping.data(),d_smids,sms*sizeof(unsigned),cudaMemcpyDeviceToHost));
            std::vector<unsigned> sorted=mapping; std::sort(sorted.begin(),sorted.end());
            if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end() || sorted.back()==0xffffffffu)
                throw std::runtime_error("launch did not map one block to each unique SM; refusing measurement");
            if(expected_smids.empty()) expected_smids=sorted;
            if(sorted!=expected_smids) throw std::runtime_error("SM set changed within process");
            if(aligned) {
                check(cudaMemcpy(raw.data(),d_results,raw.size()*sizeof(int),cudaMemcpyDeviceToHost));
                std::vector<int> order(sms); std::iota(order.begin(),order.end(),0);
                std::sort(order.begin(),order.end(),[&](int a,int b){return mapping[a]<mapping[b];});
                aligned->resize(raw.size());
                for(int r=0;r<rounds;r++) for(int col=0;col<sms;col++)
                    (*aligned)[size_t(r)*sms+col]=raw[size_t(r)*sms+order[col]];
            }
            return ms;
        };
        fp::Key warm{}; launch(warm,nullptr); launch(warm,nullptr);
        char uuid[80]; const unsigned char *b=(const unsigned char*)prop.uuid.bytes;
        std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]);
        const auto profile=profile_json(subset,rounds);
        std::cout<<"{\"type\":\"ready\",\"device_uuid\":\""<<uuid<<"\",\"profile\":"<<profile<<"}"<<std::endl;
        std::string line;
        while(std::getline(std::cin,line)) {
            if(!line.empty() && line.back()=='\r') line.pop_back();
            fp::Key key=parse_key(line);
            std::transform(line.begin(),line.end(),line.begin(),[](unsigned char c){return char(std::tolower(c));});
            auto before=std::chrono::steady_clock::now();
            std::vector<std::vector<int>> fingerprints(BURST); std::vector<float> milliseconds(BURST);
            for(int burst=0;burst<BURST;burst++) milliseconds[burst]=launch(key,&fingerprints[burst]);
            double host_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
            std::cout<<std::setprecision(9)<<"{\"type\":\"measurement\",\"challenge\":\""<<line
                <<"\",\"device_uuid\":\""<<uuid<<"\",\"profile\":"<<profile<<",\"sm_ids\":[";
            for(int i=0;i<sms;i++) std::cout<<(i?",":"")<<expected_smids[i];
            std::cout<<"],\"kernel_ms\":[";
            for(int i=0;i<BURST;i++) std::cout<<(i?",":"")<<milliseconds[i];
            std::cout<<"],\"host_measurement_ms\":"<<host_ms<<",\"fingerprints\":[";
            for(int burst=0;burst<BURST;burst++) {
                if(burst) std::cout<<","; std::cout<<"[";
                for(int r=0;r<rounds;r++) {
                    if(r) std::cout<<","; std::cout<<"[";
                    for(int s=0;s<sms;s++) std::cout<<(s?",":"")<<fingerprints[burst][size_t(r)*sms+s];
                    std::cout<<"]";
                }
                std::cout<<"]";
            }
            std::cout<<"]}"<<std::endl;
        }
        check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
        check(cudaFree(d_results)); check(cudaFree(d_smids)); check(cudaFree(d_priorities));
        check(cudaFree(d_sink)); check(cudaFree(d_counters));
        return 0;
    } catch(const std::exception &error) {
        std::cerr<<"probe refused: "<<error.what()<<std::endl;
        return 1;
    }
}
