#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "chacha20_core.h"

fp::Key key_from_hex(const std::string &s) {
    if(s.size()!=64) throw std::runtime_error("bad key length");
    fp::Key key{};
    for(int i=0;i<32;i++) key.words[i/4]|=uint32_t(std::strtoul(s.substr(2*i,2).c_str(),nullptr,16))<<(8*(i%4));
    return key;
}
std::string hex(const uint32_t words[16]) {
    std::ostringstream out; out<<std::hex<<std::setfill('0');
    for(int i=0;i<16;i++) for(int b=0;b<4;b++) out<<std::setw(2)<<((words[i]>>(8*b))&255);
    return out.str();
}
int main(int argc,char **argv) {
    uint32_t output[16];
    if(argc==7 && std::string(argv[1])=="--block") {
        fp::block(key_from_hex(argv[2]),std::strtoul(argv[3],nullptr,0),std::strtoul(argv[4],nullptr,0),
                  std::strtoul(argv[5],nullptr,0),std::strtoul(argv[6],nullptr,0),output);
        std::cout<<hex(output)<<"\n";
        return 0;
    }
    const std::string key="000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
    const std::string expected="10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e";
    fp::block(key_from_hex(key),1,0x09000000u,0x4a000000u,0,output);
    if(hex(output)!=expected) { std::cerr<<"RFC 8439 block vector FAILED\n"; return 1; }
    std::cout<<"RFC 8439 block vector passed\n";
    return 0;
}
