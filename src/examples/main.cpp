#include <chrono>
#include <cstring>
#include <vector>
#include <encodec.h>

using namespace std::chrono;

int main()
{
    encodec::encoder enc;
    encodec::decoder dec;

    float audio[24000];
    memset(audio, 0, sizeof(audio));
    
    // Warmup
    auto packet = enc.encode(audio, 32);
    auto audio2 = dec.decode(packet, 32);

    const int ntests = 100;
    const auto s0 = high_resolution_clock::now();
    for (size_t i{0} ; i < ntests ; ++i)
        packet = enc.encode(audio, 32);
    const auto s1 = high_resolution_clock::now();
    printf("Encoding rate %f encoded %zu samples into %zu bits\n", (std::size(audio)*ntests)/((s1-s0).count()*1e-9), std::size(audio), packet.size()*8);

    const auto s2 = high_resolution_clock::now();
    for (size_t i{0} ; i < ntests ; ++i)
        audio2 = dec.decode(packet, 32);
    const auto s3 = high_resolution_clock::now();
    printf("Decoding rate %f %zu bits into decoded %zu samples\n", (std::size(audio2)*ntests)/((s3-s2).count()*1e-9), packet.size()*8, audio2.size());

    return 0;
}