#include <chrono>
#include <cstring>
#include <vector>
#include <format>
#include <encodec.h>
#include <encodec_encoder24.h>
#include <encodec_decoder24.h>
#include <encodec_rvq24.h>

using namespace std::chrono;

template<class T>
auto load_file(const char* file)
{
    std::vector<T> data;
    FILE* fp = fopen(file, "rb");
    if (!fp)
    {
        printf("Failed to open `%s`\n", file);
        return data;
    }

    fseek(fp, 0, SEEK_END);
    const auto size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    data.resize(size/sizeof(T));
    auto nread = fread((char*)&data[0], sizeof(T), data.size(), fp);
    if (nread != data.size())
        printf("Read %zu/%zu samples\n", nread, data.size());
    return data;
}

template<class C>
void save_file(const char* file, const C& data)
{
    using T = typename C::value_type;

    FILE* fp = fopen(file, "wb");
    if (!fp)
    {
        printf("Failed to open `%s`\n", file);
        return;
    }

    fwrite(data.data(), sizeof(T), data.size(), fp);
    fclose(fp);
}

void test_birch_canoe(encodec::encoder& enc, encodec::decoder& dec, unsigned int bps)
{
    const auto nlevels = encodec::get_encoded_nquantizers(bps);
    const auto audio   = load_file<float>("original.dat");
    const auto packets = enc.encode(audio, nlevels);
    const auto audio2  = dec.decode(packets, nlevels);
    save_file(std::format("codes_{}bps_{}nquants.dat", bps, nlevels).c_str(),  packets);
    save_file(std::format("encoded_{}bps_{}nquants.dat", bps, nlevels).c_str(), audio2);
}

void bench(encodec::encoder& enc, encodec::decoder& dec, unsigned int bps)
{
    float audio[24000];
    memset(audio, 0, sizeof(audio));
    
    // Warmup
    auto packet = enc.encode(audio, encodec::get_encoded_nquantizers(bps));
    auto audio2 = dec.decode(packet, encodec::get_encoded_nquantizers(bps));

    const int ntests = 100;
    const auto s0 = high_resolution_clock::now();
    for (size_t i{0} ; i < ntests ; ++i)
        packet = enc.encode(audio, encodec::get_encoded_nquantizers(bps));
    const auto s1 = high_resolution_clock::now();
    printf("Encoding rate %f encoded %zu samples into %zu bits\n", (std::size(audio)*ntests)/((s1-s0).count()*1e-9), std::size(audio), packet.size()*8);

    const auto s2 = high_resolution_clock::now();
    for (size_t i{0} ; i < ntests ; ++i)
        audio2 = dec.decode(packet, encodec::get_encoded_nquantizers(bps));
    const auto s3 = high_resolution_clock::now();
    printf("Decoding rate %f %zu bits into decoded %zu samples\n", (std::size(audio2)*ntests)/((s3-s2).count()*1e-9), packet.size()*8, audio2.size());
}


int main()
{
    encodec::encoder enc(encodec::get_encoder24_weights(), encodec::get_rvq24_weights());
    encodec::decoder dec(encodec::get_decoder24_weights(), encodec::get_rvq24_weights());

    printf("Testing on birch canoe...\n");
    test_birch_canoe(enc, dec, 24000);
    test_birch_canoe(enc, dec, 12000);
    test_birch_canoe(enc, dec, 6000);
    test_birch_canoe(enc, dec, 3000);
    printf("Testing on birch canoe... Done\n");
    
    printf("Bench...\n");
    bench(enc, dec, 24000);
    printf("Bench... Done\n");

    return 0;
}