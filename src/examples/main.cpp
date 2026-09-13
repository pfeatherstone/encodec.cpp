#include <chrono>
#include <cstring>
#include <vector>
#include <sstream>
#include <encodec.h>

using namespace std::chrono;
using namespace encodec;

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

auto format(auto... args)
{
    std::stringstream ss;
    (ss << ... << args);
    return ss.str();
}

void test_birch_canoe(encoder& enc, decoder& dec, bitrates bps, const char* input)
{
    printf("test_birch_canoe bps %u...\n", bps);
    const auto audio            = load_file<float>(input);
    const auto [scale, packets] = enc.encode(audio, bps);
    const auto audio2           = dec.decode(packets, scale, bps);
    printf("test_birch_canoe bps %u... scale %f packet size %zu Done\n", bps, scale, packets.size());
    save_file(format("codes_", bps, "bps_", enc.get_rate(), "hz.dat").c_str(),  packets);
    save_file(format("encoded_", bps, "bps_", enc.get_rate(), "hz.dat").c_str(), audio2);
}

void bench(encoder& enc, decoder& dec, bitrates bps)
{
    float audio[24000];
    memset(audio, 0, sizeof(audio));
    
    // Warmup
    auto [scale, packet] = enc.encode(audio, bps);
    auto audio2          = dec.decode(packet, scale, bps);

    const int ntests = 100;
    const auto s0 = high_resolution_clock::now();
    for (size_t i{0} ; i < ntests ; ++i)
        std::tie(scale, packet) = enc.encode(audio, bps);
    const auto s1 = high_resolution_clock::now();
    printf("Encoding rate %f encoded %zu samples into %zu bits\n", (std::size(audio)*ntests)/((s1-s0).count()*1e-9), std::size(audio), packet.size()*8);

    const auto s2 = high_resolution_clock::now();
    for (size_t i{0} ; i < ntests ; ++i)
        audio2 = dec.decode(packet, scale, bps);
    const auto s3 = high_resolution_clock::now();
    printf("Decoding rate %f %zu bits into decoded %zu samples\n", (std::size(audio2)*ntests)/((s3-s2).count()*1e-9), packet.size()*8, audio2.size());
}


int main()
{
    encoder enc(RATE_24KHZ, get_encoder24_weights(), get_rvq24_weights());
    decoder dec(RATE_24KHZ, get_decoder24_weights(), get_rvq24_weights());

    // encoder enc(RATE_48KHZ, get_encoder48_weights(), get_rvq48_weights());
    // decoder dec(RATE_48KHZ, get_decoder48_weights(), get_rvq48_weights());

    printf("Testing on birch canoe...\n");
    const char* input = "original.dat";
    test_birch_canoe(enc, dec, BPS_24000, input);
    test_birch_canoe(enc, dec, BPS_12000, input);
    test_birch_canoe(enc, dec, BPS_6000, input);
    test_birch_canoe(enc, dec, BPS_3000, input);
    test_birch_canoe(enc, dec, BPS_1500, input);
    printf("Testing on birch canoe... Done\n");
    
    // printf("Bench...\n");
    // bench(enc, dec, BPS_24000);
    // printf("Bench... Done\n");

    return 0;
}
