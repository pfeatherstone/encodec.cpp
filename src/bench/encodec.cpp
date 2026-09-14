#include <algorithm>
#include <random>
#include <sstream>
#include <vector>
#include "../tests/doctest.h"
#include "nanobench.h"
#include <encodec.h>

using namespace std::chrono_literals;
using std::begin;
using std::end;
using namespace encodec;

static std::mt19937_64 RAND;

template<sample_rates rate>
struct test_details{};

template<>
struct test_details<RATE_24KHZ>
{
    static constexpr sample_rates       rate    = RATE_24KHZ;
    static constexpr std::size_t        nc      = 1;
    static std::span<const float>       encoder_weights() {return get_encoder24_weights();}
    static std::span<const float>       decoder_weights() {return get_decoder24_weights();}
    static std::span<const float>       rvq_weights    () {return get_rvq24_weights();}
    static constexpr std::string_view   test_audio = ENCODEC_DATA_PATH "/encodec_24khz_orig_24000.dat";
};

template<>
struct test_details<RATE_48KHZ>
{
    static constexpr sample_rates       rate    = RATE_48KHZ;
    static constexpr std::size_t        nc      = 2;
    static std::span<const float>       encoder_weights() {return get_encoder48_weights();}
    static std::span<const float>       decoder_weights() {return get_decoder48_weights();}
    static std::span<const float>       rvq_weights    () {return get_rvq48_weights();}
    static constexpr std::string_view   test_audio = ENCODEC_DATA_PATH "/encodec_48khz_orig_24000.dat";
};

auto format(auto... args)
{
    std::stringstream ss;
    (ss << ... << args);
    return ss.str();
}

template<class T>
auto load_file(std::string_view file)
{
    std::vector<T> data;
    FILE* fp = fopen(file.data(), "rb");
    if (!fp)
    {
        printf("Failed to open `%s`\n", file.data());
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

TEST_SUITE("[ENCODEC]") 
{
    TEST_CASE_TEMPLATE("features", T, test_details<RATE_24KHZ>, test_details<RATE_48KHZ>) 
    {
        encoder enc(T::rate, T::encoder_weights(), T::rvq_weights());
        decoder dec(T::rate, T::decoder_weights(), T::rvq_weights());
    
        // Audio
        float audio0[24000*T::nc];
        std::generate(begin(audio0), end(audio0), [&]{return std::normal_distribution<float>{}(RAND);});

        // Bench
        ankerl::nanobench::Bench bench;
        bench.minEpochTime(3s).epochs(3);

        std::vector<float> features_buf;

        // Warmup
        auto feats  = enc.features(audio0);
        auto audio2 = dec.audio(feats);

        features_buf.assign(begin(feats), end(feats));
        (void)feats;
        (void)audio2;

        bench.run(format("encode to feats ", T::rate, " hz"), [&] {
            enc.features(audio0);
        });

        bench.run(format("decode from feats ", T::rate, " hz"), [&] {
            dec.audio(features_buf);
        });
    }

    TEST_CASE_TEMPLATE("encode - decode", T, test_details<RATE_24KHZ>, test_details<RATE_48KHZ>) 
    {
        encoder enc(T::rate, T::encoder_weights(), T::rvq_weights());
        decoder dec(T::rate, T::decoder_weights(), T::rvq_weights());
        constexpr bitrates BPS[] = {BPS_24000, BPS_12000, BPS_6000, BPS_3000};
    
        // Audio
        const std::vector<float> audio = load_file<float>(T::test_audio);

        // Bench
        ankerl::nanobench::Bench bench;
        bench.minEpochTime(3s).epochs(3);

        std::vector<uint8_t> packet_buf;

        for (auto bps : BPS)
        {
            // Warmup
            auto result = enc.encode(audio, bps);
            auto scale  = result.first;
            auto packet = result.second;
            auto audio2 = dec.decode(packet, scale, bps);

            packet_buf.assign(begin(packet), end(packet));
            (void)packet;
            (void)audio2;

            bench.run(format("encode ", T::rate, " hz - bps ", bps, " nsamples ", audio.size()/T::nc), [&] {
                enc.encode(audio, bps);
            });

            bench.run(format("decode ", T::rate, " hz - bps ", bps, " nsamples ", audio.size()/T::nc), [&] {
                dec.decode(packet_buf, scale, bps);
            });
        }
    }
}
