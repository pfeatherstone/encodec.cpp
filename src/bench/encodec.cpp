#include <algorithm>
#include <random>
#include <sstream>
#include "../tests/doctest.h"
#include "nanobench.h"
#include <encodec.h>

using namespace std::chrono_literals;
using std::begin;
using std::end;
using namespace encodec;

static std::mt19937_64 RAND;

auto format(auto... args)
{
    std::stringstream ss;
    (ss << ... << args);
    return ss.str();
}

TEST_SUITE("[ENCODEC]") 
{
    TEST_CASE("encodec24khz") 
    {
        // Encodec
        encoder enc(RATE_24KHZ, get_encoder24_weights(), get_rvq24_weights());
        decoder dec(RATE_24KHZ, get_decoder24_weights(), get_rvq24_weights());
        constexpr bitrates BPS[] = {BPS_24000, BPS_12000, BPS_6000, BPS_3000};
    
        // Audio
        float audio[24000];
        std::generate(begin(audio), end(audio), [&]{return std::normal_distribution<float>{}(RAND);});

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

            bench.run(format("encode 24khz : bps ", bps), [&] {
                enc.encode(audio, bps);
            });

            bench.run(format("decode 24khz : bps ", bps), [&] {
                dec.decode(packet_buf, scale, bps);
            });
        }
    }
}
