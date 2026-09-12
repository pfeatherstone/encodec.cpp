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
        encoder enc(get_encoder24_weights(), get_rvq24_weights());
        decoder dec(get_decoder24_weights(), get_rvq24_weights());
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
            const size_t num_quants = get_encodec_nquantizers(RATE_24KHZ, bps);

            // Warmup
            auto packet = enc.encode(audio, num_quants);
            auto audio2 = dec.decode(packet, num_quants);
            packet_buf.assign(begin(packet), end(packet));
            (void)packet;
            (void)audio2;

            bench.run(format("encode 24khz : bps ", bps, " quants ", num_quants), [&] {
                enc.encode(audio, num_quants);
            });

            bench.run(format("decode 24khz : bps ", bps, " quants ", num_quants), [&] {
                dec.decode(packet_buf, num_quants);
            });
        }
    }
}
