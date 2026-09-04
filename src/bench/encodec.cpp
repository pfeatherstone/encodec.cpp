#include <algorithm>
#include <random>
#include "../tests/doctest.h"
#include "nanobench.h"
#include "format.h"
#include <encodec.h>
#include <encodec_encoder24.h>
#include <encodec_decoder24.h>
#include <encodec_rvq24.h>

using namespace std::chrono_literals;
using std::begin;
using std::end;

static std::mt19937_64 RAND;

TEST_SUITE("[ENCODEC]") 
{
    TEST_CASE("encodec24khz") 
    {
        // Encodec
        encodec::encoder enc(encodec::get_encoder24_weights(), encodec::get_rvq24_weights());
        encodec::decoder dec(encodec::get_decoder24_weights(), encodec::get_rvq24_weights());
        constexpr size_t BPS[] = {24000, 12000, 6000, 3000};
    
        // Audio
        float audio[24000];
        std::generate(begin(audio), end(audio), [&]{return std::normal_distribution<float>{}(RAND);});

        // Bench
        ankerl::nanobench::Bench bench;
        bench.minEpochTime(3s).epochs(3);

        std::vector<uint8_t> packet_buf;

        for (auto bps : BPS)
        {
            const size_t num_quants = encodec::get_encoded_nquantizers(bps);

            // Warmup
            auto packet = enc.encode(audio, num_quants);
            auto audio2 = dec.decode(packet, num_quants);
            packet_buf.assign(begin(packet), end(packet));
            (void)packet;
            (void)audio2;

            bench.run(format("encode 24khz %zu bps %zu quants", bps, num_quants), [&] {
                enc.encode(audio, num_quants);
            });

            bench.run(format("decode 24khz %zu bps %zu quants", bps, num_quants), [&] {
                dec.decode(packet_buf, num_quants);
            });
        }
    }
}