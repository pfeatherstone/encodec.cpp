#include <vector>
#include <algorithm>
#include <random>
#include "doctest.h"
#include <encodec.h>

using namespace encodec;

static std::mt19937_64 RAND;

TEST_SUITE("[ENCODEC]") 
{
    TEST_CASE("sizes") 
    {
        constexpr bitrates BPS[] = {BPS_24000, BPS_12000, BPS_6000, BPS_3000};
        encoder enc(get_encoder24_weights(), get_rvq24_weights());
        decoder dec(get_decoder24_weights(), get_rvq24_weights());

        for (size_t b{70} ; b < 75 ; ++b)
        {
            std::vector<float> audio(b*320);
            std::generate(begin(audio), end(audio), [&]{return std::normal_distribution<float>{}(RAND);});

            for (auto bps : BPS)
            {
                auto packet = enc.encode(audio,  get_encodec_nquantizers(RATE_24KHZ, bps));
                auto audio2 = dec.decode(packet, get_encodec_nquantizers(RATE_24KHZ, bps));
                REQUIRE(audio2.size()==audio.size());
            }
        }
    }
}
