#include <vector>
#include <algorithm>
#include <random>
#include "doctest.h"
#include "encodec.h"

static std::mt19937_64 RAND;

TEST_SUITE("[ENCODEC]") 
{
    TEST_CASE("sizes") 
    {
        constexpr size_t BPS[] = {24000, 12000, 6000, 3000};
        encodec::encoder enc;
        encodec::decoder dec;

        for (size_t b{70} ; b < 75 ; ++b)
        {
            std::vector<float> audio(b*320);
            std::generate(begin(audio), end(audio), [&]{return std::normal_distribution<float>{}(RAND);});

            for (auto bps : BPS)
            {
                auto packet = enc.encode(audio,  encodec::get_encoded_nquantizers(bps));
                auto audio2 = dec.decode(packet, encodec::get_encoded_nquantizers(bps));
                REQUIRE(audio2.size()==audio.size());
            }
        }
    }
}