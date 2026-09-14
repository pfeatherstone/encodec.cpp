#include <vector>
#include <algorithm>
#include <random>
#include <ostream>
#include <string_view>
#include "doctest.h"
#include <encodec.h>

using namespace encodec;

static std::mt19937_64 RAND;

struct test_data
{
    std::string_view audio0;
    std::string_view feats;
    std::string_view audio1;
};

constexpr test_data TEST_DATA_24[] = {
    {ENCODEC_DATA_PATH "/encodec_24khz_orig_24000.dat",
     ENCODEC_DATA_PATH "/encodec_24khz_feats_24000.dat",
     ENCODEC_DATA_PATH "/encodec_24khz_decod_24000.dat"},
    {ENCODEC_DATA_PATH "/encodec_24khz_orig_33333.dat",
     ENCODEC_DATA_PATH "/encodec_24khz_feats_33333.dat",
     ENCODEC_DATA_PATH "/encodec_24khz_decod_33333.dat"},
    {ENCODEC_DATA_PATH "/encodec_24khz_orig_48000.dat",
     ENCODEC_DATA_PATH "/encodec_24khz_feats_48000.dat",
     ENCODEC_DATA_PATH "/encodec_24khz_decod_48000.dat"},
    {ENCODEC_DATA_PATH "/encodec_24khz_orig_9999.dat",
     ENCODEC_DATA_PATH "/encodec_24khz_feats_9999.dat",
     ENCODEC_DATA_PATH "/encodec_24khz_decod_9999.dat"}
};

constexpr test_data TEST_DATA_48[] = {
    {ENCODEC_DATA_PATH "/encodec_48khz_orig_37000.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_feats_37000.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_decod_37000.dat"},
    {ENCODEC_DATA_PATH "/encodec_48khz_orig_47589.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_feats_47589.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_decod_47589.dat"},
    {ENCODEC_DATA_PATH "/encodec_48khz_orig_48000.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_feats_48000.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_decod_48000.dat"},
    {ENCODEC_DATA_PATH "/encodec_48khz_orig_55555.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_feats_55555.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_decod_55555.dat"}
};

template<sample_rates rate>
struct test_details{};

template<>
struct test_details<RATE_24KHZ>
{
    static constexpr sample_rates rate = RATE_24KHZ;
    static std::span<const float>      encoder_weights() {return get_encoder24_weights();}
    static std::span<const float>      decoder_weights() {return get_decoder24_weights();}
    static std::span<const float>      rvq_weights    () {return get_rvq24_weights();}
    static std::span<const test_data>  test_datas     () {return TEST_DATA_24;}
};

template<>
struct test_details<RATE_48KHZ>
{
    static constexpr sample_rates rate = RATE_48KHZ;
    static std::span<const float>      encoder_weights() {return get_encoder48_weights();}
    static std::span<const float>      decoder_weights() {return get_decoder48_weights();}
    static std::span<const float>      rvq_weights    () {return get_rvq48_weights();}
    static std::span<const test_data>  test_datas     () {return TEST_DATA_48;}
};

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

constexpr auto max_diff(std::span<const float> a, std::span<const float> b)
{
    float diff{-1.f};
    size_t index{0};

    for (size_t i{0} ; i < a.size() ; ++i)
    {
        const auto d = std::abs(a[i]-b[i]);

        if (d > diff)
        {
            diff  = d;
            index = i;
        }
    }

    return std::make_pair(index, diff);
}

TEST_SUITE("[ENCODEC]") 
{
    TEST_CASE_TEMPLATE("test encodec against pytorch", T, test_details<RATE_24KHZ>, test_details<RATE_48KHZ>) 
    {
        encoder enc(T::rate, T::encoder_weights(), T::rvq_weights());
        decoder dec(T::rate, T::decoder_weights(), T::rvq_weights());

        for (const auto& [file_orig, file_feats, file_decod] : T::test_datas())
        {
            const std::vector<float> audio0       = load_file<float>(file_orig);
            const std::vector<float> feats_exp    = load_file<float>(file_feats);
            const std::vector<float> audio1_exp   = load_file<float>(file_decod);

            const auto feats_cal = enc.features(audio0);
            REQUIRE(feats_cal.size() == feats_exp.size());
            const auto [i_feat, d_feat] = max_diff(feats_cal, feats_exp);
            CHECK(d_feat < 2.1e-4);

            const auto audio1_cal = dec.audio(feats_exp);
            REQUIRE(audio1_cal.size() == audio1_exp.size());
            const auto [i_aud, d_aud]  = max_diff(audio1_cal, audio1_exp);
            CHECK(d_aud < 7e-4);
        }
    }
}
