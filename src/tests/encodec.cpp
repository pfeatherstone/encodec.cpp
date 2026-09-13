#include <vector>
#include <algorithm>
#include <random>
#include "doctest.h"
#include <encodec.h>

using namespace encodec;

static std::mt19937_64 RAND;

struct test_data
{
    const char* audio0;
    const char* feats;
    const char* audio1;
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
    {ENCODEC_DATA_PATH "/encodec_48khz_orig_24000.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_feats_24000.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_decod_24000.dat"},
    {ENCODEC_DATA_PATH "/encodec_48khz_orig_33333.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_feats_33333.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_decod_33333.dat"},
    {ENCODEC_DATA_PATH "/encodec_48khz_orig_48000.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_feats_48000.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_decod_48000.dat"},
    {ENCODEC_DATA_PATH "/encodec_48khz_orig_9999.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_feats_9999.dat",
     ENCODEC_DATA_PATH "/encodec_48khz_decod_9999.dat"}
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

TEST_SUITE("[ENCODEC]") 
{
    TEST_CASE_TEMPLATE("test encodec against pytorch", T, test_details<RATE_24KHZ>, test_details<RATE_48KHZ>) 
    {
        encoder enc(T::rate, T::encoder_weights(), T::rvq_weights());
        decoder dec(T::rate, T::decoder_weights(), T::rvq_weights());

        for (const auto& [file_orig, file_feats, file_decod] : T::test_datas())
        {
            std::vector<float> audio0       = load_file<float>(file_orig);
            std::vector<float> feats_exp    = load_file<float>(file_feats);
            std::vector<float> audio1_exp   = load_file<float>(file_decod);

            auto feats_cal = enc.features(audio0);
            REQUIRE(feats_cal.size() == feats_exp.size());
            for (size_t i{0} ; i < feats_cal.size() ; ++i)
                CHECK(std::abs(feats_cal[i] - feats_exp[i]) < 1e-4);

            auto audio1_cal = dec.audio(feats_exp);
            REQUIRE(audio1_cal.size() == audio1_exp.size());
            for (size_t i{0} ; i < audio1_cal.size() ; ++i)
                CHECK(std::abs(audio1_cal[i] - audio1_exp[i]) < 5e-4);
        }
    }
}
