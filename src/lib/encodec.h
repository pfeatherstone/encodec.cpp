#pragma once

#include <cstdint>
#include <span>
#include <memory>

namespace encodec
{

//----------------------------------------------------------------------------------------------------------------

    enum sample_rates : unsigned int
    {
        RATE_24KHZ = 24000,
        RATE_48KHZ = 48000
    };

    enum bitrates : unsigned int
    {
        BPS_1500  = 1500,
        BPS_3000  = 3000,
        BPS_6000  = 6000,
        BPS_12000 = 12000,
        BPS_24000 = 24000
    };

//----------------------------------------------------------------------------------------------------------------

    std::span<const float> get_encoder24_weights();
    std::span<const float> get_decoder24_weights();
    std::span<const float> get_rvq24_weights();

//----------------------------------------------------------------------------------------------------------------

    std::span<const float> get_encoder48_weights();
    std::span<const float> get_decoder48_weights();
    std::span<const float> get_rvq48_weights();

//----------------------------------------------------------------------------------------------------------------

    class encoder
    {
    private:
        struct impl;
        std::unique_ptr<impl> state;
        
    public:
        encoder(sample_rates rate, std::span<const float> encoder_weights, std::span<const float> rvq_weights);
        ~encoder();
        encoder(encoder&& other);
        encoder& operator=(encoder&& other);

        std::span<float>            features(std::span<const float>    audio);
        std::span<const uint16_t>   codes   (std::span<float>          feats, bitrates bps);
        std::span<const uint8_t>    packet  (std::span<const uint16_t> codes);
        std::span<const uint8_t>    encode  (std::span<const float>    audio, bitrates bps);
    };

//----------------------------------------------------------------------------------------------------------------

    class decoder
    {
    private:
        struct impl;
        std::unique_ptr<impl> state;
        
    public:
        decoder(sample_rates rate, std::span<const float> decoder_weights, std::span<const float> rvq_weights);
        ~decoder();
        decoder(decoder&& other);
        decoder& operator=(decoder&& other);

        std::span<const uint16_t> codes   (std::span<const uint8_t> packet);
        std::span<const float>    features(std::span<const uint16_t> codes, bitrates bps);
        std::span<const float>    audio   (std::span<const float> features);
        std::span<const float>    decode  (std::span<const uint8_t> packet, bitrates bps);
    };

//----------------------------------------------------------------------------------------------------------------

}
