#pragma once

#include <cstdint>
#include <span>
#include <memory>

namespace encodec
{

//----------------------------------------------------------------------------------------------------------------

    unsigned int get_encodec_bps(unsigned int num_quantizers);
    unsigned int get_encoded_nquantizers(unsigned int bps);

//----------------------------------------------------------------------------------------------------------------

    std::span<const float> get_encoder24_weights();
    std::span<const float> get_decoder24_weights();
    std::span<const float> get_rvq24_weights();

//----------------------------------------------------------------------------------------------------------------

    class encoder
    {
    private:
        struct impl;
        std::unique_ptr<impl> state;
        
    public:
        encoder(std::span<const float> encoder_weights, std::span<const float> rvq_weights);
        ~encoder();
        encoder(encoder&& other);
        encoder& operator=(encoder&& other);

        std::span<const uint8_t> encode(std::span<const float> audio, unsigned int num_quantizers);
    };

//----------------------------------------------------------------------------------------------------------------

    class decoder
    {
    private:
        struct impl;
        std::unique_ptr<impl> state;
        
    public:
        decoder(std::span<const float> decocer_weights, std::span<const float> rvq_weights);
        ~decoder();
        decoder(decoder&& other);
        decoder& operator=(decoder&& other);

        std::span<const float> decode(std::span<const uint8_t> packet, unsigned int num_quantizers);
    };

//----------------------------------------------------------------------------------------------------------------

}
