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

    class encoder
    {
    private:
        struct impl;
        std::unique_ptr<impl> state;
        
    public:
        encoder();
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
        decoder();
        ~decoder();
        decoder(decoder&& other);
        decoder& operator=(decoder&& other);

        std::span<const float> decode(std::span<const uint8_t> packet, unsigned int num_quantizers);
    };

//----------------------------------------------------------------------------------------------------------------

}