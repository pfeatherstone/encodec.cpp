#include <cassert>
#include <cstring>
#include <cmath>
#include <vector>
#include <limits>
#include <numeric>
#include <algorithm>
#include "encodec.h"
#include "incbin.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

//----------------------------------------------------------------------------------------------------------------

INCBIN(encoder, ENCODER_DATA);
INCBIN(rvq,     RVQ_DATA);

namespace encodec
{

//----------------------------------------------------------------------------------------------------------------

    const std::span encoder_weights{(const float*)gencoderData, gencoderSize/sizeof(float)};
    const std::span rvq_weights{(const float*)grvqData, grvqSize/sizeof(float)};

//----------------------------------------------------------------------------------------------------------------

    constexpr double    SAMPLE_RATE     = 24000;
    constexpr unsigned  STRIDE          = 320;
    constexpr unsigned  NLEVELS         = 32;
    constexpr unsigned  CODEBOOK_SIZE   = 1024;
    constexpr unsigned  CODEBOOK_DIM    = 128;

//----------------------------------------------------------------------------------------------------------------

    unsigned int get_encodec_bps(unsigned int nlevels)      { return (SAMPLE_RATE / STRIDE) * nlevels * 10; }
    unsigned int get_encoded_nquantizers(unsigned int bps)  { return (bps / 10) * STRIDE / SAMPLE_RATE; }

//----------------------------------------------------------------------------------------------------------------

    auto get_rvq_row(size_t level, size_t codebook_idx)
    {
        assert(NLEVELS*CODEBOOK_SIZE*CODEBOOK_DIM == rvq_weights.size());
        assert(level        < NLEVELS);
        assert(codebook_idx < CODEBOOK_SIZE);
        const size_t index = (level*CODEBOOK_SIZE + codebook_idx)*CODEBOOK_DIM;
        return rvq_weights.subspan(index, CODEBOOK_DIM);
    }

//----------------------------------------------------------------------------------------------------------------

    constexpr void subtract(std::span<const float> a, std::span<const float> b, std::span<float> c)
    {
        assert(a.size()==b.size() && a.size()==c.size());

        for (size_t i{0} ; i < a.size() ; ++i)
            c[i] = a[i] - b[i];
    }

//----------------------------------------------------------------------------------------------------------------

    constexpr void add(std::span<const float> a, std::span<const float> b, std::span<float> c)
    {
        assert(a.size()==b.size() && a.size()==c.size());

        for (size_t i{0} ; i < a.size() ; ++i)
            c[i] = a[i] + b[i];
    }
    
//----------------------------------------------------------------------------------------------------------------

#if defined(__AVX2__)
    inline static float dot(const float* a, const float* b, const size_t len)
    {
        const size_t len0 = (len/32)*32;
        const size_t len1 = (len/8)*8;

        __m256 dotProdVal0 = _mm256_setzero_ps();
        __m256 dotProdVal1 = _mm256_setzero_ps();
        __m256 dotProdVal2 = _mm256_setzero_ps();
        __m256 dotProdVal3 = _mm256_setzero_ps();

        size_t i{0};
        for (; i < len0; i += 32) 
        {
            const __m256 a0Val = _mm256_loadu_ps(&a[i+0]);
            const __m256 a1Val = _mm256_loadu_ps(&a[i+8]);
            const __m256 a2Val = _mm256_loadu_ps(&a[i+16]);
            const __m256 a3Val = _mm256_loadu_ps(&a[i+24]);
            const __m256 b0Val = _mm256_loadu_ps(&b[i+0]);
            const __m256 b1Val = _mm256_loadu_ps(&b[i+8]);
            const __m256 b2Val = _mm256_loadu_ps(&b[i+16]);
            const __m256 b3Val = _mm256_loadu_ps(&b[i+24]);

            dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
            dotProdVal1 = _mm256_fmadd_ps(a1Val, b1Val, dotProdVal1);
            dotProdVal2 = _mm256_fmadd_ps(a2Val, b2Val, dotProdVal2);
            dotProdVal3 = _mm256_fmadd_ps(a3Val, b3Val, dotProdVal3);
        }

        for (; i < len1; i += 8) 
        {
            const __m256 a0Val = _mm256_loadu_ps(&a[i]);
            const __m256 b0Val = _mm256_loadu_ps(&b[i]);
            dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
        }

        dotProdVal0 = _mm256_add_ps(_mm256_add_ps(dotProdVal0, dotProdVal1),
                                    _mm256_add_ps(dotProdVal2, dotProdVal3));
        
        alignas(32) float dotProductVector[8];
        _mm256_store_ps(dotProductVector, dotProdVal0); // Store the results back into the dot product vector

        float ret = dotProductVector[0]
                    + dotProductVector[1]
                    + dotProductVector[2]
                    + dotProductVector[3]
                    + dotProductVector[4]
                    + dotProductVector[5]
                    + dotProductVector[6]
                    + dotProductVector[7];

        for (; i < len; ++i)
            ret += a[i]*b[i];

        return ret;
    }
#else
    constexpr float dot(const float* a, const float* b, const size_t len)
    {
        return std::inner_product(a, a+len, b, 0.0f);
    }
#endif

//----------------------------------------------------------------------------------------------------------------

    constexpr float squared_dist(std::span<const float> a, std::span<const float> b)
    {
        assert(a.size()==b.size());

        float val{0};
        for (size_t i = 0 ; i < a.size() ; ++i)
        {
            const float tmp = a[i]-b[i];
            val += tmp*tmp;
        }
        return val;
    } 

//----------------------------------------------------------------------------------------------------------------

    constexpr void elu_(std::span<const float> in, std::span<float> out, const float alpha=1.0)
    {
        for (size_t i{0} ; i < in.size() ; ++i)
            out[i] = in[i] > 0 ? in[i] : alpha*(std::exp(in[i])-1);
    }

    constexpr auto elu(const float alpha=1.0) 
    {
        return [=](std::span<float> buf) {
            elu_(buf, buf, alpha);
        };
    }

//----------------------------------------------------------------------------------------------------------------

    constexpr auto identity = [](auto buf) {/*no-op*/};

//----------------------------------------------------------------------------------------------------------------

    constexpr void rvq_encode(std::span<float> feats, std::span<uint16_t> codes, unsigned int nlevels)
    {
        const size_t nfeats = feats.size() / CODEBOOK_DIM;

        for (size_t i{0} ; i < nfeats ; ++i)
        {
            auto x = feats.subspan(i*CODEBOOK_DIM, CODEBOOK_DIM);

            for (size_t l{0} ; l < nlevels ; ++l)
            {
                // Find best codebook
                float                  min_dist{std::numeric_limits<float>::max()};
                size_t                 codebook_idx{0};
                std::span<const float> codebook_best{};

                for (size_t c{0} ; c < CODEBOOK_SIZE ; ++c)
                {
                    const auto codebook = get_rvq_row(l, c);
                    const auto dist     = squared_dist(x, codebook);

                    if (dist < min_dist)
                    {
                        min_dist        = dist;
                        codebook_idx    = c;
                        codebook_best   = codebook;
                    }
                }

                codes[i*nlevels + l] = codebook_idx;

                // Update residual
                subtract(x, codebook_best, x);
            }
        }
    }

    constexpr void rvq_decode(std::span<const uint16_t> codes, std::span<float> feats, unsigned int nlevels)
    {
        const size_t nfeats = feats.size() / CODEBOOK_DIM;

        for (size_t i{0} ; i < nfeats ; ++i)
        {
            auto x = feats.subspan(i*CODEBOOK_DIM, CODEBOOK_DIM);

            for (size_t l{0} ; l < nlevels ; ++l)
            {
                const auto codebook = get_rvq_row(l, codes[i*nlevels+l]);
                add(x, codebook, x);
            }
        }
    }

//----------------------------------------------------------------------------------------------------------------

    constexpr void pack_codes(std::span<const uint16_t> codes, std::span<uint8_t> bytes)
    {
        const size_t nblocks = codes.size()/4; // 40 bits blocks
        const size_t rem     = codes.size()%4;

        for (size_t b{0} ; b < nblocks ; ++b)
        {
            const uint64_t word = uint64_t(codes[b*4+0]) |
                                 (uint64_t(codes[b*4+1]) << 10) | 
                                 (uint64_t(codes[b*4+2]) << 20) |
                                 (uint64_t(codes[b*4+3]) << 30);
            bytes[b*5+0] =  word        & 0xffull;
            bytes[b*5+1] = (word >>  8) & 0xffull;
            bytes[b*5+2] = (word >> 16) & 0xffull;
            bytes[b*5+3] = (word >> 24) & 0xffull;
            bytes[b*5+4] = (word >> 32) & 0xffull;
        }
        
        if (rem > 0)
        {
            uint64_t word{0};
            for (size_t k{0} ; k < rem ; ++k)
                word |= uint64_t(codes[nblocks*4+k]) << (10 * k);
            
            const size_t tail_bytes = (rem * 10 + 7) / 8;
            for (size_t j{0}; j < tail_bytes; ++j)
                bytes[nblocks*5+j] = uint8_t(word >> (8*j));
        }
    }

    constexpr void unpack_bits(std::span<const uint8_t> bytes, std::span<uint16_t> codes)
    {
        const size_t nblocks = bytes.size()/5; // 40 bits blocks
        const size_t rem     = bytes.size()%5;

        for (size_t b{0} ; b < nblocks ; ++b)
        {
            const uint64_t word = uint64_t(bytes[b*5+0])        |
                                 (uint64_t(bytes[b*5+1]) << 8)  |
                                 (uint64_t(bytes[b*5+2]) << 16) |
                                 (uint64_t(bytes[b*5+3]) << 24) |
                                 (uint64_t(bytes[b*5+4]) << 32);
            codes[b*4+0] =  word        & 0x3ffull;
            codes[b*4+1] = (word >> 10) & 0x3ffull;
            codes[b*4+2] = (word >> 20) & 0x3ffull;
            codes[b*4+3] = (word >> 30) & 0x3ffull;
        }

        if (rem > 0)
        {
            uint64_t word{0};
            for (size_t j{0} ; j < rem ; ++j)
                word |= uint64_t(bytes[nblocks*5+j]) << (j*8);

            const size_t tail_codes = (rem*8) / 10;
            for (size_t k{0}; k < tail_codes; ++k)
                codes[nblocks*4+k] = (word >> (10 * k)) & 0x3ffull;
        }
    }

//----------------------------------------------------------------------------------------------------------------

    struct conv
    {
        size_t              nin{};
        size_t              nout{};
        size_t              k{};
        size_t              s{};
        size_t              d{1};
        size_t              pad() {return d * (k - 1) + 1 - s;}
        std::vector<float>  weights{};      // shape [nout,k,nin]]
        std::vector<float>  bias{};         // shape [nout]
        std::vector<float>  scratch_in;     // shape [pad+nin]
        std::vector<float>  scratch_out;    // shape [T//s, nout]

        conv(size_t nin_, size_t nout_, size_t k_, size_t s_=1)
        : nin{nin_}, 
          nout{nout_}, 
          k{k_}, 
          s{s_}, 
          weights(nout*nin*k),
          bias(nout)
        {}

        auto load_state_dict(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < (weights.size() + bias.size())) throw std::runtime_error("Not enough data in weights");
            const auto w = data.subspan(0,              weights.size());
            const auto b = data.subspan(weights.size(), bias.size());
            std::copy(begin(w), end(w), begin(weights));
            std::copy(begin(b), end(b), begin(bias));
            return data.subspan(weights.size()+bias.size());
        }

        template<class PreAct>
        std::span<float> operator()(std::span<const float> input, PreAct fn)
        {
            // input  shape [T,nin]
            // output shape [T//s,nout]
            const size_t p    = pad();
            const size_t Tin  = input.size() / nin;
            const size_t Tinp = Tin+p;
            const size_t Tout = (Tinp - k) / s + 1;
            scratch_in.resize(Tinp*nin);
            scratch_out.resize(Tout*nout);

            // Pad
            for (size_t i{0} ; i < p ; ++i)
            {
                const auto i0 = (p-i)*nin;
                const auto i1 = i0+nin;
                auto       o0 = i*nin;
                std::copy(begin(input)+i0, begin(input)+i1, begin(scratch_in)+o0);
            }
            std::copy(begin(input), end(input), begin(scratch_in)+p*nin);

            // Pre-act
            fn(scratch_in);

            // Conv
            size_t i{0};
            size_t j{0};

            for (; (i+k) <= Tinp ; i+=s, ++j)
                for (size_t c{0} ; c < nout ; ++c)
                    scratch_out[j*nout+c] = dot(&scratch_in[i*nin], &weights[c*k*nin], k*nin) + bias[c];            

            assert((j*nout)<=scratch_out.size());
            return scratch_out;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct resnet_block
    {
        conv b0;
        conv b1;
        conv b2;

        resnet_block(size_t c)
        : b0(c,   c/2, 3),
          b1(c/2, c,   1),
          b2(c,   c,   1)
        {}

        auto load_state_dict(std::span<const float> data) -> std::span<const float>
        {
            data = b0.load_state_dict(data);
            data = b1.load_state_dict(data);
            data = b2.load_state_dict(data);
            return data;
        }

        std::span<float> operator()(std::span<const float> input)
        {
            auto x = b0(input, elu());
            x      = b1(x,     elu());
            auto y = b2(input, identity);
            add(x, y, y);
            return y;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct encoder_block
    {
        resnet_block b0;
        conv         b1;

        encoder_block(size_t c1, size_t c2, size_t s)
        : b0(c1),
          b1(c1, c2, s*2, s)
        {}

        auto load_state_dict(std::span<const float> data) -> std::span<const float>
        {
            data = b0.load_state_dict(data);
            data = b1.load_state_dict(data);
            return data;
        }

        std::span<float> operator()(std::span<const float> input)
        {
            auto x = b0(input);
            x      = b1(x, elu());
            return x;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct encoder::impl
    {
        conv          b0;
        encoder_block b1;
        encoder_block b2;
        encoder_block b3;
        encoder_block b4;

        impl()
        : b0(  1,  32, 7),
          b1( 32,  64, 2),
          b2( 64, 128, 4),
          b3(128, 256, 5),
          b4(256, 512, 8)
        
        {
            auto weights = encoder_weights;
            weights = b0.load_state_dict(weights);
            weights = b1.load_state_dict(weights);
            weights = b2.load_state_dict(weights);
            weights = b3.load_state_dict(weights);
            weights = b4.load_state_dict(weights);
            printf("left over weights %zu\n", weights.size());
        }

        std::span<const float> encode(std::span<const float> audio, unsigned int num_quantizers)
        {
            assert(num_quantizers >= 1 && num_quantizers <= NLEVELS);

            auto x = b0(audio, identity);
            x      = b1(x);
            x      = b2(x);
            x      = b3(x);
            x      = b4(x);
            return x;

            // // Run RVQ
            // codes.resize(nfeats*num_quantizers);
            // rvq_encode(feats, codes, num_quantizers);

            // // Pack codes
            // buf.resize((codes.size()*10 + 7) / 8);
            // pack_codes(codes, buf);
            // return buf;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct decoder::impl
    {
        // Ort::Session                        model;
        // Ort::AllocatorWithDefaultOptions    alloc;
        // Ort::Value                          output{nullptr};
        std::vector<uint16_t>               codes;
        std::vector<float>                  feats;

        // impl() : model(default_env(), gdecoder_onnxData, gdecoder_onnxSize, create_default_options()) {}
        
        std::span<const float> decode(std::span<const uint8_t> packet, unsigned int num_quantizers)
        {
            assert(num_quantizers >= 1 && num_quantizers <= NLEVELS);

            // Unpack bits
            const size_t ncodes = (packet.size()*8)/10;
            const size_t nfeats = ncodes/num_quantizers;
            assert(ncodes % num_quantizers == 0);
            codes.resize(ncodes);
            unpack_bits(packet, codes);

            // RVQ
            feats.resize(nfeats*CODEBOOK_DIM);
            memset(&feats[0], 0, feats.size()*sizeof(float));
            rvq_decode(codes, feats, num_quantizers);

            // // Decode
            // const char* input_names[]   = {"feats"};
            // const char* output_names[]  = {"audio"};
            // const int64_t shape[]       = {1,(int64_t)nfeats, (int64_t)CODEBOOK_DIM};
            // const auto meminfo          = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            // Ort::Value inputs[]         = {Ort::Value::CreateTensor<float>(meminfo, feats.data(), feats.size(), shape, 3)};

            // // Run encoder
            // output = Ort::Value(nullptr);
            // model.Run(Ort::RunOptions{nullptr}, input_names, inputs, 1, output_names, &output, 1);
            // const auto   shape2  = output.GetTensorTypeAndShapeInfo().GetShape();
            // const size_t naudio  = shape2[2];
            // const float* audio   = output.GetTensorData<float>();
            // assert(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT == output.GetTensorTypeAndShapeInfo().GetElementType());

            // return std::span{audio, naudio};

            return {};
        }
    };
    
//----------------------------------------------------------------------------------------------------------------

    encoder::encoder() : state{std::make_unique<impl>()} {}
    encoder::~encoder()                          = default;
    encoder::encoder(encoder&& other)            = default;
    encoder& encoder::operator=(encoder&& other) = default;

    std::span<const float> encoder::encode(std::span<const float> audio, unsigned int num_quantizers)
    {
        return state->encode(audio, num_quantizers);
    }

//----------------------------------------------------------------------------------------------------------------

    decoder::decoder() : state{std::make_unique<impl>()} {}
    decoder::~decoder()                          = default;
    decoder::decoder(decoder&& other)            = default;
    decoder& decoder::operator=(decoder&& other) = default;

    std::span<const float> decoder::decode(std::span<const uint8_t> packet, unsigned int num_quantizers)
    {
        return state->decode(packet, num_quantizers);
    }

//----------------------------------------------------------------------------------------------------------------

}