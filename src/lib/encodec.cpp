#include <cassert>
#include <cstring>
#include <cmath>
#include <vector>
#include <limits>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>
#include "encodec.h"
#include "incbin.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

//----------------------------------------------------------------------------------------------------------------

using MatrixXf = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
using VectorXf = Eigen::Vector<float, -1>;
using ArrayXf  = Eigen::Array<float, -1, 1>;

//----------------------------------------------------------------------------------------------------------------

INCBIN(encoder, ENCODER_DATA);
INCBIN(decoder, DECODER_DATA);
INCBIN(rvq,     RVQ_DATA);

//----------------------------------------------------------------------------------------------------------------

namespace encodec
{

//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
// MATH
//----------------------------------------------------------------------------------------------------------------
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

    inline float dot(const float* a, const float* b, const size_t len)
    {
        Eigen::Map<const VectorXf> ae(a, len);
        Eigen::Map<const VectorXf> be(b, len);
        return ae.dot(be);
    }

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
//----------------------------------------------------------------------------------------------------------------
// CONSTANTS
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
    
    constexpr double    SAMPLE_RATE     = 24000;
    constexpr unsigned  STRIDE          = 320;
    constexpr unsigned  NLEVELS         = 32;
    constexpr unsigned  CODEBOOK_SIZE   = 1024;
    constexpr unsigned  CODEBOOK_DIM    = 128;

    static const std::span encoder_weights{(const float*)gencoderData, gencoderSize/sizeof(float)};
    static const std::span decoder_weights{(const float*)gdecoderData, gdecoderSize/sizeof(float)};
    static const std::span rvq_weights    {(const float*)grvqData,     grvqSize/sizeof(float)};

//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
// RVQ
//----------------------------------------------------------------------------------------------------------------
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
//----------------------------------------------------------------------------------------------------------------
// ACTIVATIONS
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------

    template <class Derived>
    auto elu(const Eigen::ArrayBase<Derived>& x, float alpha = 1.0)
    {
        return (x > 0.0f).select(x, alpha * (x.exp() - 1.0f));
    }

    template <class Derived>
    auto sigmoid(const Eigen::ArrayBase<Derived>& x)
    {
        return (1.0f + (-x).exp()).inverse();
    }

//----------------------------------------------------------------------------------------------------------------

    struct elu_layer
    {
        bool                inplace{};
        float               alpha{};
        std::vector<float>  tmp;

        elu_layer(bool inplace_, float alpha_ = 1.0) : inplace{inplace_}, alpha{alpha_} {}

        std::span<float> operator()(std::span<float> input)
        {
            if (inplace)
            {
                auto x = Eigen::Map<ArrayXf>(input.data(), input.size());
                x = elu(x, alpha);
                return input;
            }
            else
            {
                tmp.resize(input.size());
                auto x = Eigen::Map<const ArrayXf>(input.data(), input.size());
                auto y = Eigen::Map<ArrayXf>(tmp.data(), tmp.size());
                y = elu(x, alpha);
                return tmp;
            }
        }
    };

//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
// NN
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------

    struct linear
    {
        MatrixXf w;
        VectorXf b;
        MatrixXf out;

        linear(size_t nin_, size_t nout_)
        : w(nout_, nin_),
          b(nout_)
        {
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < size_t(w.size() + b.size())) throw std::runtime_error("Not enough data in weights");
            size_t off{0};
            auto w_ = Eigen::Map<const MatrixXf>(data.subspan(off, w.size()).data(), nout(), nin()); off += w.size();
            auto b_ = Eigen::Map<const VectorXf>(data.subspan(off, b.size()).data(), nout());        off += b.size();
            w       = w_;
            b       = b_;
            return data.subspan(off);
        }

        size_t nin()  {return w.cols();}
        size_t nout() {return w.rows();}

        std::span<float> operator()(std::span<const float> input)
        {
            const size_t Tin = input.size() / nin();
            auto x = Eigen::Map<const MatrixXf>(input.data(), Tin, nin());
            out.noalias() = x * w.transpose() ;
            out.rowwise() += b.transpose();
            return std::span<float>{out.data(), static_cast<size_t>(out.size())};
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct conv
    {
        size_t              nin{};
        size_t              nout{};
        size_t              k{};
        size_t              s{};
        size_t              d{1};
        size_t              pad() {return d * (k - 1) + 1 - s;}
        MatrixXf            w;              // shape [nout,k*nin]
        VectorXf            b;              // shape [nout]
        std::vector<float>  scratch_in;     // shape [(p+k),nin]
        std::vector<float>  scratch_out;    // shape [T//s, nout]

        conv(size_t nin_, size_t nout_, size_t k_, size_t s_=1)
        : nin{nin_}, 
          nout{nout_}, 
          k{k_}, 
          s{s_}, 
          w(nout, k*nin),
          b(nout)
        {}

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < size_t(w.size() + b.size())) throw std::runtime_error("Not enough data in weights");
            size_t off{0};
            auto w_ = Eigen::Map<const MatrixXf>(data.subspan(off, w.size()).data(), nout, k*nin); off += w.size();
            auto b_ = Eigen::Map<const VectorXf>(data.subspan(off, b.size()).data(), nout);        off += b.size();
            w       = w_;
            b       = b_;
            return data.subspan(off);
        }

        std::span<float> operator()(std::span<const float> input)
        {
            // input  shape [T,nin]
            // output shape [T//s,nout]
            const size_t p    = pad();
            const size_t Tin  = input.size() / nin;
            const size_t Tinp = Tin+p;
            const size_t Tout = (Tinp - k) / s + 1;
            scratch_in.resize((p+k)*nin);
            scratch_out.resize(Tout*nout);

            // Padded buf
            for (size_t i{0} ; i < p ; ++i)
                std::copy_n(begin(input)+(p-i)*nin, nin, begin(scratch_in)+i*nin);
            std::copy_n(begin(input), k*nin, begin(scratch_in)+p*nin);

            size_t i{0};
            size_t j{0};

            // Conv in padded windows
            for (; i < p; i+=s, ++j)
            {
                auto x = Eigen::Map<const VectorXf>(&scratch_in[i*nin], k*nin); 
                auto y = Eigen::Map<VectorXf>(&scratch_out[j*nout],nout);
                y.noalias() = w*x + b;
            }
            // Conv in rest
            for (; (i+k) <= Tinp; i+=s, ++j)
            {
                auto x = Eigen::Map<const VectorXf>(&input[(i-p)*nin], k*nin); 
                auto y = Eigen::Map<VectorXf>(&scratch_out[j*nout],nout);
                y.noalias() = w*x + b;
            }

            assert((j*nout)<=scratch_out.size());
            return scratch_out;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct conv_transpose
    {
        size_t              nin{};
        size_t              nout{};
        size_t              k{};
        size_t              s{};
        size_t              d{1};
        size_t              pad() {return d * (k - 1) + 1 - s;}
        std::vector<float>  weights{};      // shape [nin,k,nout]
        std::vector<float>  bias{};         // shape [nout]
        std::vector<float>  scratch_in;     // shape [Tin,nin]
        std::vector<float>  scratch_out;    // shape [Tout_padded,nout]

        conv_transpose(size_t nin_, size_t nout_, size_t k_, size_t s_ = 1)
        : nin{nin_}, 
          nout{nout_}, 
          k{k_}, 
          s{s_}, 
          weights(nout*nin*k),
          bias(nout)
        {}

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < (weights.size() + bias.size())) throw std::runtime_error("Not enough data in weights");
            const auto w = data.subspan(0,              weights.size());
            const auto b = data.subspan(weights.size(), bias.size());
            std::copy(begin(w), end(w), begin(weights));
            std::copy(begin(b), end(b), begin(bias));
            return data.subspan(weights.size()+bias.size());
        }

        std::span<float> operator()(std::span<const float> input)
        {
            // input  shape [Tin,nin]
            // output shape [Tin*s,nout] for your EnCodec-style padding

            const size_t p           = pad();
            const size_t Tin         = input.size() / nin;
            const size_t Tout_padded = (Tin - 1) * s + d * (k - 1) + 1;
            const size_t Tout        = Tout_padded - p;

            // Pre-act
            scratch_in.assign(begin(input), end(input));
            
            // Scatter transposed convolution.
            // out_padded[t*s + kk*d, co] += x[t,ci] * weight[ci,kk,co]
            scratch_out.assign(Tout_padded * nout, 0.0f);

            for (size_t t{0}; t < Tin; ++t)
            {
                for (size_t ci{0}; ci < nin; ++ci)
                {
                    const float x = scratch_in[t * nin + ci];

                    for (size_t kk{0}; kk < k; ++kk)
                    {
                        const size_t to = t * s + kk * d;

                        const float* w = &weights[(ci * k + kk) * nout];
                        float*       y = &scratch_out[to * nout];

                        for (size_t co{0}; co < nout; ++co)
                            y[co] += x * w[co];
                    }
                }
            }

            // Add bias
            for (size_t t{0}; t < Tout; ++t)
                for (size_t co{0}; co < nout; ++co)
                    scratch_out[t * nout + co] += bias[co];

            auto out = std::span{scratch_out}.subspan(0, Tout*nout);
            return out;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct lstm_cell
    {
        MatrixXf wih;
        MatrixXf whh;
        VectorXf bias;
        VectorXf gates;
        VectorXf h;
        VectorXf c;
        MatrixXf xw;      // [T, 4H]
        MatrixXf out;     // [T, H]

        lstm_cell(size_t input_size, size_t hidden_size)
        : wih(4*hidden_size, input_size),
          whh(4*hidden_size, hidden_size),
          bias(4*hidden_size),
          gates(4*hidden_size),
          h(hidden_size),
          c(hidden_size)
        {
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            const size_t total_weights = wih.size()+whh.size()+bias.size()*2;
            if (data.size() < total_weights) throw std::runtime_error("Not enough data in weights");
            size_t off{0};
            auto wih_   = Eigen::Map<const MatrixXf>(data.subspan(off, wih.size()).data(), wih.rows(), wih.cols()); off += wih.size();
            auto whh_   = Eigen::Map<const MatrixXf>(data.subspan(off, whh.size()).data(), whh.rows(), whh.cols()); off += whh.size();
            auto bih_   = Eigen::Map<const VectorXf>(data.subspan(off, bias.size()).data(), bias.size());           off += bias.size();
            auto bhh_   = Eigen::Map<const VectorXf>(data.subspan(off, bias.size()).data(), bias.size());           off += bias.size();
            wih         = wih_;
            whh         = whh_;
            bias        = bih_ + bhh_;
            return data.subspan(off);
        }

        void apply_gates()
        {
            const size_t H = h.size();
            auto i = gates.segment(0 * H, H).array();
            auto f = gates.segment(1 * H, H).array();
            auto g = gates.segment(2 * H, H).array();
            auto o = gates.segment(3 * H, H).array();
            c.array() = sigmoid(f) * c.array()  + sigmoid(i) * g.tanh();
            h.array() = sigmoid(o) * c.array().tanh();
        }

        auto& operator()(const MatrixXf& X)
        {
            const size_t T = X.rows();
            const size_t H = whh.cols();
            out.resize(T, H);

            // Zero hidden state and cell state
            h.setZero();
            c.setZero();

            // Precompute input projection for all timesteps
            xw.resize(T, 4*H);
            xw.noalias() = X * wih.transpose();
            xw.rowwise() += bias.transpose();

            for (size_t t{0}; t < T; ++t)
            {
                // gates = xw[t] + whh * h
                gates.noalias() = xw.row(t).transpose() + whh * h;
                apply_gates();
                out.row(t) = h.transpose();
            }

            return out;
        }   
    };

//----------------------------------------------------------------------------------------------------------------

    struct encodec_lstm
    {
        lstm_cell   cells[2];
        MatrixXf    out;

        encodec_lstm(size_t dim)
        : cells{lstm_cell(dim, dim), lstm_cell(dim, dim)}
        {
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            data = cells[0].load_weights(data);
            data = cells[1].load_weights(data);
            return data;
        }

        std::span<float> operator()(std::span<const float> input)
        {
            const size_t C = cells[0].wih.cols();
            const size_t T = input.size() / C;
            out.resize(T, C);
            auto X = Eigen::Map<const MatrixXf>(input.data(), T, C);
            auto& Y = cells[0](X);
            auto& Z = cells[1](Y);
            out.noalias() = X + Z;
            return std::span{out.data(), (size_t)out.size()};
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct resnet_block
    {
        elu_layer   a0;
        elu_layer   a1;
        conv        b0;
        linear      b1;
        linear      b2;

        resnet_block(size_t c)
        : a0(false),
          a1(true),
          b0(c,   c/2, 3),
          b1(c/2, c),
          b2(c,   c)
        {}

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            data = b0.load_weights(data);
            data = b1.load_weights(data);
            data = b2.load_weights(data);
            return data;
        }

        std::span<float> operator()(std::span<float> input)
        {
            auto x = b1(a1(b0(a0(input))));
            auto y = b2(input);
            add(x, y, y);
            return y;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct encoder_block
    {
        resnet_block b0;
        elu_layer    a1;
        conv         b1;

        encoder_block(size_t c1, size_t c2, size_t s)
        : b0(c1),
          a1(true),
          b1(c1, c2, s*2, s)
        {}

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            data = b0.load_weights(data);
            data = b1.load_weights(data);
            return data;
        }

        std::span<float> operator()(std::span<float> input)
        {
            return b1(a1(b0(input)));
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct decoder_block
    {
        conv_transpose  b0;
        resnet_block    b1;

        decoder_block(size_t c1, size_t c2, size_t s)
        : b0(c1, c2, s*2, s),
          b1(c2)
        {}

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            data = b0.load_weights(data);
            data = b1.load_weights(data);
            return data;
        }

        std::span<float> operator()(std::span<const float> input)
        {
            return b1(b0(input));
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
        encodec_lstm  b5;
        elu_layer     a6;
        conv          b6;
        std::vector<uint16_t> codes;
        std::vector<uint8_t>  buf;

        impl()
        : b0(  1,  32, 7),
          b1( 32,  64, 2),
          b2( 64, 128, 4),
          b3(128, 256, 5),
          b4(256, 512, 8),
          b5(512),
          a6(true),
          b6(512, 128, 7)
        {
            auto weights = encoder_weights;
            weights = b0.load_weights(weights);
            weights = b1.load_weights(weights);
            weights = b2.load_weights(weights);
            weights = b3.load_weights(weights);
            weights = b4.load_weights(weights);
            weights = b5.load_weights(weights);
            weights = b6.load_weights(weights);
            assert(weights.size()==0);
        }

        std::span<const float> encode(std::span<const float> audio, unsigned int num_quantizers)
        {
            assert(num_quantizers >= 1 && num_quantizers <= NLEVELS);

            // Run encoder
            auto x = b0(audio);
            x      = b1(x);
            x      = b2(x);
            x      = b3(x);
            x      = b4(x);
            x      = b5(x);
            x      = b6(a6(x));
            return x;
            // // Run RVQ
            // const size_t nfeats = x.size() / CODEBOOK_DIM;
            // codes.resize(nfeats*num_quantizers);
            // rvq_encode(x, codes, num_quantizers);

            // // Pack codes
            // buf.resize((codes.size()*10 + 7) / 8);
            // pack_codes(codes, buf);
            // return buf;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct decoder::impl
    {
        conv                    b0;
        encodec_lstm            b1;
        decoder_block           b2;
        // std::vector<uint16_t>   codes;
        // std::vector<float>      feats;

        impl()
        : b0(128, 512, 7),
          b1(512),
          b2(512, 256, 8)
        {
            auto weights = decoder_weights;
            weights = b0.load_weights(weights);
            weights = b1.load_weights(weights);
            weights = b2.load_weights(weights);
            // assert(weights.size()==0);
            printf("Leftover weights %zu\n", weights.size());
        }

        std::span<const float> decode(std::span<const float> feats, unsigned int num_quantizers)
        {
            assert(num_quantizers >= 1 && num_quantizers <= NLEVELS);

            // // Unpack bits
            // const size_t ncodes = (packet.size()*8)/10;
            // const size_t nfeats = ncodes/num_quantizers;
            // assert(ncodes % num_quantizers == 0);
            // codes.resize(ncodes);
            // unpack_bits(packet, codes);

            // // RVQ
            // feats.resize(nfeats*CODEBOOK_DIM);
            // memset(&feats[0], 0, feats.size()*sizeof(float));
            // rvq_decode(codes, feats, num_quantizers);

            // Run decoder
            auto x = b0(feats);
            x      = b1(x);
            x      = b2(x);

            return x;
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

    std::span<const float> decoder::decode(std::span<const float> feats, unsigned int num_quantizers)
    {
        return state->decode(feats, num_quantizers);
    }

//----------------------------------------------------------------------------------------------------------------

}