#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include "encodec.h"

//----------------------------------------------------------------------------------------------------------------

using MatrixXf      = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
using MatrixXu16    = Eigen::Matrix<uint16_t, -1, -1, Eigen::RowMajor>;
using VectorXf      = Eigen::Vector<float, -1>;
using ArrayXf       = Eigen::Array<float, -1, 1>;

//----------------------------------------------------------------------------------------------------------------

extern const float       ENCODER_WEIGHTS[];
extern const std::size_t ENCODER_SIZE;
extern const float       DECODER_WEIGHTS[];
extern const std::size_t DECODER_SIZE;
extern const float       RVQ_WEIGHTS[];
extern const std::size_t RVQ_SIZE;

//----------------------------------------------------------------------------------------------------------------

namespace encodec
{

//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
// MATH
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------

    constexpr void add(std::span<const float> a, std::span<const float> b, std::span<float> c)
    {
        for (size_t i{0} ; i < a.size() ; ++i)
            c[i] = a[i] + b[i];
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

//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
// RVQ
//----------------------------------------------------------------------------------------------------------------
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

    unsigned int get_encodec_bps(unsigned int nlevels)      { return (SAMPLE_RATE / STRIDE) * nlevels * 10; }
    unsigned int get_encoded_nquantizers(unsigned int bps)  { return (bps / 10) * STRIDE / SAMPLE_RATE; }

//----------------------------------------------------------------------------------------------------------------

    auto codebook(size_t l)
    {
        return Eigen::Map<const MatrixXf>(&RVQ_WEIGHTS[l*CODEBOOK_SIZE*CODEBOOK_DIM], CODEBOOK_SIZE, CODEBOOK_DIM);
    }
    
//----------------------------------------------------------------------------------------------------------------

    struct rvq
    {
        VectorXf Cnorms[NLEVELS];
        MatrixXf dists;
        std::vector<uint16_t> codes;
        std::vector<uint8_t>  codes_packed;
        std::vector<float>    feats;

        rvq()
        {
            for (size_t l{0} ; l < NLEVELS ; ++l)
                Cnorms[l] = codebook(l).rowwise().squaredNorm();
        }

        std::span<const uint8_t> encode(std::span<float> feats, size_t nlevels)
        {
            // RVQ Encode
            const size_t T = feats.size() / CODEBOOK_DIM;
            codes.resize(T*nlevels);
            codes_packed.resize((codes.size()*10 + 7) / 8);

            auto X = Eigen::Map<MatrixXf>(&feats[0], T, CODEBOOK_DIM);

            for (size_t l{0} ; l < nlevels ; ++l)
            {
                auto C = codebook(l);
                dists.noalias() = -2.0f * X * C.transpose();
                dists.rowwise() += Cnorms[l].transpose();
                
                for (size_t t{0}; t < T; ++t)
                {
                    Eigen::Index best_idx{0};
                    dists.row(t).minCoeff(&best_idx);
                    X.row(t) -= C.row(best_idx);
                    codes[t*nlevels+l] = best_idx;                    
                }
            }   

            // Pack
            pack_codes(codes, codes_packed);
            return codes_packed;
        }

        std::span<float> decode(std::span<const uint8_t> codes_packed, size_t nlevels)
        {
            // Unpack bits
            const size_t ncodes = (codes_packed.size()*8)/10;
            const size_t T      = ncodes/nlevels;
            codes.resize(ncodes);
            feats.resize(T*CODEBOOK_DIM);
            unpack_bits(codes_packed, codes);

            auto X = Eigen::Map<const MatrixXu16>(&codes[0], T, nlevels);
            auto Y = Eigen::Map<MatrixXf>(&feats[0], T, CODEBOOK_DIM);
            Y.setZero();

            // RVQ decode
            for (size_t l{0}; l < nlevels; ++l)
            {
                auto C = codebook(l);
                for (size_t t{0}; t < T; ++t)
                    Y.row(t) += C.row(X(t,l));
            }

            return feats;
        }
    };

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
        size_t   nin{};
        size_t   nout{};
        size_t   k{};
        size_t   s{};
        size_t   pad() {return (k - 1) + 1 - s;}
        MatrixXf w;         // shape [nout,k*nin]
        VectorXf b;         // shape [nout]
        MatrixXf patches;   // shape [Tout, k*nin]
        MatrixXf out;       // [Tout_padded, nout]

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
            const size_t p    = pad();
            const size_t Tin  = input.size() / nin;
            const size_t Tinp = Tin+p;
            const size_t Tout = (Tinp-k)/s + 1;

            patches.resize(Tout, k*nin);

            // im2col : patches[j, :] = padded_input[j*s : j*s+k, :]
            size_t i{0};
            for (; (i*s) < p ; ++i)
            {
                for (size_t kk = 0; kk < k; ++kk)
                {
                    const size_t tp = i*s + kk;
                    const size_t ti = tp < p ? p - tp : tp - p;
                    std::copy_n(input.data()+ti*nin,nin,patches.data()+(i*k+kk)*nin);
                }
            }
            for (; i < Tout ; ++i)
                std::copy_n(input.data()+(i*s-p)*nin,k*nin,patches.data()+i*k*nin);

            // GEMM
            out.noalias() = patches * w.transpose();
            out.rowwise() += b.transpose();

            return std::span<float>{out.data(), (size_t)out.size()};
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct conv_transpose
    {
        size_t   nin{};
        size_t   nout{};
        size_t   k{};
        size_t   s{};
        size_t   pad() {return (k - 1) + 1 - s;}
        MatrixXf w;         // [nin, k*nout], raw layout [nin,k,nout]
        VectorXf b;         // [nout]
        MatrixXf patches;   // [Tin, k*nout]
        MatrixXf out;       // [Tout_padded, nout]

        conv_transpose(size_t nin_, size_t nout_, size_t k_, size_t s_ = 1)
        : nin{nin_}, 
          nout{nout_}, 
          k{k_}, 
          s{s_}, 
          w(nin,k*nout),
          b(nout)
        {}

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < size_t(w.size() + b.size())) throw std::runtime_error("Not enough data in weights");
            size_t off{0};
            auto w_ = Eigen::Map<const MatrixXf>(data.subspan(off, w.size()).data(), nin, k*nout); off += w.size();
            auto b_ = Eigen::Map<const VectorXf>(data.subspan(off, b.size()).data(), nout);        off += b.size();
            w       = w_;
            b       = b_;
            return data.subspan(off);
        }

        std::span<float> operator()(std::span<const float> input)
        {
            const size_t Tin         = input.size() / nin;
            const size_t p           = pad();
            const size_t Tout_padded = (Tin - 1) * s + (k - 1) + 1;
            const size_t Tout        = Tout_padded - p;
            out.setZero(Tout_padded, nout);

            // GEMM
            auto X = Eigen::Map<const MatrixXf>(input.data(), Tin, nin);
            patches.noalias() = X * w; // [Tin, k*nout]

            // col2im / overlap-add
            for (size_t t{0}; t < Tin; ++t)
            {
                for (size_t kk{0}; kk < k; ++kk)
                {
                    const size_t to = t*s + kk;
                    out.row(to) += patches.block(t, kk * nout, 1, nout);
                }
            }

            // Add bias
            out.rowwise() += b.transpose();

            return std::span<float>{out.data(), Tout*nout};
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
        elu_layer       a0;
        conv_transpose  b0;
        resnet_block    b1;

        decoder_block(size_t c1, size_t c2, size_t s)
        : a0(true),
          b0(c1, c2, s*2, s),
          b1(c2)
        {}

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            data = b0.load_weights(data);
            data = b1.load_weights(data);
            return data;
        }

        std::span<float> operator()(std::span<float> input)
        {
            return b1(b0(a0(input)));
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
        rvq           rvq_;

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
            auto weights = std::span{ENCODER_WEIGHTS, ENCODER_SIZE};
            weights = b0.load_weights(weights);
            weights = b1.load_weights(weights);
            weights = b2.load_weights(weights);
            weights = b3.load_weights(weights);
            weights = b4.load_weights(weights);
            weights = b5.load_weights(weights);
            weights = b6.load_weights(weights);
            if (!weights.empty()) throw std::runtime_error("Failed to load encoder weights");
        }

        std::span<const uint8_t> encode(std::span<const float> audio, unsigned int num_quantizers)
        {
            assert(num_quantizers >= 1 && num_quantizers <= NLEVELS);

            auto x = b0(audio);
            x      = b1(x);
            x      = b2(x);
            x      = b3(x);
            x      = b4(x);
            x      = b5(x);
            x      = b6(a6(x));
            return rvq_.encode(x, num_quantizers);
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct decoder::impl
    {
        conv          b0;
        encodec_lstm  b1;
        decoder_block b2;
        decoder_block b3;
        decoder_block b4;
        decoder_block b5;
        elu_layer     a6;
        conv          b6;
        rvq           rvq_;

        impl()
        : b0(128, 512, 7),
          b1(512),
          b2(512, 256, 8),
          b3(256, 128, 5),
          b4(128,  64, 4),
          b5( 64,  32, 2),
          a6(true),
          b6( 32,   1, 7)
        {
            auto weights = std::span{DECODER_WEIGHTS, DECODER_SIZE};
            weights = b0.load_weights(weights);
            weights = b1.load_weights(weights);
            weights = b2.load_weights(weights);
            weights = b3.load_weights(weights);
            weights = b4.load_weights(weights);
            weights = b5.load_weights(weights);
            weights = b6.load_weights(weights);
            if (!weights.empty()) throw std::runtime_error("Failed to load decoder weights");
        }

        std::span<const float> decode(std::span<const uint8_t> packet, unsigned int num_quantizers)
        {
            assert(num_quantizers >= 1 && num_quantizers <= NLEVELS);

            auto x  = rvq_.decode(packet, num_quantizers);
            x       = b0(x);
            x       = b1(x);
            x       = b2(x);
            x       = b3(x);
            x       = b4(x);
            x       = b5(x);
            x       = b6(a6(x));
            return x;
        }
    };
    
//----------------------------------------------------------------------------------------------------------------

    encoder::encoder() : state{std::make_unique<impl>()} {}
    encoder::~encoder()                          = default;
    encoder::encoder(encoder&& other)            = default;
    encoder& encoder::operator=(encoder&& other) = default;

    std::span<const uint8_t> encoder::encode(std::span<const float> audio, unsigned int num_quantizers)
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