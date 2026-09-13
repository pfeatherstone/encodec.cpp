#include <cassert>
#include <vector>
#include <array>
#include <optional>
#include <Eigen/Dense>
#include "encodec.h"

//----------------------------------------------------------------------------------------------------------------

using MatrixXf      = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
using MatrixXu16    = Eigen::Matrix<uint16_t, -1, -1, Eigen::RowMajor>;
using VectorXf      = Eigen::Vector<float, -1>;
using ArrayXf       = Eigen::Array<float, -1, 1>;

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

    float normalize(std::span<float> input, size_t nchannels, float eps=1e-8f)
    {
        assert(nchannels > 0);
        assert(input.size() % nchannels == 0);

        const size_t T      = input.size() / nchannels;
        auto X              = Eigen::Map<MatrixXf>(input.data(), T, nchannels); // [T,C]
        const float scale   = std::sqrt(X.rowwise().mean().squaredNorm() / T) + eps;
        X.array() /= scale;

        return scale;
    }

//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
// CONSTANTS
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------

    constexpr unsigned  STRIDE          = 320;
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

    constexpr bitrates get_encodec_bps(sample_rates rate, unsigned int nlevels)
    {
        return static_cast<bitrates>((static_cast<unsigned int>(rate) / STRIDE) * nlevels * 10); 
    }

    constexpr unsigned int get_encodec_nquantizers(sample_rates rate, bitrates bps)
    {
        return static_cast<unsigned int>(bps) * STRIDE / (static_cast<unsigned int>(rate) * 10);
    }

    static_assert(get_encodec_bps(RATE_24KHZ, 32) == BPS_24000);
    static_assert(get_encodec_bps(RATE_24KHZ, 16) == BPS_12000);
    static_assert(get_encodec_bps(RATE_24KHZ, 8)  == BPS_6000);
    static_assert(get_encodec_bps(RATE_24KHZ, 4)  == BPS_3000);
    static_assert(get_encodec_bps(RATE_24KHZ, 2)  == BPS_1500);
    static_assert(get_encodec_bps(RATE_48KHZ, 16) == BPS_24000);
    static_assert(get_encodec_bps(RATE_48KHZ, 8)  == BPS_12000);
    static_assert(get_encodec_bps(RATE_48KHZ, 4)  == BPS_6000);
    static_assert(get_encodec_bps(RATE_48KHZ, 2)  == BPS_3000);
    static_assert(get_encodec_bps(RATE_48KHZ, 1)  == BPS_1500);

//----------------------------------------------------------------------------------------------------------------

    struct rvq
    {
        size_t                  nlevels{};
        std::vector<float>      weights;
        MatrixXf                Cnorms;
        MatrixXf                dists;
        std::vector<uint16_t>   codes;
        std::vector<uint8_t>    codes_packed;
        std::vector<float>      feats;

        auto codebook(size_t l) const
        {
            assert(l < nlevels);
            return Eigen::Map<const MatrixXf>(&weights[l*CODEBOOK_SIZE*CODEBOOK_DIM], CODEBOOK_SIZE, CODEBOOK_DIM);
        }
        
        rvq(std::span<const float> weights_) : weights(weights_.begin(), weights_.end())
        {
            constexpr size_t LEVEL_SIZE = CODEBOOK_SIZE * CODEBOOK_DIM;
            nlevels = weights.size() / LEVEL_SIZE;
            if (weights.size() % LEVEL_SIZE != 0) throw std::runtime_error("Bad rvq weights");
            if (!(nlevels == 32 || nlevels == 16)) throw std::runtime_error("Bad rvq weights");
        
            Cnorms.resize(nlevels, CODEBOOK_SIZE);

            for (size_t l{0}; l < nlevels; ++l)
                Cnorms.row(l) = codebook(l).rowwise().squaredNorm().transpose();
        }

        std::span<const uint16_t> get_codes(std::span<float> features, size_t num_quantizers)
        {
            assert(num_quantizers >= 1);
            assert(num_quantizers <= nlevels);
            assert(features.size() % CODEBOOK_DIM == 0);

            // RVQ Encode
            const size_t T = features.size() / CODEBOOK_DIM;
            codes.resize(T*num_quantizers);

            auto X = Eigen::Map<MatrixXf>(&features[0], T, CODEBOOK_DIM);

            for (size_t l{0} ; l < num_quantizers ; ++l)
            {
                auto C = codebook(l);
                dists.noalias() = -2.0f * X * C.transpose();
                dists.rowwise() += Cnorms.row(l);
                
                for (size_t t{0}; t < T; ++t)
                {
                    Eigen::Index best_idx{0};
                    dists.row(t).minCoeff(&best_idx);
                    X.row(t) -= C.row(best_idx);
                    codes[t*num_quantizers+l] = best_idx;                    
                }
            }

            return codes;
        }

        std::span<float> get_features(std::span<const uint16_t> codes, size_t num_quantizers)
        {
            assert(num_quantizers >= 1);
            assert(num_quantizers <= nlevels);
            assert(codes.size() % num_quantizers == 0);

            const size_t T = codes.size()/num_quantizers;
            feats.resize(T*CODEBOOK_DIM);

            auto X = Eigen::Map<const MatrixXu16>(&codes[0], T, num_quantizers);
            auto Y = Eigen::Map<MatrixXf>(&feats[0], T, CODEBOOK_DIM);
            Y.setZero();

            // RVQ decode
            for (size_t l{0}; l < num_quantizers; ++l)
            {
                auto C = codebook(l);
                for (size_t t{0}; t < T; ++t)
                    Y.row(t) += C.row(X(t,l));
            }

            return feats;
        }

        std::span<const uint8_t> pack(std::span<const uint16_t> codes)
        {
            codes_packed.resize((codes.size()*10 + 7) / 8);
            pack_codes(codes, codes_packed);
            return codes_packed;
        }

        std::span<const uint16_t> unpack(std::span<const uint8_t> packet)
        {
            const size_t ncodes = (packet.size()*8)/10;
            codes.resize(ncodes);
            unpack_bits(packet, codes);
            return codes;
        }

        std::span<const uint8_t> encode(std::span<float> features, size_t num_quantizers)
        {
            return pack(get_codes(features, num_quantizers));
        }

        std::span<const float> decode(std::span<const uint8_t> packet, size_t num_quantizers)
        {
            return get_features(unpack(packet), num_quantizers);
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

    struct time_group_norm
    {
        VectorXf w; // [C,1]
        VectorXf b; // [C,1]
        float    eps{};

        time_group_norm() = default;
        
        time_group_norm(size_t nin_, float eps_ = 1e-5f)
        : w(nin_),
          b(nin_),
          eps{eps_}
        {
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < size_t(w.size()+b.size())) throw std::runtime_error("Not enough data in groupnorm weights");
            size_t off{0};
            w = Eigen::Map<const VectorXf>(data.subspan(off, w.size()).data(), w.size()); off += w.size();
            b = Eigen::Map<const VectorXf>(data.subspan(off, b.size()).data(), b.size()); off += b.size();
            return data.subspan(off);
        }

        size_t nin() const noexcept { return b.size(); }

        std::span<float> operator()(std::span<float> input)
        {
            assert(nin() > 0);
            assert(input.size() % nin() == 0);
            
            const size_t T = input.size() / nin();
            auto X = Eigen::Map<MatrixXf>(input.data(), T, nin()); // [T,C]

            // GroupNorm(1, C): statistics over all C*T values.
            const float mean = X.mean();
            const float var  = (X.array() - mean).square().mean();
            const float inv  = 1.0f / std::sqrt(var + eps);

            // Normalize.
            X.array() = (X.array() - mean) * inv;

            // Per-channel affine transform.
            X.array().rowwise() *= w.transpose().array(); 
            X.rowwise()         += b.transpose();

            return input;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct linear
    {
        MatrixXf w;
        VectorXf b;
        MatrixXf out;
        std::optional<time_group_norm> norm;

        linear(size_t nin_, size_t nout_, bool norm_)
        : w(nout_, nin_),
          b(nout_)
        {
            if (norm_)
                norm.emplace(nout_);
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < size_t(w.size() + b.size())) throw std::runtime_error("Not enough data in weights");
            size_t off{0};
            auto w_ = Eigen::Map<const MatrixXf>(data.subspan(off, w.size()).data(), nout(), nin()); off += w.size();
            auto b_ = Eigen::Map<const VectorXf>(data.subspan(off, b.size()).data(), nout());        off += b.size();
            w       = w_;
            b       = b_;
            data    = data.subspan(off);
            if (norm) data = norm->load_weights(data);
            return data;
        }

        size_t nin()  {return w.cols();}
        size_t nout() {return w.rows();}

        std::span<float> operator()(std::span<const float> input)
        {
            assert(input.size() % nin() == 0);

            // Linear
            const size_t Tin = input.size() / nin();
            auto x = Eigen::Map<const MatrixXf>(input.data(), Tin, nin());
            out.noalias() = x * w.transpose() ;
            out.rowwise() += b.transpose();
            std::span<float> y(out.data(), (size_t)out.size());

            // Norm
            if (norm)
                y = (*norm)(y);

            return y;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    using padding = std::array<size_t,2>;

    struct conv
    {
        size_t   nin{};
        size_t   nout{};
        size_t   k{};
        size_t   s{};
        bool     causal{};
        auto     pad_total()  const { return k - s;}
        padding  pads()       const { const auto p = pad_total(); return causal ? padding{p,0} : padding{p-p/2,p/2}; }
        MatrixXf w;         // shape [nout,k*nin]
        VectorXf b;         // shape [nout]
        MatrixXf patches;   // shape [Tout, k*nin]
        MatrixXf out;       // [Tout, nout]
        std::optional<time_group_norm> norm;

        conv(size_t nin_, size_t nout_, size_t k_, size_t s_=1, bool causal_=true, bool norm_=false)
        : nin{nin_}, 
          nout{nout_}, 
          k{k_}, 
          s{s_}, 
          causal{causal_},
          w(nout, k*nin),
          b(nout)
        {
            if (norm_)
                norm.emplace(nout_);
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < size_t(w.size() + b.size())) throw std::runtime_error("Not enough data in weights");
            size_t off{0};
            auto w_ = Eigen::Map<const MatrixXf>(data.subspan(off, w.size()).data(), nout, k*nin); off += w.size();
            auto b_ = Eigen::Map<const VectorXf>(data.subspan(off, b.size()).data(), nout);        off += b.size();
            w       = w_;
            b       = b_;
            data    = data.subspan(off);
            if (norm) data = norm->load_weights(data);
            return data;
        }

        std::span<float> operator()(std::span<const float> input)
        {
            assert(input.size() % nin == 0);

            const auto [pl,pr]  = pads();
            const size_t Tin    = input.size() / nin;
            const size_t Tinp   = Tin+pl+pr;
            const size_t Tout   = (Tinp-k)/s + 1;

            patches.resize(Tout, k*nin);

            // im2col : patches[i, :] = padded_input[i*s : i*s+k, :]
            size_t i{0};

            // Left reflected padding
            for (; (i*s) < pl ; ++i)
            {
                for (size_t kk{0}; kk < k; ++kk)
                {
                    const size_t tp = i*s + kk;
                    const size_t ti = tp < pl ? pl - tp : tp - pl;
                    std::copy_n(input.data()+ti*nin,nin,patches.data()+(i*k+kk)*nin);
                }
            }

            // Middle
            for (; i < Tout && (i*s+k) <= (pl+Tin); ++i)
                std::copy_n(input.data()+(i*s-pl)*nin,k*nin,patches.data()+i*k*nin);

            // Right reflected padding
            for (; i < Tout; ++i)
            {
                for (size_t kk{0}; kk < k; ++kk)
                {
                    const size_t tp = i*s + kk;
                    const size_t ti = tp < (pl+Tin) ? tp - pl : 2*Tin + pl - tp - 2;
                    std::copy_n(input.data()+ti*nin,nin,patches.data()+(i*k+kk)*nin);
                }
            }

            // GEMM
            out.noalias() = patches * w.transpose();
            out.rowwise() += b.transpose();
            std::span<float> y(out.data(), (size_t)out.size());

            // Norm
            if (norm)
                y = (*norm)(y);

            return y;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct conv_transpose
    {
        size_t   nin{};
        size_t   nout{};
        size_t   k{};
        size_t   s{};
        bool     causal{};
        auto     pad_total()  const { return k - s;}
        padding  pads()       const { const auto p = pad_total(); return causal ? padding{0,p} : padding{p-p/2,p/2}; }
        MatrixXf w;         // [nin, k*nout], raw layout [nin,k,nout]
        VectorXf b;         // [nout]
        MatrixXf patches;   // [Tin, k*nout]
        MatrixXf out;       // [Tout_padded, nout]
        std::optional<time_group_norm> norm;

        conv_transpose(size_t nin_, size_t nout_, size_t k_, size_t s_ = 1, bool causal_= true, bool norm_=false)
        : nin{nin_}, 
          nout{nout_}, 
          k{k_}, 
          s{s_}, 
          causal{causal_},
          w(nin,k*nout),
          b(nout)
        {
            if (norm_)
                norm.emplace(nout_);
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < size_t(w.size() + b.size())) throw std::runtime_error("Not enough data in weights");
            size_t off{0};
            auto w_ = Eigen::Map<const MatrixXf>(data.subspan(off, w.size()).data(), nin, k*nout); off += w.size();
            auto b_ = Eigen::Map<const VectorXf>(data.subspan(off, b.size()).data(), nout);        off += b.size();
            w       = w_;
            b       = b_;
            data    = data.subspan(off);
            if (norm) data = norm->load_weights(data);
            return data;
        }

        std::span<float> operator()(std::span<const float> input)
        {
            assert(input.size() % nin == 0);

            const auto [pl,pr]        = pads();
            const size_t Tin          = input.size() / nin;
            const size_t Tout_padded  = (Tin - 1) * s + k;
            const size_t Tout         = Tout_padded - pl - pr;
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
                    out.row(to) += patches.block(t, kk*nout, 1, nout);
                }
            }

            // Add bias
            out.rowwise() += b.transpose();
            std::span<float> y(out.data(), out.size());

            // Norm
            if (norm)
                y = (*norm)(y);

            // Crop
            return std::span<float>{out.data()+pl*nout, Tout*nout};
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

        resnet_block(size_t c, bool causal, bool norm)
        : a0(false),
          a1(true),
          b0(c,   c/2, 3, 1, causal, norm),
          b1(c/2, c, norm),
          b2(c,   c, norm)
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

        encoder_block(size_t c1, size_t c2, size_t s, bool causal, bool norm)
        : b0(c1, causal, norm),
          a1(true),
          b1(c1, c2, s*2, s, causal, norm)
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

        decoder_block(size_t c1, size_t c2, size_t s, bool causal, bool norm)
        : a0(true),
          b0(c1, c2, s*2, s, causal, norm),
          b1(c2, causal, norm)
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
        sample_rates  rate;
        conv          b0;
        encoder_block b1;
        encoder_block b2;
        encoder_block b3;
        encoder_block b4;
        encodec_lstm  b5;
        elu_layer     a6;
        conv          b6;
        rvq           rvq_;
        bool causal() const noexcept {return rate==RATE_24KHZ;}
        bool norm()   const noexcept {return rate==RATE_48KHZ;}
        size_t nc()   const noexcept {return rate==RATE_24KHZ ? 1 : 2;}

        impl(sample_rates rate_, std::span<const float> weights, std::span<const float> rvq_weights)
        : rate{rate_},
          b0(nc(),  32, 7, 1, causal(), norm()),
          b1( 32,  64, 2, causal(), norm()),
          b2( 64, 128, 4, causal(), norm()),
          b3(128, 256, 5, causal(), norm()),
          b4(256, 512, 8, causal(), norm()),
          b5(512),
          a6(true),
          b6(512, 128, 7, 1, causal(), norm()),
          rvq_(rvq_weights)
        {
            weights = b0.load_weights(weights);
            weights = b1.load_weights(weights);
            weights = b2.load_weights(weights);
            weights = b3.load_weights(weights);
            weights = b4.load_weights(weights);
            weights = b5.load_weights(weights);
            weights = b6.load_weights(weights);
            if (!weights.empty()) throw std::runtime_error("Failed to load encoder weights");
        }

        std::span<float> features(std::span<const float> audio)
        {
            auto x = b0(audio);
            x      = b1(x);
            x      = b2(x);
            x      = b3(x);
            x      = b4(x);
            x      = b5(x);
            x      = b6(a6(x));
            return x;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct decoder::impl
    {
        sample_rates  rate;
        conv          b0;
        encodec_lstm  b1;
        decoder_block b2;
        decoder_block b3;
        decoder_block b4;
        decoder_block b5;
        elu_layer     a6;
        conv          b6;
        rvq           rvq_;
        bool causal() const noexcept {return rate==RATE_24KHZ;}
        bool norm()   const noexcept {return rate==RATE_48KHZ;}
        size_t nc()   const noexcept {return rate==RATE_24KHZ ? 1 : 2;}

        impl(sample_rates rate_, std::span<const float> weights, std::span<const float> rvq_weights)
        : rate{rate_},
          b0(128, 512, 7, 1, causal(), norm()),
          b1(512),
          b2(512, 256, 8, causal(), norm()),
          b3(256, 128, 5, causal(), norm()),
          b4(128,  64, 4, causal(), norm()),
          b5( 64,  32, 2, causal(), norm()),
          a6(true),
          b6( 32,  nc(), 7, 1, causal(), norm()),
          rvq_(rvq_weights)
        {
            weights = b0.load_weights(weights);
            weights = b1.load_weights(weights);
            weights = b2.load_weights(weights);
            weights = b3.load_weights(weights);
            weights = b4.load_weights(weights);
            weights = b5.load_weights(weights);
            weights = b6.load_weights(weights);
            if (!weights.empty()) throw std::runtime_error("Failed to load decoder weights");
        }

        std::span<const float> audio(std::span<const float> features)
        {
            auto x  = b0(features);
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

    encoder::encoder(sample_rates rate, std::span<const float> w0, std::span<const float> w1) : state{std::make_unique<impl>(rate,w0,w1)} {}
    encoder::~encoder()                          = default;
    encoder::encoder(encoder&& other)            = default;
    encoder& encoder::operator=(encoder&& other) = default;

    std::span<float> encoder::features(std::span<const float> audio)
    {
        return state->features(audio);
    }

    std::span<const uint16_t> encoder::codes(std::span<float> features, bitrates bps)
    {
        return state->rvq_.get_codes(features, get_encodec_nquantizers(state->rate, bps));
    }

    std::span<const uint8_t> encoder::packet(std::span<const uint16_t> codes)
    {
        return state->rvq_.pack(codes);
    }

    std::span<const uint8_t> encoder::encode(std::span<const float> audio, bitrates bps)
    {
        return packet(codes(features(audio), bps));
    }

//----------------------------------------------------------------------------------------------------------------

    decoder::decoder(sample_rates rate, std::span<const float> w0, std::span<const float> w1) : state{std::make_unique<impl>(rate,w0,w1)} {}
    decoder::~decoder()                          = default;
    decoder::decoder(decoder&& other)            = default;
    decoder& decoder::operator=(decoder&& other) = default;

    std::span<const uint16_t> decoder::codes(std::span<const uint8_t> packet)
    {
        return state->rvq_.unpack(packet);
    }

    std::span<const float> decoder::features(std::span<const uint16_t> codes, bitrates bps)
    {
        return state->rvq_.get_features(codes, get_encodec_nquantizers(state->rate, bps));
    }

    std::span<const float> decoder::audio (std::span<const float> features)
    {
        return state->audio(features);
    }

    std::span<const float> decoder::decode(std::span<const uint8_t> packet, bitrates bps)
    {
        return audio(features(codes(packet), bps));
    }

//----------------------------------------------------------------------------------------------------------------

}