// inbed Metal kernels. Compiled at startup with newLibraryWithSource (no offline toolchain needed).
//
// Conventions: activations are float [tokens, cols] row-major; matmul weights and embedding tables
// are half; biases and norm weights are float. Packed varlen batches: `pos[t]` is the token's
// position in its sequence (so its sequence starts at t - pos[t]), `seq_end[t]` is the exclusive end.

#include <metal_stdlib>
using namespace metal;

// Abramowitz–Stegun 7.1.26, |error| < 1.5e-7 (same as the CPU reference)
inline float erf_as(float x) {
    float s = sign(x);
    x = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return s * y;
}

inline float gelu(float x) { return 0.5f * x * (1.0f + erf_as(x * 0.70710678118654752f)); }

// ------------------------------------------------------------------ gemm: C = act(A · Wᵀ + bias) + R

struct GemmP { uint M, N, K, has_bias, has_res, act; };

#define BM 64
#define BN 64
#define BK 32
#define AP (BK + 4)   // padded row strides avoid threadgroup bank conflicts
#define WP (BN + 4)
#define CP (BN + 4)

// 128 threads = 4 simdgroups in a 2x2 grid, each owning a 32x32 block of the 64x64 tile.
// A tile is stored [m][k]; the W tile is stored transposed as [k][n] so B fragments load directly.
kernel void gemm(device const float* A [[buffer(0)]],
                 device const half* W [[buffer(1)]],
                 device const float* bias [[buffer(2)]],
                 device const float* R [[buffer(3)]],
                 device float* C [[buffer(4)]],
                 constant GemmP& p [[buffer(5)]],
                 uint2 tg [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[(BM * AP + BK * WP) > (BM * CP) ? (BM * AP + BK * WP) : (BM * CP)];
    threadgroup float* As = smem;
    threadgroup float* Ws = smem + BM * AP;
    const uint m0 = tg.y * BM, n0 = tg.x * BN;
    const uint sm = (sg / 2) * 32, sn = (sg % 2) * 32;
    const bool vec = (p.K % BK) == 0;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        if (vec) {
            // 64 rows x 8 float4 per tile = 512 vectors, 4 per thread
            for (uint j = 0; j < 4; j++) {
                uint idx = tid + 128 * j, r = idx / 8, c = (idx % 8) * 4;
                uint gm = m0 + r;
                float4 v = gm < p.M ? *((device const float4*)(A + gm * p.K + k0 + c)) : float4(0.0f);
                *((threadgroup float4*)(As + r * AP + c)) = v;
                uint gn = n0 + r;
                half4 w = gn < p.N ? *((device const half4*)(W + gn * p.K + k0 + c)) : half4(0.0h);
                Ws[(c + 0) * WP + r] = float(w.x);
                Ws[(c + 1) * WP + r] = float(w.y);
                Ws[(c + 2) * WP + r] = float(w.z);
                Ws[(c + 3) * WP + r] = float(w.w);
            }
        } else {
            for (uint i = tid; i < BM * BK; i += 128) {
                uint r = i / BK, c = i % BK, gm = m0 + r, gn = n0 + r, gk = k0 + c;
                As[r * AP + c] = (gm < p.M && gk < p.K) ? A[gm * p.K + gk] : 0.0f;
                Ws[c * WP + r] = (gn < p.N && gk < p.K) ? float(W[gn * p.K + gk]) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[4], b[4];
            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], As + (sm + i * 8) * AP + kk, AP);
            for (uint j = 0; j < 4; j++) simdgroup_load(b[j], Ws + kk * WP + sn + j * 8, WP);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], smem + (sm + i * 8) * CP + sn + j * 8, CP);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BM * BN; i += 128) {
        uint r = i / BN, c = i % BN, gm = m0 + r, gn = n0 + c;
        if (gm < p.M && gn < p.N) {
            float v = smem[r * CP + c];
            if (p.has_bias) v += bias[gn];
            if (p.act == 1) v = gelu(v);
            if (p.has_res) v += R[gm * p.N + gn];
            C[gm * p.N + gn] = v;
        }
    }
}

// epilogue after an MPS matmul, in place: C = act(C + bias) + R
kernel void epilogue(device float* C [[buffer(0)]],
                     device const float* bias [[buffer(1)]],
                     device const float* R [[buffer(2)]],
                     constant GemmP& p [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint n = gid.x, m = gid.y;
    if (n >= p.N || m >= p.M) return;
    uint i = m * p.N + n;
    float v = C[i];
    if (p.has_bias) v += bias[n];
    if (p.act == 1) v = gelu(v);
    if (p.has_res) v += R[i];
    C[i] = v;
}

// ------------------------------------------------------------------ embeddings

struct EmbedP { uint cols, has_pos, has_type; };

kernel void embed(device const uint* ids [[buffer(0)]],
                  device const uint* pos [[buffer(1)]],
                  device const half* word [[buffer(2)]],
                  device const half* pt [[buffer(3)]],
                  device const half* tt [[buffer(4)]],
                  device float* out [[buffer(5)]],
                  constant EmbedP& p [[buffer(6)]],
                  uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, t = gid.y;
    if (c >= p.cols) return;
    float v = float(word[ids[t] * p.cols + c]);
    if (p.has_pos) v += float(pt[pos[t] * p.cols + c]);
    if (p.has_type) v += float(tt[c]);
    out[t * p.cols + c] = v;
}

// ------------------------------------------------------------------ norms (one 256-thread group per row)

struct NormP { uint cols; float eps; };

inline float block_sum(float v, threadgroup float* red, uint lane, uint sg) {
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint i = 0; i < 8; i++) t += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

kernel void layer_norm(device const float* x [[buffer(0)]],
                       device const float* g [[buffer(1)]],
                       device const float* b [[buffer(2)]],
                       device float* out [[buffer(3)]],
                       constant NormP& p [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    device const float* xr = x + row * p.cols;
    float s = 0.0f;
    for (uint i = tid; i < p.cols; i += 256) s += xr[i];
    float mean = block_sum(s, red, lane, sg) / float(p.cols);
    float v = 0.0f;
    for (uint i = tid; i < p.cols; i += 256) { float d = xr[i] - mean; v += d * d; }
    float var = block_sum(v, red, lane, sg) / float(p.cols);
    float r = 1.0f / sqrt(var + p.eps);
    for (uint i = tid; i < p.cols; i += 256) out[row * p.cols + i] = (xr[i] - mean) * r * g[i] + b[i];
}

kernel void rms_norm(device const float* x [[buffer(0)]],
                     device const float* g [[buffer(1)]],
                     device float* out [[buffer(2)]],
                     constant NormP& p [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    device const float* xr = x + row * p.cols;
    float s = 0.0f;
    for (uint i = tid; i < p.cols; i += 256) s += xr[i] * xr[i];
    float r = 1.0f / sqrt(block_sum(s, red, lane, sg) / float(p.cols) + p.eps);
    for (uint i = tid; i < p.cols; i += 256) out[row * p.cols + i] = xr[i] * r * g[i];
}

// ------------------------------------------------------------------ Qwen3 q/k RMSNorm + rotary (one simdgroup per token-head)

struct RopeP { uint stride, n_q, n_kv, hd; float eps, theta; };

kernel void qk_norm_rope(device float* qkv [[buffer(0)]],
                         device const uint* pos [[buffer(1)]],
                         device const float* qn [[buffer(2)]],
                         device const float* kn [[buffer(3)]],
                         constant RopeP& p [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint h = tg.x, t = tg.y;
    device float* x = qkv + t * p.stride + h * p.hd;
    device const float* w = h < p.n_q ? qn : kn;
    uint per = p.hd / 32;
    float v[8];
    float ss = 0.0f;
    for (uint j = 0; j < per; j++) { v[j] = x[lane + 32 * j]; ss += v[j] * v[j]; }
    ss = simd_sum(ss);
    float r = 1.0f / sqrt(ss / float(p.hd) + p.eps);
    for (uint j = 0; j < per; j++) v[j] = v[j] * r * w[lane + 32 * j];
    uint hp = per / 2;
    float fpos = float(pos[t]);
    for (uint j = 0; j < hp; j++) {
        uint i = lane + 32 * j;
        float inv = 1.0f / pow(p.theta, float(2 * i) / float(p.hd));
        float ang = fpos * inv;
        float c = precise::cos(ang), s = precise::sin(ang);
        float a = v[j], b = v[j + hp];
        v[j] = a * c - b * s;
        v[j + hp] = b * c + a * s;
    }
    for (uint j = 0; j < per; j++) x[lane + 32 * j] = v[j];
}

// ------------------------------------------------------------------ attention (one simdgroup per token-head, online softmax)

struct AttnP { uint n_q, n_kv, hd, causal; float scale; uint tokens; };

kernel void attention(device const float* qkv [[buffer(0)]],
                      device const uint* pos [[buffer(1)]],
                      device const uint* seq_end [[buffer(2)]],
                      device float* out [[buffer(3)]],
                      constant AttnP& p [[buffer(4)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint h = tg.x, t = tg.y;
    uint stride = (p.n_q + 2 * p.n_kv) * p.hd;
    uint kvh = h / (p.n_q / p.n_kv);
    threadgroup float q[256];
    for (uint d = lane; d < p.hd; d += 32) q[d] = qkv[t * stride + h * p.hd + d] * p.scale;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint start = t - pos[t];
    uint end = p.causal ? t + 1 : seq_end[t];
    uint per = p.hd / 32;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.0f;
    device const float* kbase = qkv + (p.n_q + kvh) * p.hd;
    device const float* vbase = qkv + (p.n_q + p.n_kv + kvh) * p.hd;
    for (uint j0 = start; j0 < end; j0 += 32) {
        uint j = j0 + lane;
        float s = -INFINITY;
        if (j < end) {
            device const float* k = kbase + j * stride;
            float d = 0.0f;
            for (uint e = 0; e < p.hd; e++) d += q[e] * k[e];
            s = d;
        }
        float mx = max(m, simd_max(s));
        float pj = (j < end) ? exp(s - mx) : 0.0f;
        float corr = exp(m - mx);
        l = l * corr + simd_sum(pj);
        for (uint e = 0; e < per; e++) acc[e] *= corr;
        uint cnt = min(32u, end - j0);
        for (uint jj = 0; jj < cnt; jj++) {
            float pp = simd_shuffle(pj, ushort(jj));
            device const float* v = vbase + (j0 + jj) * stride;
            for (uint e = 0; e < per; e++) acc[e] += pp * v[lane + 32 * e];
        }
        m = mx;
    }
    device float* o = out + t * p.n_q * p.hd + h * p.hd;
    for (uint e = 0; e < per; e++) o[lane + 32 * e] = acc[e] / l;
}

// ------------------------------------------------------------------ flash attention (simdgroup MMA)
//
// Threadgroup = 8 simdgroups = 64 consecutive query rows of one head. Key blocks of FK rows are
// staged in threadgroup memory (K, then V, in the same padded buffer); each simdgroup computes
// Sᵀ = K·Qᵀ for its 8 rows with 8x8 MMAs (stored back transposed), does the online-softmax update (4 lanes per row),
// rescales O by diag(corr) and adds P·V.
// Rows mask keys outside their own sequence (and the future, when causal), so packed varlen
// batches work. Instantiated for head_dim 32, 64, 128; tensors must be padded to 64 rows.

#define FQ 8      // query rows per simdgroup
#define FSG 4     // simdgroups per threadgroup
#define FROWS (FQ * FSG)
#define FK 32     // keys per block

template <uint HD>
kernel void flash_attention(device const float* qkv [[buffer(0)]],
                            device const uint* pos [[buffer(1)]],
                            device const uint* seq_end [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant AttnP& p [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float KV[FK * (HD + 4)];        // K tile, then V tile: [key][d], row stride HD + 4
    threadgroup float Ss[FSG][FQ * FK];
    threadgroup float Ds[FSG][64];
    threadgroup uint rs[FROWS], re[FROWS];
    threadgroup uint range[2];

    const uint T = p.tokens;
    const uint hd = HD;
    constexpr uint ND = HD / 8;
    const uint h = tg.y, t0 = tg.x * FROWS;
    const uint stride = (p.n_q + 2 * p.n_kv) * hd;
    const uint kvh = h / (p.n_q / p.n_kv);
    const uint r0 = t0 + sg * FQ;
    const uint nthreads = FSG * 32;
    const uint ks = hd + 4;   // padded tile row stride
    const uint nv = hd / 4;   // float4 per row

    if (tid < FROWS) {
        uint r = t0 + tid;
        rs[tid] = r < T ? r - pos[r] : 0xffffffffu;
        re[tid] = r < T ? (p.causal ? r + 1 : seq_end[r]) : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint lo = 0xffffffffu, hi = 0;
        for (uint i = 0; i < FROWS; i++) { lo = min(lo, rs[i]); hi = max(hi, re[i]); }
        range[0] = lo;
        range[1] = hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lo = range[0], hi = range[1];
    const bool active = r0 < T;

    simdgroup_float8x8 q[ND], o[ND];
    for (uint d = 0; d < ND; d++) {
        o[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        // stored transposed ([d][row]) so S is computed as Sᵀ = K · Qᵀ with untransposed K loads
        if (active) simdgroup_load(q[d], qkv + r0 * stride + h * hd + d * 8, stride, ulong2(0, 0), true);
    }
    // softmax layout: lane handles row (lane / 4), columns (lane % 4) * 8 .. + 8
    const uint row = lane / 4, c0 = (lane % 4) * 8;
    const uint my_s = rs[sg * FQ + row], my_e = re[sg * FQ + row];
    float m_i = -INFINITY, l_i = 0.0f;

    for (uint kb = lo; kb < hi; kb += FK) {
        // ---- K tile [key][d]
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < FK * nv; i += nthreads) {
            uint key = i / nv, d = (i % nv) * 4;
            float4 x = kb + key < hi ? *((device const float4*)(qkv + (kb + key) * stride + (p.n_q + kvh) * hd + d)) : float4(0.0f);
            *((threadgroup float4*)(KV + key * ks + d)) = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            // FK/8 independent accumulators: interleaving them hides MMA latency
            simdgroup_float8x8 sacc[FK / 8];
            for (uint j = 0; j < FK / 8; j++) sacc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < ND; d++) {
                for (uint j = 0; j < FK / 8; j++) {
                    simdgroup_float8x8 ka;
                    simdgroup_load(ka, KV + (j * 8) * ks + d * 8, ks);
                    simdgroup_multiply_accumulate(sacc[j], ka, q[d], sacc[j]);
                }
            }
            for (uint j = 0; j < FK / 8; j++) simdgroup_store(sacc[j], Ss[sg] + j * 8, FK, ulong2(0, 0), true);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            // ---- online softmax, 4 lanes per row
            float v[8];
            float mx = -INFINITY;
            for (uint c = 0; c < 8; c++) {
                uint key = kb + c0 + c;
                bool ok = key >= my_s && key < my_e;
                v[c] = ok ? Ss[sg][row * FK + c0 + c] * p.scale : -INFINITY;
                mx = max(mx, v[c]);
            }
            mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
            mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
            float m_new = max(m_i, mx);
            float corr = m_new == -INFINITY ? 1.0f : exp(m_i - m_new);
            float sum = 0.0f;
            for (uint c = 0; c < 8; c++) {
                float e = v[c] == -INFINITY ? 0.0f : exp(v[c] - m_new);
                Ss[sg][row * FK + c0 + c] = e;
                sum += e;
            }
            sum += simd_shuffle_xor(sum, ushort(1));
            sum += simd_shuffle_xor(sum, ushort(2));
            l_i = l_i * corr + sum;
            m_i = m_new;
            for (uint e = lane; e < 64; e += 32) Ds[sg][e] = 0.0f;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane % 4 == 0) Ds[sg][row * 8 + row] = corr;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 dm;
            simdgroup_load(dm, Ds[sg], 8);
            for (uint d = 0; d < ND; d++) simdgroup_multiply(o[d], dm, o[d]);
        }
        // ---- V tile [key][d]
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < FK * nv; i += nthreads) {
            uint key = i / nv, d = (i % nv) * 4;
            float4 x = kb + key < hi ? *((device const float4*)(qkv + (kb + key) * stride + (p.n_q + p.n_kv + kvh) * hd + d)) : float4(0.0f);
            *((threadgroup float4*)(KV + key * ks + d)) = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint j = 0; j < FK / 8; j++) {
                simdgroup_float8x8 pm;
                simdgroup_load(pm, Ss[sg] + j * 8, FK);
                for (uint d = 0; d < ND; d++) {
                    simdgroup_float8x8 vm;
                    simdgroup_load(vm, KV + (j * 8) * ks + d * 8, ks);
                    simdgroup_multiply_accumulate(o[d], pm, vm, o[d]);
                }
            }
        }
    }
    if (!active) return;
    for (uint e = lane; e < 64; e += 32) Ds[sg][e] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane % 4 == 0) Ds[sg][row * 8 + row] = l_i > 0.0f ? 1.0f / l_i : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, Ds[sg], 8);
    for (uint d = 0; d < ND; d++) {
        simdgroup_multiply(o[d], dm, o[d]);
        simdgroup_store(o[d], out + r0 * (p.n_q * hd) + h * hd + d * 8, p.n_q * hd);
    }
}

template [[host_name("flash_attention_32")]] kernel void flash_attention<32>(device const float*, device const uint*, device const uint*, device float*, constant AttnP&, uint2, uint, uint, uint);
template [[host_name("flash_attention_64")]] kernel void flash_attention<64>(device const float*, device const uint*, device const uint*, device float*, constant AttnP&, uint2, uint, uint, uint);
template [[host_name("flash_attention_128")]] kernel void flash_attention<128>(device const float*, device const uint*, device const uint*, device float*, constant AttnP&, uint2, uint, uint, uint);

// ------------------------------------------------------------------ row softmax for matmul attention
// In place on S [rows × ld]: row r is query q0 + r of its sequence; causal rows see keys <= q0 + r.

struct SoftmaxP { uint cols, ld, q0, causal; float scale; };

kernel void softmax_rows(device float* S [[buffer(0)]],
                         constant SoftmaxP& p [[buffer(1)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    device float* x = S + row * p.ld;
    uint n = p.causal ? min(p.cols, p.q0 + row + 1) : p.cols;
    float mx = -INFINITY;
    for (uint i = tid; i < n; i += 256) mx = max(mx, x[i] * p.scale);
    mx = simd_max(mx);
    if (lane == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    for (uint i = 1; i < 8; i++) mx = max(mx, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = tid; i < n; i += 256) {
        float e = exp(x[i] * p.scale - mx);
        x[i] = e;
        sum += e;
    }
    float tot = block_sum(sum, red, lane, sg);
    float inv = 1.0f / tot;
    for (uint i = tid; i < p.cols; i += 256) x[i] = i < n ? x[i] * inv : 0.0f;
}

// ------------------------------------------------------------------ SwiGLU: [t, 2I] -> [t, I]

kernel void silu_mul(device const float* x [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint& I [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, t = gid.y;
    if (i >= I) return;
    float g = x[t * 2 * I + i];
    out[t * I + i] = g / (1.0f + exp(-g)) * x[t * 2 * I + I + i];
}
