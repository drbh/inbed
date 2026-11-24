// CUDA kernels for inbed, compiled with NVRTC at startup. f16 weights and tables, f32 activations.
// No headers: NVRTC needs none of the CUDA toolkit's, and half conversions are inline PTX.
//
// Activations are row-major [tokens, cols] over a packed varlen batch (see backend.rs).

typedef unsigned short half_t;

__device__ __forceinline__ float h2f(half_t h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ half_t f2h(float f) {
    half_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the result.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    __shared__ float total;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) part[w] = v;
    __syncthreads();
    if (w == 0) {
        float s = lane < (int)(blockDim.x >> 5) ? part[lane] : 0.f;
        s = warp_sum(s);
        if (lane == 0) total = s;
    }
    __syncthreads();
    float r = total;
    __syncthreads();
    return r;
}

// out[t] = word[ids[t]] (+ ptab[pos[t]]) (+ ttab[0]); grid (ceil(cols/256), tokens)
extern "C" __global__ void embed(const unsigned* ids, const unsigned* pos, const half_t* word,
                                 const half_t* ptab, const half_t* ttab, float* out, int cols,
                                 int has_pos, int has_type) {
    int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float v = h2f(word[(size_t)ids[t] * cols + c]);
    if (has_pos) v += h2f(ptab[(size_t)pos[t] * cols + c]);
    if (has_type) v += h2f(ttab[c]);
    out[(size_t)t * cols + c] = v;
}

// f32 -> f16, the GEMM input (for tensors no kernel wrote an f16 copy of)
extern "C" __global__ void to_f16(const float* x, half_t* y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += step) y[i] = f2h(x[i]);
}

// f16 -> f32 (reading back a tensor that only exists in f16)
extern "C" __global__ void to_f32(const half_t* x, float* y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += step) y[i] = h2f(x[i]);
}

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.f + erff(x * 0.70710678118654752f));
}

// y = act(y + bias) + res, plus an f16 copy when y16 is set; grid (ceil(n/256), m)
extern "C" __global__ void epilogue(float* y, const float* bias, const float* res, half_t* y16,
                                    int n, int has_bias, int has_res, int act) {
    int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    size_t i = (size_t)r * n + c;
    float v = y[i];
    if (has_bias) v += bias[c];
    if (act) v = gelu(v);
    if (has_res) v += res[i];
    y[i] = v;
    if (y16) y16[i] = f2h(v);
}

// one block per row; writes f32 and the f16 copy the next matmul reads
extern "C" __global__ void rms_norm(const float* x, const float* g, float* out, half_t* out16,
                                    int cols, float eps) {
    const float* xr = x + (size_t)blockIdx.x * cols;
    float* yr = out + (size_t)blockIdx.x * cols;
    half_t* hr = out16 + (size_t)blockIdx.x * cols;
    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += xr[c] * xr[c];
    float r = rsqrtf(block_sum(s) / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = xr[c] * r * g[c];
        yr[c] = v;
        hr[c] = f2h(v);
    }
}

extern "C" __global__ void layer_norm(const float* x, const float* g, const float* b, float* out,
                                      half_t* out16, int cols, float eps) {
    const float* xr = x + (size_t)blockIdx.x * cols;
    float* yr = out + (size_t)blockIdx.x * cols;
    half_t* hr = out16 + (size_t)blockIdx.x * cols;
    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += xr[c];
    float mean = block_sum(s) / cols;
    float v = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = xr[c] - mean;
        v += d * d;
    }
    float r = rsqrtf(block_sum(v) / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float y = (xr[c] - mean) * r * g[c] + b[c];
        yr[c] = y;
        hr[c] = f2h(y);
    }
}

// cos/sin of position * inverse frequency, [len][hd/2] as (cos, sin)
extern "C" __global__ void rope_table(float2* cs, int len, int hd, float theta) {
    int half = hd / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len * half) return;
    int p = idx / half, i = idx % half;
    float inv = 1.f / powf(theta, (float)(2 * i) / (float)hd);
    float sn, c;
    sincosf((float)p * inv, &sn, &c);
    cs[idx] = make_float2(c, sn);
}

// In place on the fused qkv rows: per-head RMSNorm of the q and k heads, then rotate-half RoPE.
// One warp per (head, token), 4 heads per block; lane owns dims lane + 32*j. head_dim is a
// multiple of 64 (≤ 256), so a dim and its rotation partner (+ head_dim/2) live in the same lane.
extern "C" __global__ void qk_norm_rope(float* qkv, const unsigned* pos, const float* qn,
                                        const float* kn, const float2* table, int stride, int n_q,
                                        int n_heads, int hd, float eps) {
    int h = blockIdx.x * 4 + (threadIdx.x >> 5), t = blockIdx.y, lane = threadIdx.x & 31;
    if (h >= n_heads) return;
    const float* w = h < n_q ? qn : kn;
    float* x = qkv + (size_t)t * stride + h * hd;
    int nj = hd / 32;
    float v[8];
    float s = 0.f;
    for (int j = 0; j < nj; j++) {
        v[j] = x[lane + 32 * j];
        s += v[j] * v[j];
    }
    float r = rsqrtf(warp_sum(s) / hd + eps);
    for (int j = 0; j < nj; j++) v[j] = v[j] * r * w[lane + 32 * j];
    const float2* row = table + (size_t)pos[t] * (hd / 2);
    int hj = nj / 2;  // the partner of j is j + hj
    for (int j = 0; j < hj; j++) {
        float2 cs = row[lane + 32 * j];
        float a = v[j], b = v[j + hj];
        v[j] = a * cs.x - b * cs.y;
        v[j + hj] = b * cs.x + a * cs.y;
    }
    for (int j = 0; j < nj; j++) x[lane + 32 * j] = v[j];
}

// out = silu(x[:, :i]) * x[:, i:], in f16: only the down projection reads it; grid (ceil(i/256), rows)
extern "C" __global__ void silu_mul(const float* x, half_t* out, int i) {
    int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= i) return;
    const float* xr = x + (size_t)r * 2 * i;
    float g = xr[c];
    out[(size_t)r * i + c] = f2h(g / (1.f + expf(-g)) * xr[i + c]);
}

// Attention over a packed varlen batch, flash-style (online softmax), f32 math, f16 output (only
// the output projection reads it).
//
// Grid (tiles, n_q): a tile is QT consecutive queries of one sequence; `tiles` holds
// (sequence start, sequence length, first query offset) per tile. Keys and values come through
// shared memory in tiles of KT = 32. Each of the 4 warps owns QW queries: lane j scores key j
// against all QW of them (one float4 key load feeds 4·QW multiply-adds), then every lane
// accumulates its own output dims. Supports GQA (n_q a multiple of n_kv) and causal masking.
#define QW 8
#define QT (4 * QW)
#define KT 32
template <int HD>
__device__ void attention(const float* qkv, const unsigned* tiles, half_t* out, int n_q, int n_kv,
                          int causal, float scale) {
    constexpr int NI = HD / 32, KS = HD + 4;
    extern __shared__ float4 sm4[];
    float* qs = (float*)sm4;          // [QT][HD], pre-scaled
    float* ks = qs + QT * HD;         // [KT][KS]: the padding keeps float4 rows conflict-free
    float* vs = ks + KT * KS;         // [KT][HD]
    const unsigned* tl = tiles + 3 * blockIdx.x;
    const int s0 = tl[0], len = tl[1], q0 = tl[2];
    const int h = blockIdx.y, kvh = h / (n_q / n_kv);
    const size_t stride = (size_t)(n_q + 2 * n_kv) * HD;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nq = min(QT, len - q0);
    const int qb = warp * QW;  // this warp's first query, within the tile
    const float NEG_INF = __int_as_float(0xff800000);
    for (int i = threadIdx.x; i < QT * HD; i += blockDim.x) {
        int r = i / HD, d = i % HD;
        qs[i] = r < nq ? qkv[(size_t)(s0 + q0 + r) * stride + h * HD + d] * scale : 0.f;
    }
    float m[QW], l[QW], acc[QW][NI];
#pragma unroll
    for (int a = 0; a < QW; a++) {
        m[a] = -1e30f;
        l[a] = 0.f;
#pragma unroll
        for (int i = 0; i < NI; i++) acc[a][i] = 0.f;
    }
    const int kend = causal ? min(len, q0 + QT) : len;
    for (int k0 = 0; k0 < kend; k0 += KT) {
        __syncthreads();
        for (int i = threadIdx.x; i < KT * HD; i += blockDim.x) {
            int r = i / HD, d = i % HD, j = k0 + r;
            float kv = 0.f, vv = 0.f;
            if (j < kend) {
                const float* row = qkv + (size_t)(s0 + j) * stride;
                kv = row[(n_q + kvh) * HD + d];
                vv = row[(n_q + n_kv + kvh) * HD + d];
            }
            ks[r * KS + d] = kv;
            vs[r * HD + d] = vv;
        }
        __syncthreads();
        if (qb >= nq) continue;  // warp-uniform; the next iteration still reaches the barriers
        float s[QW];
#pragma unroll
        for (int a = 0; a < QW; a++) s[a] = 0.f;
        const float4* k4 = (const float4*)(ks + lane * KS);
#pragma unroll 4
        for (int d4 = 0; d4 < HD / 4; d4++) {
            const float4 kv = k4[d4];
#pragma unroll
            for (int a = 0; a < QW; a++) {
                const float4 qv = ((const float4*)(qs + (qb + a) * HD))[d4];
                s[a] += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            }
        }
        const int j = k0 + lane;
        float p[QW];
#pragma unroll
        for (int a = 0; a < QW; a++) {
            const bool valid = qb + a < nq && j < kend && (!causal || j <= q0 + qb + a);
            const float sv = valid ? s[a] : NEG_INF;
            const float mn = fmaxf(m[a], warp_max(sv));
            p[a] = valid ? expf(sv - mn) : 0.f;
            const float corr = expf(m[a] - mn);
            l[a] = l[a] * corr + warp_sum(p[a]);
#pragma unroll
            for (int i = 0; i < NI; i++) acc[a][i] *= corr;
            m[a] = mn;
        }
        for (int jj = 0; jj < KT; jj++) {
            float vv[NI];
#pragma unroll
            for (int i = 0; i < NI; i++) vv[i] = vs[jj * HD + lane + 32 * i];
#pragma unroll
            for (int a = 0; a < QW; a++) {
                const float pj = __shfl_sync(0xffffffffu, p[a], jj);
#pragma unroll
                for (int i = 0; i < NI; i++) acc[a][i] += pj * vv[i];
            }
        }
    }
    const size_t oc = (size_t)n_q * HD;
#pragma unroll
    for (int a = 0; a < QW; a++) {
        const int r = qb + a;
        if (r >= nq) continue;
        half_t* o = out + (size_t)(s0 + q0 + r) * oc + h * HD;
        const float inv = 1.f / l[a];
#pragma unroll
        for (int i = 0; i < NI; i++) o[lane + 32 * i] = f2h(acc[a][i] * inv);
    }
}

extern "C" __global__ void attention_32(const float* qkv, const unsigned* tiles, half_t* out,
                                        int n_q, int n_kv, int causal, float scale) {
    attention<32>(qkv, tiles, out, n_q, n_kv, causal, scale);
}
extern "C" __global__ void attention_64(const float* qkv, const unsigned* tiles, half_t* out,
                                        int n_q, int n_kv, int causal, float scale) {
    attention<64>(qkv, tiles, out, n_q, n_kv, causal, scale);
}
extern "C" __global__ void attention_128(const float* qkv, const unsigned* tiles, half_t* out,
                                         int n_q, int n_kv, int causal, float scale) {
    attention<128>(qkv, tiles, out, n_q, n_kv, causal, scale);
}

// ---------------------------------------------------------------- tensor-core attention
//
// The same computation on the tensor cores (mma.sync m16n8k16, f16 inputs, f32 accumulation),
// FlashAttention-2 style. A block is 4 warps over FQ = 64 queries of one sequence and one head;
// each warp owns 16 query rows. Per key tile of FK = 64: S = Q·Kᵀ (16×64 per warp) in registers,
// online softmax on S, then O += P·V with P re-packed from S's accumulator layout as the A
// operand. Q, K and Vᵀ are staged in shared memory as f16 (rows padded by 8 halves so the
// fragment loads are conflict-free). Q is pre-scaled by scale·log2(e), so softmax uses exp2.

__device__ __forceinline__ unsigned pack2(float a, float b) {
    return (unsigned)f2h(a) | ((unsigned)f2h(b) << 16);
}

__device__ __forceinline__ void mma16816(float* c, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

#define FQ 64
#define FK 64
template <int HD>
__device__ void fattn(const float* qkv, const unsigned* tiles, half_t* out, int n_q, int n_kv,
                      int causal, float scale) {
    constexpr int QS = HD + 8, VS = FK + 8, NT = HD / 8, KK = HD / 16;
    extern __shared__ float4 smf[];
    half_t* qs = (half_t*)smf;  // [FQ][QS]
    half_t* ks = qs + FQ * QS;  // [FK][QS]
    half_t* vt = ks + FK * QS;  // [HD][VS]: V transposed, so P·V's B fragments are contiguous
    const unsigned* tl = tiles + 3 * blockIdx.x;
    const int s0 = tl[0], len = tl[1], q0 = tl[2];
    const int h = blockIdx.y, kvh = h / (n_q / n_kv);
    const size_t stride = (size_t)(n_q + 2 * n_kv) * HD;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int nq = min(FQ, len - q0);
    const float NEG_INF = __int_as_float(0xff800000);
    const float qscale = scale * 1.4426950408889634f;
    for (int i = threadIdx.x; i < FQ * HD / 2; i += blockDim.x) {
        int r = i / (HD / 2), d = (i % (HD / 2)) * 2;
        float2 v = make_float2(0.f, 0.f);
        if (r < nq) v = *(const float2*)(qkv + (size_t)(s0 + q0 + r) * stride + h * HD + d);
        *(unsigned*)(qs + r * QS + d) = pack2(v.x * qscale, v.y * qscale);
    }
    float o[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; nt++) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    const int r0 = warp * 16 + g, r1 = r0 + 8;  // this thread's two query rows, in the tile
    const int kend = causal ? min(len, q0 + FQ) : len;
    for (int k0 = 0; k0 < kend; k0 += FK) {
        __syncthreads();
        for (int i = threadIdx.x; i < FK * HD / 2; i += blockDim.x) {
            int r = i / (HD / 2), d = (i % (HD / 2)) * 2, j = k0 + r;
            float2 kv = make_float2(0.f, 0.f), vv = make_float2(0.f, 0.f);
            if (j < kend) {
                const float* row = qkv + (size_t)(s0 + j) * stride;
                kv = *(const float2*)(row + (n_q + kvh) * HD + d);
                vv = *(const float2*)(row + (n_q + n_kv + kvh) * HD + d);
            }
            *(unsigned*)(ks + r * QS + d) = pack2(kv.x, kv.y);
            vt[d * VS + r] = f2h(vv.x);
            vt[(d + 1) * VS + r] = f2h(vv.y);
        }
        __syncthreads();
        // S = Q·Kᵀ for this warp's 16 rows × FK keys
        float s[FK / 8][4];
#pragma unroll
        for (int nt = 0; nt < FK / 8; nt++) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KK; kk++) {
            const half_t* qa = qs + r0 * QS + kk * 16 + 2 * t;
            unsigned a[4] = {*(const unsigned*)qa, *(const unsigned*)(qa + 8 * QS),
                             *(const unsigned*)(qa + 8), *(const unsigned*)(qa + 8 * QS + 8)};
#pragma unroll
            for (int nt = 0; nt < FK / 8; nt++) {
                const half_t* kb = ks + (nt * 8 + g) * QS + kk * 16 + 2 * t;
                unsigned b[2] = {*(const unsigned*)kb, *(const unsigned*)(kb + 8)};
                mma16816(s[nt], a, b);
            }
        }
        // mask, then online softmax over the row (a row's 64 values live in 4 lanes)
        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int nt = 0; nt < FK / 8; nt++) {
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int key = k0 + nt * 8 + 2 * t + c;
                const bool in = key < kend;
                if (!(in && (!causal || key <= q0 + r0))) s[nt][c] = NEG_INF;
                if (!(in && (!causal || key <= q0 + r1))) s[nt][2 + c] = NEG_INF;
                mx0 = fmaxf(mx0, s[nt][c]);
                mx1 = fmaxf(mx1, s[nt][2 + c]);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float c0 = exp2f(m0 - mn0), c1 = exp2f(m1 - mn1);
        m0 = mn0;
        m1 = mn1;
        float ps0 = 0.f, ps1 = 0.f;
        unsigned pa[FK / 16][4];
#pragma unroll
        for (int nt = 0; nt < FK / 8; nt++) {
            const float p00 = exp2f(s[nt][0] - mn0), p01 = exp2f(s[nt][1] - mn0);
            const float p10 = exp2f(s[nt][2] - mn1), p11 = exp2f(s[nt][3] - mn1);
            ps0 += p00 + p01;
            ps1 += p10 + p11;
            // two n8 accumulator tiles make one k16 A fragment
            pa[nt / 2][(nt & 1) * 2] = pack2(p00, p01);
            pa[nt / 2][(nt & 1) * 2 + 1] = pack2(p10, p11);
        }
        l0 = l0 * c0 + ps0;
        l1 = l1 * c1 + ps1;
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            o[nt][0] *= c0;
            o[nt][1] *= c0;
            o[nt][2] *= c1;
            o[nt][3] *= c1;
        }
        // O += P·V
#pragma unroll
        for (int kk = 0; kk < FK / 16; kk++) {
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const half_t* vb = vt + (nt * 8 + g) * VS + kk * 16 + 2 * t;
                unsigned b[2] = {*(const unsigned*)vb, *(const unsigned*)(vb + 8)};
                mma16816(o[nt], pa[kk], b);
            }
        }
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const size_t oc = (size_t)n_q * HD;
    const float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int d = h * HD + nt * 8 + 2 * t;
        if (r0 < nq) *(unsigned*)(out + (size_t)(s0 + q0 + r0) * oc + d) = pack2(o[nt][0] * i0, o[nt][1] * i0);
        if (r1 < nq) *(unsigned*)(out + (size_t)(s0 + q0 + r1) * oc + d) = pack2(o[nt][2] * i1, o[nt][3] * i1);
    }
}

extern "C" __global__ void fattn_32(const float* qkv, const unsigned* tiles, half_t* out, int n_q,
                                    int n_kv, int causal, float scale) {
    fattn<32>(qkv, tiles, out, n_q, n_kv, causal, scale);
}
extern "C" __global__ void fattn_64(const float* qkv, const unsigned* tiles, half_t* out, int n_q,
                                    int n_kv, int causal, float scale) {
    fattn<64>(qkv, tiles, out, n_q, n_kv, causal, scale);
}
extern "C" __global__ void fattn_128(const float* qkv, const unsigned* tiles, half_t* out, int n_q,
                                     int n_kv, int causal, float scale) {
    fattn<128>(qkv, tiles, out, n_q, n_kv, causal, scale);
}
