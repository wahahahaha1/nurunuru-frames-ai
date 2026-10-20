// WGSL sources for NuruNuru Frames AI: frame ingest, RIFE v4.6 (IFNet) inference and rendering.
// Conventions:
//   * Tensors live in f32 storage buffers, planar [C][H][W], at the model resolution (mw x mh) or a
//     block's resolution (mw / scale x mh / scale).
//   * IFNet runs once per source pair at timestep 0.5 and yields, on the mid-time grid q:
//     f0 (q -> position in frame A), f1 (q -> position in frame B) in model pixels, and the blend mask m.
//   * The renderer receives them as textures: flow = (f0, f1) in UV units, mask.r = m.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});

  const fullscreenVS = /* wgsl */ `
struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4f(p[i], 0.0, 1.0);
  o.uv = vec2f((p[i].x + 1.0) * 0.5, (1.0 - p[i].y) * 0.5);
  return o;
}
`;

  // Video frame (texture_external) -> rgba8unorm frame texture, with a 2x2 tap box filter when downscaling.
  const ingest = fullscreenVS + /* wgsl */ `
struct IngestU { texel: vec2f, taps: f32, pad: f32 };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var ext: texture_external;
@group(0) @binding(2) var<uniform> U: IngestU;
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  if (U.taps < 1.5) {
    return vec4f(textureSampleBaseClampToEdge(ext, samp, in.uv).rgb, 1.0);
  }
  let d = U.texel * 0.25;
  let c = textureSampleBaseClampToEdge(ext, samp, in.uv + vec2f(-d.x, -d.y)).rgb
        + textureSampleBaseClampToEdge(ext, samp, in.uv + vec2f( d.x, -d.y)).rgb
        + textureSampleBaseClampToEdge(ext, samp, in.uv + vec2f(-d.x,  d.y)).rgb
        + textureSampleBaseClampToEdge(ext, samp, in.uv + vec2f( d.x,  d.y)).rgb;
  return vec4f(c * 0.25, 1.0);
}
`;

  // Frame texture -> RGB tensor at model resolution (box filter of taps x taps bilinear samples).
  const toTensor = /* wgsl */ `
struct TU { w: u32, h: u32, taps: u32, pad: u32 };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outT: array<f32>;
@group(0) @binding(3) var<uniform> U: TU;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.w || g.y >= U.h) { return; }
  let inv = vec2f(1.0 / f32(U.w), 1.0 / f32(U.h));
  let base = vec2f(g.xy) * inv;
  let n = U.taps;
  var s = vec3f(0.0);
  for (var j = 0u; j < n; j++) {
    for (var i = 0u; i < n; i++) {
      s += textureSampleLevel(img, samp, base + (vec2f(f32(i), f32(j)) + 0.5) / f32(n) * inv, 0.0).rgb;
    }
  }
  s = clamp(s / f32(n * n), vec3f(0.0), vec3f(1.0));
  let plane = U.w * U.h;
  let o = g.y * U.w + g.x;
  outT[o] = s.r;
  outT[plane + o] = s.g;
  outT[2u * plane + o] = s.b;
}
`;

  // Block input at the block's resolution: bilinear downscale (PyTorch align_corners=False) of
  //   first block: img0, img1, timestep                      (7 channels)
  //   later:       warp(img0, f0), warp(img1, f1), timestep, mask, (f0, f1) / scale   (12 channels)
  // Warping = bilinear sampling at x + flow with border clamp (grid_sample, align_corners=True).
  const prep = /* wgsl */ `
struct PU { mw: u32, mh: u32, bw: u32, bh: u32, scale: f32, first: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: PU;
@group(0) @binding(1) var<storage, read> img0: array<f32>;
@group(0) @binding(2) var<storage, read> img1: array<f32>;
@group(0) @binding(3) var<storage, read> fm: array<f32>;
@group(0) @binding(4) var<storage, read_write> outX: array<f32>;

fn at(buf: u32, c: u32, x: u32, y: u32) -> f32 {
  let i = (c * U.mh + y) * U.mw + x;
  if (buf == 0u) { return img0[i]; }
  return img1[i];
}

fn sampleImg(buf: u32, p: vec2f) -> vec3f {
  let q = clamp(p, vec2f(0.0), vec2f(f32(U.mw - 1u), f32(U.mh - 1u)));
  let x0 = u32(floor(q.x));
  let y0 = u32(floor(q.y));
  let x1 = min(x0 + 1u, U.mw - 1u);
  let y1 = min(y0 + 1u, U.mh - 1u);
  let a = q - floor(q);
  var o: vec3f;
  for (var c = 0u; c < 3u; c++) {
    let top = mix(at(buf, c, x0, y0), at(buf, c, x1, y0), a.x);
    let bot = mix(at(buf, c, x0, y1), at(buf, c, x1, y1), a.x);
    o[c] = mix(top, bot, a.y);
  }
  return o;
}

// 11 values of the block input before the timestep channel is inserted, at full-res pixel (x, y).
struct Px { w0: vec3f, w1: vec3f, mask: f32, flow: vec4f };
fn pixel(x: u32, y: u32) -> Px {
  var r: Px;
  let plane = U.mw * U.mh;
  let i = y * U.mw + x;
  if (U.first == 1u) {
    r.w0 = vec3f(img0[i], img0[plane + i], img0[2u * plane + i]);
    r.w1 = vec3f(img1[i], img1[plane + i], img1[2u * plane + i]);
    return r;
  }
  r.flow = vec4f(fm[i], fm[plane + i], fm[2u * plane + i], fm[3u * plane + i]);
  r.mask = fm[4u * plane + i];
  let p = vec2f(f32(x), f32(y));
  r.w0 = sampleImg(0u, p + r.flow.xy);
  r.w1 = sampleImg(1u, p + r.flow.zw);
  return r;
}

@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.bw || g.y >= U.bh) { return; }
  let src = max((vec2f(g.xy) + 0.5) * U.scale - 0.5, vec2f(0.0));
  let x0 = u32(floor(src.x));
  let y0 = u32(floor(src.y));
  let x1 = min(x0 + 1u, U.mw - 1u);
  let y1 = min(y0 + 1u, U.mh - 1u);
  let a = src - floor(src);
  let p00 = pixel(x0, y0);
  var w0 = p00.w0;
  var w1 = p00.w1;
  var mask = p00.mask;
  var flow = p00.flow;
  if (a.x > 0.0 || a.y > 0.0) {
    let p10 = pixel(x1, y0);
    let p01 = pixel(x0, y1);
    let p11 = pixel(x1, y1);
    w0 = mix(mix(p00.w0, p10.w0, a.x), mix(p01.w0, p11.w0, a.x), a.y);
    w1 = mix(mix(p00.w1, p10.w1, a.x), mix(p01.w1, p11.w1, a.x), a.y);
    mask = mix(mix(p00.mask, p10.mask, a.x), mix(p01.mask, p11.mask, a.x), a.y);
    flow = mix(mix(p00.flow, p10.flow, a.x), mix(p01.flow, p11.flow, a.x), a.y);
  }
  let plane = U.bw * U.bh;
  let o = g.y * U.bw + g.x;
  outX[o] = w0.r;
  outX[plane + o] = w0.g;
  outX[2u * plane + o] = w0.b;
  outX[3u * plane + o] = w1.r;
  outX[4u * plane + o] = w1.g;
  outX[5u * plane + o] = w1.b;
  outX[6u * plane + o] = 0.5;
  if (U.first == 0u) {
    outX[7u * plane + o] = mask;
    let f = flow / U.scale;
    outX[8u * plane + o] = f.x;
    outX[9u * plane + o] = f.y;
    outX[10u * plane + o] = f.z;
    outX[11u * plane + o] = f.w;
  }
}
`;

  // 3x3 convolution (padding 1, stride 1 or 2) + bias [+ residual] [+ LeakyReLU(0.2)].
  // Workgroup: TX x TY threads, each computing PX x PY pixels (spaced TX / TY apart) x 16 output
  // channels; the input tile (CI channels at a time) and its weights are staged in shared memory.
  // Split-K (small layers, to keep the GPU busy): workgroup z = cog + ncog * ks covers input
  // channels [ks * cinPer, (ks + 1) * cinPer) and, with partial = 1, writes its raw sums to slice
  // ks of the output; convReduce then adds the slices, bias, residual and activation.
  // Weights: vec4 index wOff + ((cog * cin + ci) * 9 + tap) * 4 + v holds output channels
  // cog * 16 + v * 4 + 0..3; bias: vec4 index bOff + cog * 4 + v.
  function conv({ stride, tx, ty, px, py, ci }) {
    const TW = tx * px;
    const TH = ty * py;
    const IW = (TW - 1) * stride + 3;
    const IH = (TH - 1) * stride + 3;
    const NT = tx * ty;
    const pix = [];
    for (let j = 0; j < py; j++) for (let i = 0; i < px; i++) pix.push([i, j]);
    const accDecl = pix.map((_, k) => `var a${k}0 = vec4f(0.0); var a${k}1 = vec4f(0.0); var a${k}2 = vec4f(0.0); var a${k}3 = vec4f(0.0);`).join('\n  ');
    const inner = pix
      .map(
        ([i, j], k) => `{
            let v = sIn[cb + (${j * ty}u + lid.y) * ${stride}u * ${IW}u + ky * ${IW}u + (${i * tx}u + lid.x) * ${stride}u + kx];
            a${k}0 += v * w0; a${k}1 += v * w1; a${k}2 += v * w2; a${k}3 += v * w3;
          }`
      )
      .join('\n          ');
    const store = pix
      .map(
        ([i, j], k) => `{
    let ox = ox0 + ${i * tx}u + lid.x;
    let oy = oy0 + ${j * ty}u + lid.y;
    if (ox < U.ow && oy < U.oh) {
      emit(a${k}0, 0u, ox, oy); emit(a${k}1, 1u, ox, oy); emit(a${k}2, 2u, ox, oy); emit(a${k}3, 3u, ox, oy);
    }
  }`
      )
      .join('\n  ');
    return /* wgsl */ `
struct CU {
  cin: u32, cout: u32, iw: u32, ih: u32, ow: u32, oh: u32, wOff: u32, bOff: u32,
  res: u32, leaky: u32, ncog: u32, cinPer: u32, partial: u32, pad0: u32, pad1: u32, pad2: u32,
};
@group(0) @binding(0) var<uniform> U: CU;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> wts: array<vec4f>;
@group(0) @binding(3) var<storage, read> resid: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;

var<workgroup> sIn: array<f32, ${ci * IW * IH}>;
var<workgroup> sW: array<vec4f, ${ci * 36}>;

var<private> cog: u32;
var<private> ks: u32;

fn emit(acc: vec4f, v: u32, ox: u32, oy: u32) {
  if (U.partial == 1u) {
    for (var e = 0u; e < 4u; e++) {
      let co = cog * 16u + v * 4u + e;
      outp[((ks * U.cout + co) * U.oh + oy) * U.ow + ox] = acc[e];
    }
    return;
  }
  let y = acc + wts[U.bOff + cog * 4u + v];
  for (var e = 0u; e < 4u; e++) {
    let co = cog * 16u + v * 4u + e;
    let o = (co * U.oh + oy) * U.ow + ox;
    var r = y[e];
    if (U.res == 1u) { r += resid[o]; }
    if (U.leaky == 1u) { r = max(r, 0.2 * r); }
    outp[o] = r;
  }
}

@compute @workgroup_size(${tx}, ${ty}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) tid: u32) {
  cog = wg.z % U.ncog;
  let kz = wg.z / U.ncog; // workgroup-uniform (the loop below contains barriers)
  ks = kz;
  let cBeg = kz * U.cinPer;
  let cEnd = min(U.cin, cBeg + U.cinPer);
  let ox0 = wg.x * ${TW}u;
  let oy0 = wg.y * ${TH}u;
  let ix0 = i32(ox0 * ${stride}u) - 1;
  let iy0 = i32(oy0 * ${stride}u) - 1;
  ${accDecl}
  for (var c0 = cBeg; c0 < cEnd; c0 += ${ci}u) {
    workgroupBarrier();
    for (var k = tid; k < ${ci * IW * IH}u; k += ${NT}u) {
      let c = c0 + k / ${IW * IH}u;
      let r = k % ${IW * IH}u;
      let x = ix0 + i32(r % ${IW}u);
      let y = iy0 + i32(r / ${IW}u);
      var v = 0.0;
      if (c < cEnd && x >= 0 && y >= 0 && x < i32(U.iw) && y < i32(U.ih)) {
        v = inp[(c * U.ih + u32(y)) * U.iw + u32(x)];
      }
      sIn[k] = v;
    }
    for (var k = tid; k < ${ci * 36}u; k += ${NT}u) {
      let c = c0 + k / 36u;
      var w = vec4f(0.0);
      if (c < cEnd) { w = wts[U.wOff + (cog * U.cin + c) * 36u + k % 36u]; }
      sW[k] = w;
    }
    workgroupBarrier();
    for (var cc = 0u; cc < ${ci}u; cc++) {
      let cb = cc * ${IW * IH}u;
      for (var ky = 0u; ky < 3u; ky++) {
        for (var kx = 0u; kx < 3u; kx++) {
          let wb = (cc * 9u + ky * 3u + kx) * 4u;
          let w0 = sW[wb];
          let w1 = sW[wb + 1u];
          let w2 = sW[wb + 2u];
          let w3 = sW[wb + 3u];
          ${inner}
        }
      }
    }
  }
  ${store}
}
`;
  }

  // Split-K epilogue: out = act(sum of ks partial slices + bias [+ residual]).
  const convReduce = /* wgsl */ `
struct RU { n: u32, plane: u32, ks: u32, res: u32, bOff: u32, leaky: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: RU;
@group(0) @binding(1) var<storage, read> part: array<f32>;
@group(0) @binding(2) var<storage, read> wts: array<vec4f>;
@group(0) @binding(3) var<storage, read> resid: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= U.n) { return; }
  let co = i / U.plane;
  var r = wts[U.bOff + co / 4u][co % 4u];
  for (var s = 0u; s < U.ks; s++) { r += part[s * U.n + i]; }
  if (U.res == 1u) { r += resid[i]; }
  if (U.leaky == 1u) { r = max(r, 0.2 * r); }
  outp[i] = r;
}
`;

  // ConvTranspose2d(cin, 24, 4, stride 2, padding 1) + PixelShuffle(2), only the 5 channels IFNet
  // uses (flow 4 + mask 1), at 4x the feature resolution. Weights: for (ci, sub, ky, kx) two vec4
  // holding output channels c * 4 + sub for c = 0..3 and c = 4; bias: two vec4 per sub.
  // Split-K: dispatch z = ks covers input channels [ks * cinPer, ...) and writes slice ks of the
  // output (bias in slice 0 only); accum adds the slices.
  const deconv = /* wgsl */ `
struct DU { cin: u32, fw: u32, fh: u32, wOff: u32, bOff: u32, cinPer: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> U: DU;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> wts: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outT: array<f32>;

@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  let ow = U.fw * 4u;
  let oh = U.fh * 4u;
  if (g.x >= ow || g.y >= oh) { return; }
  let x = g.x >> 1u;
  let y = g.y >> 1u;
  let sub = (g.y & 1u) * 2u + (g.x & 1u);
  // taps: y = 2 * iy - 1 + ky
  let my = i32(y >> 1u);
  let mx = i32(x >> 1u);
  var iy = array<i32, 2>(my, my - 1);
  var ky = array<u32, 2>(1u, 3u);
  if ((y & 1u) == 1u) { iy = array<i32, 2>(my, my + 1); ky = array<u32, 2>(2u, 0u); }
  var ix = array<i32, 2>(mx, mx - 1);
  var kx = array<u32, 2>(1u, 3u);
  if ((x & 1u) == 1u) { ix = array<i32, 2>(mx, mx + 1); kx = array<u32, 2>(2u, 0u); }
  let ks = g.z;
  var acc0 = vec4f(0.0);
  var acc1 = 0.0;
  if (ks == 0u) {
    acc0 = wts[U.bOff + sub * 2u];
    acc1 = wts[U.bOff + sub * 2u + 1u].x;
  }
  let cEnd = min(U.cin, (ks + 1u) * U.cinPer);
  for (var c = ks * U.cinPer; c < cEnd; c++) {
    let wc = U.wOff + (c * 4u + sub) * 32u;
    for (var j = 0; j < 2; j++) {
      if (iy[j] < 0 || iy[j] >= i32(U.fh)) { continue; }
      for (var i = 0; i < 2; i++) {
        if (ix[i] < 0 || ix[i] >= i32(U.fw)) { continue; }
        let v = feat[(c * U.fh + u32(iy[j])) * U.fw + u32(ix[i])];
        let wi = wc + (ky[j] * 4u + kx[i]) * 2u;
        acc0 += v * wts[wi];
        acc1 += v * wts[wi + 1u].x;
      }
    }
  }
  let plane = ow * oh;
  let o = ks * 5u * plane + g.y * ow + g.x;
  outT[o] = acc0.x;
  outT[plane + o] = acc0.y;
  outT[2u * plane + o] = acc0.z;
  outT[3u * plane + o] = acc0.w;
  outT[4u * plane + o] = acc1;
}
`;

  // Upsample a block's (flow, mask) output to model resolution (bilinear, align_corners=False) and
  // add it to the running estimate: flow += up(tmp) * scale, mask += up(tmp).
  const accum = /* wgsl */ `
struct AU { mw: u32, mh: u32, tw: u32, th: u32, scale: f32, first: u32, ks: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: AU;
@group(0) @binding(1) var<storage, read> tmp: array<f32>;
@group(0) @binding(2) var<storage, read_write> fm: array<f32>;

// sum of the deconv's split-K slices
fn T(i: u32) -> f32 {
  var v = 0.0;
  for (var s = 0u; s < U.ks; s++) { v += tmp[s * 5u * U.tw * U.th + i]; }
  return v;
}

@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.mw || g.y >= U.mh) { return; }
  let src = max((vec2f(g.xy) + 0.5) / U.scale - 0.5, vec2f(0.0));
  let x0 = min(u32(floor(src.x)), U.tw - 1u);
  let y0 = min(u32(floor(src.y)), U.th - 1u);
  let x1 = min(x0 + 1u, U.tw - 1u);
  let y1 = min(y0 + 1u, U.th - 1u);
  let a = src - floor(src);
  let tp = U.tw * U.th;
  let mp = U.mw * U.mh;
  let o = g.y * U.mw + g.x;
  for (var c = 0u; c < 5u; c++) {
    let b = c * tp;
    let top = mix(T(b + y0 * U.tw + x0), T(b + y0 * U.tw + x1), a.x);
    let bot = mix(T(b + y1 * U.tw + x0), T(b + y1 * U.tw + x1), a.x);
    var v = mix(top, bot, a.y);
    if (c < 4u) { v *= U.scale; }
    if (U.first == 0u) { v += fm[c * mp + o]; }
    fm[c * mp + o] = v;
  }
}
`;

  // Final (flow, mask) -> textures for the renderer, plus scene-change statistics: the share of
  // pixels whose two warped samples disagree (sampled on a 2x2 grid).
  const pack = /* wgsl */ `
struct KU { mw: u32, mh: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: KU;
@group(0) @binding(1) var<storage, read> fm: array<f32>;
@group(0) @binding(2) var<storage, read> img0: array<f32>;
@group(0) @binding(3) var<storage, read> img1: array<f32>;
@group(0) @binding(4) var flowOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var maskOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var<storage, read_write> stats: array<atomic<u32>, 4>;

var<workgroup> cnt: array<atomic<u32>, 2>;

fn lumAt(buf: u32, p: vec2f) -> f32 {
  let q = clamp(round(p), vec2f(0.0), vec2f(f32(U.mw - 1u), f32(U.mh - 1u)));
  let i = u32(q.y) * U.mw + u32(q.x);
  let plane = U.mw * U.mh;
  if (buf == 0u) { return 0.299 * img0[i] + 0.587 * img0[plane + i] + 0.114 * img0[2u * plane + i]; }
  return 0.299 * img1[i] + 0.587 * img1[plane + i] + 0.114 * img1[2u * plane + i];
}

@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) tid: u32) {
  if (tid < 2u) { atomicStore(&cnt[tid], 0u); }
  workgroupBarrier();
  if (g.x < U.mw && g.y < U.mh) {
    let plane = U.mw * U.mh;
    let i = g.y * U.mw + g.x;
    let f = vec4f(fm[i], fm[plane + i], fm[2u * plane + i], fm[3u * plane + i]);
    let m = 1.0 / (1.0 + exp(-fm[4u * plane + i]));
    let size = vec2f(f32(U.mw), f32(U.mh));
    textureStore(flowOut, g.xy, vec4f(f.xy / size, f.zw / size));
    let p = vec2f(g.xy);
    let e = abs(lumAt(0u, p + f.xy) - lumAt(1u, p + f.zw));
    textureStore(maskOut, g.xy, vec4f(m, e, 0.0, 1.0));
    if ((g.x & 1u) == 0u && (g.y & 1u) == 0u) {
      atomicAdd(&cnt[1], 1u);
      if (e > 0.1) { atomicAdd(&cnt[0], 1u); }
    }
  }
  workgroupBarrier();
  if (tid == 0u) {
    atomicAdd(&stats[0], atomicLoad(&cnt[0]));
    atomicAdd(&stats[1], atomicLoad(&cnt[1]));
  }
}
`;

  // Interpolated frame at phase t, rendered straight into the output (canvas or test texture).
  // A point at q on the mid-time grid sits at q + f0 in A, q + f1 in B and, assuming piecewise
  // linear motion through q at t = 0.5, at q + (1 - 2t) f0 (t < 0.5) or q + (2t - 1) f1 (t > 0.5).
  // For each output pixel p the mid-time point is found by fixed-point iteration; both frames are
  // then sampled there and blended with RIFE's mask, shifted towards the nearer frame. At t = 0.5
  // this is exactly RIFE's output.
  // mode: 0 interpolate, 1 frame A, 2 frame B, 3 plain blend, 4 flow visualisation
  const render = fullscreenVS + /* wgsl */ `
struct RU {
  rect: vec4f,
  t: f32,
  mode: u32,
  masking: f32,
  sceneFrac: f32,
  split: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};
@group(0) @binding(0) var<uniform> U: RU;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var texA: texture_2d<f32>;
@group(0) @binding(3) var texB: texture_2d<f32>;
@group(0) @binding(4) var flowT: texture_2d<f32>;
@group(0) @binding(5) var maskT: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> stats: array<u32, 4>;

fn A(p: vec2f) -> vec3f { return textureSampleLevel(texA, samp, p, 0.0).rgb; }
fn B(p: vec2f) -> vec3f { return textureSampleLevel(texB, samp, p, 0.0).rgb; }
fn Fl(p: vec2f) -> vec4f { return textureSampleLevel(flowT, samp, p, 0.0); }
fn Mk(p: vec2f) -> vec4f { return textureSampleLevel(maskT, samp, p, 0.0); }
fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }

fn hue(h: f32) -> vec3f {
  return clamp(abs(fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}

fn interp(p: vec2f, t: f32) -> vec3f {
  var q = p;
  if (t < 0.5) {
    let c = 1.0 - 2.0 * t;
    for (var i = 0; i < 4; i++) { q = p - c * Fl(q).xy; }
  } else {
    let c = 2.0 * t - 1.0;
    for (var i = 0; i < 4; i++) { q = p - c * Fl(q).zw; }
  }
  let f = Fl(q);
  let mk = Mk(q);
  let a = A(q + f.xy);
  let b = B(q + f.zw);
  let wa = mk.r * (1.0 - t);
  let wb = (1.0 - mk.r) * t;
  var col = (wa * a + wb * b) / max(wa + wb, 1e-6);
  // Optional artifact masking: fall back towards a plain blend where the two warps disagree.
  if (U.masking > 0.0) {
    let u = smoothstep(0.08, 0.25, abs(lum(a) - lum(b)));
    col = mix(col, mix(A(p), B(p), t), U.masking * u);
  }
  return col;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let p = (in.pos.xy - U.rect.xy) / U.rect.zw;
  if (p.x < 0.0 || p.y < 0.0 || p.x > 1.0 || p.y > 1.0) { return vec4f(0.0); }
  let t = U.t;
  var mode = U.mode;
  if (mode == 0u && stats[1] > 0u && f32(stats[0]) > U.sceneFrac * f32(stats[1])) { mode = select(2u, 1u, t < 0.5); }
  if (mode == 0u && in.pos.x < U.split) { mode = 1u; }
  var col: vec3f;
  switch mode {
    case 1u: { col = A(p); }
    case 2u: { col = B(p); }
    case 3u: { col = mix(A(p), B(p), t); }
    case 4u: {
      let f = Fl(p);
      let v = f.zw - f.xy;
      let m = clamp(length(v) * 25.0, 0.0, 1.0);
      col = mix(vec3f(lum(A(p))), hue(atan2(v.y, v.x) / 6.2831853 + 0.5), m * 0.8);
    }
    default: { col = interp(p, t); }
  }
  if (U.split > 0.0 && abs(in.pos.x - U.split) < 1.0) { col = vec3f(1.0); }
  return vec4f(col, 1.0);
}
`;

  S.WGSL = { ingest, toTensor, prep, conv, convReduce, deconv, accum, pack, render };
})();
