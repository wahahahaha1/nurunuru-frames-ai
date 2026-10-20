// WGSL sources for NuruNuru Frames AI: frame ingest, RIFE (IFNet, 4.25 family) inference and rendering.
// Conventions:
//   * Tensors live in f32 storage buffers, planar [C][H][W], at the model's padded resolution
//     (pw x ph, the picture in the top-left cw x ch) or a block's resolution (pw / scale x ph / scale).
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

  // Frame texture -> RGB planes of the frame tensor (box filter of taps x taps bilinear samples).
  // The tensor is pw x ph (the model's padded size); the picture fills the top-left cw x ch and the
  // rest is zero, like the zero padding of the reference implementation.
  const toTensor = /* wgsl */ `
struct TU { cw: u32, ch: u32, pw: u32, ph: u32, taps: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outT: array<f32>;
@group(0) @binding(3) var<uniform> U: TU;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.pw || g.y >= U.ph) { return; }
  var s = vec3f(0.0);
  if (g.x < U.cw && g.y < U.ch) {
    let inv = vec2f(1.0 / f32(U.cw), 1.0 / f32(U.ch));
    let base = vec2f(g.xy) * inv;
    let n = U.taps;
    for (var j = 0u; j < n; j++) {
      for (var i = 0u; i < n; i++) {
        s += textureSampleLevel(img, samp, base + (vec2f(f32(i), f32(j)) + 0.5) / f32(n) * inv, 0.0).rgb;
      }
    }
    s = clamp(s / f32(n * n), vec3f(0.0), vec3f(1.0));
  }
  let plane = U.pw * U.ph;
  let o = g.y * U.pw + g.x;
  outT[o] = s.r;
  outT[plane + o] = s.g;
  outT[2u * plane + o] = s.b;
}
`;

  // Block input at the block's resolution (ts = the timestep being inferred): bilinear downscale (PyTorch align_corners=False) of
  //   first block: img0, img1, feat0, feat1, timestep                                   (15 channels)
  //   later:  warp(img0, f0), warp(img1, f1), warp(feat0, f0), warp(feat1, f1), timestep,
  //           mask, block features, (f0, f1) / scale                                    (28 channels)
  // Frame tensors hold RGB (planes 0-2) and the Head features (planes 3-6). Warping = bilinear
  // sampling at x + flow with border clamp (grid_sample, align_corners=True).
  const prep = /* wgsl */ `
struct PU { pw: u32, ph: u32, bw: u32, bh: u32, scale: f32, first: u32, ts: f32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: PU;
@group(0) @binding(1) var<storage, read> img0: array<f32>;
@group(0) @binding(2) var<storage, read> img1: array<f32>;
@group(0) @binding(3) var<storage, read> fm: array<f32>;
@group(0) @binding(4) var<storage, read_write> outX: array<f32>;

fn at(buf: u32, i: u32) -> f32 {
  if (buf == 0u) { return img0[i]; }
  return img1[i];
}

// channels [c0, c0 + 4) of a frame tensor, bilinear at p with border clamp
fn sample4(buf: u32, c0: u32, p: vec2f) -> vec4f {
  let q = clamp(p, vec2f(0.0), vec2f(f32(U.pw - 1u), f32(U.ph - 1u)));
  let x0 = u32(floor(q.x));
  let y0 = u32(floor(q.y));
  let x1 = min(x0 + 1u, U.pw - 1u);
  let y1 = min(y0 + 1u, U.ph - 1u);
  let a = q - floor(q);
  let plane = U.pw * U.ph;
  var o = vec4f(0.0);
  for (var c = 0u; c < 4u; c++) {
    let b = (c0 + c) * plane;
    let top = mix(at(buf, b + y0 * U.pw + x0), at(buf, b + y0 * U.pw + x1), a.x);
    let bot = mix(at(buf, b + y1 * U.pw + x0), at(buf, b + y1 * U.pw + x1), a.x);
    o[c] = mix(top, bot, a.y);
  }
  return o;
}

fn pixel(x: u32, y: u32) -> array<f32, 28> {
  var v: array<f32, 28>;
  let plane = U.pw * U.ph;
  let i = y * U.pw + x;
  var flow = vec4f(0.0);
  var a = vec4f(0.0);
  var b = vec4f(0.0);
  var fa = vec4f(0.0);
  var fb = vec4f(0.0);
  if (U.first == 1u) {
    for (var c = 0u; c < 4u; c++) {
      a[c] = select(0.0, img0[c * plane + i], c < 3u);
      b[c] = select(0.0, img1[c * plane + i], c < 3u);
      fa[c] = img0[(3u + c) * plane + i];
      fb[c] = img1[(3u + c) * plane + i];
    }
  } else {
    flow = vec4f(fm[i], fm[plane + i], fm[2u * plane + i], fm[3u * plane + i]);
    let p = vec2f(f32(x), f32(y));
    a = sample4(0u, 0u, p + flow.xy);
    b = sample4(1u, 0u, p + flow.zw);
    fa = sample4(0u, 3u, p + flow.xy);
    fb = sample4(1u, 3u, p + flow.zw);
  }
  for (var c = 0u; c < 3u; c++) { v[c] = a[c]; v[3u + c] = b[c]; }
  for (var c = 0u; c < 4u; c++) { v[6u + c] = fa[c]; v[10u + c] = fb[c]; }
  v[14] = U.ts;
  if (U.first == 0u) {
    for (var c = 0u; c < 9u; c++) { v[15u + c] = fm[(4u + c) * plane + i]; } // mask + 8 features
    for (var c = 0u; c < 4u; c++) { v[24u + c] = flow[c]; }
  }
  return v;
}

@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.bw || g.y >= U.bh) { return; }
  let src = max((vec2f(g.xy) + 0.5) * U.scale - 0.5, vec2f(0.0));
  let x0 = u32(floor(src.x));
  let y0 = u32(floor(src.y));
  let x1 = min(x0 + 1u, U.pw - 1u);
  let y1 = min(y0 + 1u, U.ph - 1u);
  let a = src - floor(src);
  var v = pixel(x0, y0);
  if (a.x > 0.0 || a.y > 0.0) {
    let v10 = pixel(x1, y0);
    let v01 = pixel(x0, y1);
    let v11 = pixel(x1, y1);
    for (var c = 0u; c < 28u; c++) { v[c] = mix(mix(v[c], v10[c], a.x), mix(v01[c], v11[c], a.x), a.y); }
  }
  let n = select(28u, 15u, U.first == 1u);
  let plane = U.bw * U.bh;
  let o = g.y * U.bw + g.x;
  for (var c = 0u; c < n; c++) {
    var r = v[c];
    if (c >= 24u) { r = r / U.scale; }
    outX[c * plane + o] = r;
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
      if (co >= U.cout) { break; }
      outp[((ks * U.cout + co) * U.oh + oy) * U.ow + ox] = acc[e];
    }
    return;
  }
  let y = acc + wts[U.bOff + cog * 4u + v];
  for (var e = 0u; e < 4u; e++) {
    let co = cog * 16u + v * 4u + e;
    if (co >= U.cout) { break; }
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

  // ConvTranspose2d(cin, 4 * NC, 4, stride 2, padding 1) + PixelShuffle(2) -> NC channels at 4x the
  // feature resolution (IFBlock.lastconv: flow 4, mask 1, features 8). Weights: for (ci, sub, ky, kx)
  // NV = ceil(NC / 4) vec4 holding output channels c * 4 + sub; bias: NV vec4 per sub.
  // Split-K: dispatch z = ks covers input channels [ks * cinPer, ...) and writes slice ks of the
  // output (bias in slice 0 only); accum adds the slices.
  function deconv(NC) {
    const NV = Math.ceil(NC / 4);
    const vs = [...Array(NV).keys()];
    return /* wgsl */ `
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
  ${vs.map((v) => `var acc${v} = vec4f(0.0);`).join(' ')}
  if (ks == 0u) {
    ${vs.map((v) => `acc${v} = wts[U.bOff + sub * ${NV}u + ${v}u];`).join(' ')}
  }
  let cEnd = min(U.cin, (ks + 1u) * U.cinPer);
  for (var c = ks * U.cinPer; c < cEnd; c++) {
    let wc = U.wOff + (c * 4u + sub) * ${16 * NV}u;
    for (var j = 0; j < 2; j++) {
      if (iy[j] < 0 || iy[j] >= i32(U.fh)) { continue; }
      for (var i = 0; i < 2; i++) {
        if (ix[i] < 0 || ix[i] >= i32(U.fw)) { continue; }
        let v = feat[(c * U.fh + u32(iy[j])) * U.fw + u32(ix[i])];
        let wi = wc + (ky[j] * 4u + kx[i]) * ${NV}u;
        ${vs.map((v) => `acc${v} += v * wts[wi + ${v}u];`).join(' ')}
      }
    }
  }
  let plane = ow * oh;
  let o = ks * ${NC}u * plane + g.y * ow + g.x;
  ${[...Array(NC).keys()].map((c) => `outT[o + ${c}u * plane] = acc${c >> 2}[${c & 3}];`).join('\n  ')}
}
`;
  }

  // Plain ConvTranspose2d(cin, 4, 4, stride 2, padding 1) (Head.cnn3): output 2fw x 2fh x 4 channels,
  // written at float offset oOff (the feature planes of a frame tensor). Weights: one vec4 per (ci, ky, kx).
  const deconvPlain = /* wgsl */ `
struct EU { cin: u32, fw: u32, fh: u32, wOff: u32, bOff: u32, oOff: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> U: EU;
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> wts: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outT: array<f32>;

@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  let ow = U.fw * 2u;
  let oh = U.fh * 2u;
  if (g.x >= ow || g.y >= oh) { return; }
  let my = i32(g.y >> 1u);
  let mx = i32(g.x >> 1u);
  var iy = array<i32, 2>(my, my - 1);
  var ky = array<u32, 2>(1u, 3u);
  if ((g.y & 1u) == 1u) { iy = array<i32, 2>(my, my + 1); ky = array<u32, 2>(2u, 0u); }
  var ix = array<i32, 2>(mx, mx - 1);
  var kx = array<u32, 2>(1u, 3u);
  if ((g.x & 1u) == 1u) { ix = array<i32, 2>(mx, mx + 1); kx = array<u32, 2>(2u, 0u); }
  var acc = wts[U.bOff];
  for (var c = 0u; c < U.cin; c++) {
    for (var j = 0; j < 2; j++) {
      if (iy[j] < 0 || iy[j] >= i32(U.fh)) { continue; }
      for (var i = 0; i < 2; i++) {
        if (ix[i] < 0 || ix[i] >= i32(U.fw)) { continue; }
        let v = feat[(c * U.fh + u32(iy[j])) * U.fw + u32(ix[i])];
        acc += v * wts[U.wOff + c * 16u + ky[j] * 4u + kx[i]];
      }
    }
  }
  let plane = ow * oh;
  let o = U.oOff + g.y * ow + g.x;
  outT[o] = acc.x;
  outT[o + plane] = acc.y;
  outT[o + 2u * plane] = acc.z;
  outT[o + 3u * plane] = acc.w;
}
`;

  // Upsample a block's output (NC channels, ks split-K slices) to model resolution (bilinear,
  // align_corners=False) into the running state fm: flow (0-3) += up(tmp) * scale; mask (4) and
  // block features (5-12) are replaced.
  const accum = /* wgsl */ `
struct AU { pw: u32, ph: u32, tw: u32, th: u32, scale: f32, first: u32, ks: u32, nc: u32 };
@group(0) @binding(0) var<uniform> U: AU;
@group(0) @binding(1) var<storage, read> tmp: array<f32>;
@group(0) @binding(2) var<storage, read_write> fm: array<f32>;

// sum of the deconv's split-K slices
fn T(i: u32) -> f32 {
  var v = 0.0;
  for (var s = 0u; s < U.ks; s++) { v += tmp[s * U.nc * U.tw * U.th + i]; }
  return v;
}

@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.pw || g.y >= U.ph) { return; }
  let src = max((vec2f(g.xy) + 0.5) / U.scale - 0.5, vec2f(0.0));
  let x0 = min(u32(floor(src.x)), U.tw - 1u);
  let y0 = min(u32(floor(src.y)), U.th - 1u);
  let x1 = min(x0 + 1u, U.tw - 1u);
  let y1 = min(y0 + 1u, U.th - 1u);
  let a = src - floor(src);
  let tp = U.tw * U.th;
  let mp = U.pw * U.ph;
  let o = g.y * U.pw + g.x;
  for (var c = 0u; c < U.nc; c++) {
    let b = c * tp;
    let top = mix(T(b + y0 * U.tw + x0), T(b + y0 * U.tw + x1), a.x);
    let bot = mix(T(b + y1 * U.tw + x0), T(b + y1 * U.tw + x1), a.x);
    var v = mix(top, bot, a.y);
    if (c < 4u) {
      v *= U.scale;
      if (U.first == 0u) { v += fm[c * mp + o]; }
    }
    fm[c * mp + o] = v;
  }
}
`;

  // Final (flow, mask) of the picture area -> textures for the renderer, plus scene-change statistics:
  // the share of pixels whose two warped samples disagree (sampled on a 2x2 grid).
  const pack = /* wgsl */ `
struct KU { cw: u32, ch: u32, pw: u32, ph: u32 };
@group(0) @binding(0) var<uniform> U: KU;
@group(0) @binding(1) var<storage, read> fm: array<f32>;
@group(0) @binding(2) var<storage, read> img0: array<f32>;
@group(0) @binding(3) var<storage, read> img1: array<f32>;
@group(0) @binding(4) var flowOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var maskOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var<storage, read_write> stats: array<atomic<u32>, 4>;

var<workgroup> cnt: array<atomic<u32>, 2>;

fn lumAt(buf: u32, p: vec2f) -> f32 {
  let q = clamp(round(p), vec2f(0.0), vec2f(f32(U.cw - 1u), f32(U.ch - 1u)));
  let i = u32(q.y) * U.pw + u32(q.x);
  let plane = U.pw * U.ph;
  if (buf == 0u) { return 0.299 * img0[i] + 0.587 * img0[plane + i] + 0.114 * img0[2u * plane + i]; }
  return 0.299 * img1[i] + 0.587 * img1[plane + i] + 0.114 * img1[2u * plane + i];
}

@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) tid: u32) {
  if (tid < 2u) { atomicStore(&cnt[tid], 0u); }
  workgroupBarrier();
  if (g.x < U.cw && g.y < U.ch) {
    let plane = U.pw * U.ph;
    let i = g.y * U.pw + g.x;
    let f = vec4f(fm[i], fm[plane + i], fm[2u * plane + i], fm[3u * plane + i]);
    let m = 1.0 / (1.0 + exp(-fm[4u * plane + i]));
    let size = vec2f(f32(U.cw), f32(U.ch));
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

  // Brightness changes (fades, flashes): the mean luma ratio k = B / A of the pair. Samples of A and B
  // are scaled to the brightness expected at time t (linear in between: A * (1 + (k - 1) t)) before
  // they are compared or blended. Needs bindings meanA / meanB and sets gA / gB.
  const gain = /* wgsl */ `
var<private> gA: f32 = 1.0;
var<private> gB: f32 = 1.0;
fn setGain(t: f32) {
  let ma = f32(meanA[0]) / max(f32(meanA[1]), 1.0);
  let mb = f32(meanB[0]) / max(f32(meanB[1]), 1.0);
  var k = clamp((mb + 1.0) / (ma + 1.0), 0.33, 3.0); // (+1: sums are luma * 256, keeps black stable)
  // ignore changes within 1 % (content moving in and out at the edges shifts the mean a little)
  k = 1.0 + sign(k - 1.0) * max(abs(k - 1.0) - 0.01, 0.0);
  gA = 1.0 + (k - 1.0) * t;
  gB = gA / k;
}
fn Ag(p: vec2f) -> vec3f { return A(p) * gA; }
fn Bg(p: vec2f) -> vec3f { return B(p) * gB; }
`;

  // Block matching at phase t (NuruNuru Frames' renderer): the forward / backward block fields are
  // aligned to t by fixed-point iteration, both frames are warped along each and the two
  // predictions are weighted by how well they agree; where neither agrees (occlusion) the nearer,
  // one-sided sample is taken. Returns (colour, photometric error).
  const bmInterp = /* wgsl */ `
fn Fb(p: vec2f) -> vec4f { return textureSampleLevel(fieldF, samp, p * U.fieldScale, 0.0); }
fn Gb(p: vec2f) -> vec4f { return textureSampleLevel(fieldG, samp, p * U.fieldScale, 0.0); }

var<private> bmErr2: f32 = 0.0; // set by bmInterp: error of the worse direction (both must match)

fn bmInterp(p: vec2f, t: f32) -> vec4f {
  var v = Fb(p).xy;
  for (var i = 0; i < 3; i++) { v = Fb(p - t * v).xy; }
  var w = Gb(p).xy;
  for (var i = 0; i < 3; i++) { w = Gb(p - (1.0 - t) * w).xy; }
  let fq = Fb(p - t * v);
  let gq = Gb(p - (1.0 - t) * w);
  let a0 = Ag(p - t * v);
  let b0 = Bg(p + (1.0 - t) * v);
  let b1 = Bg(p - (1.0 - t) * w);
  let a1 = Ag(p + t * w);
  let pF = abs(lum(a0) - lum(b0));
  let pG = abs(lum(a1) - lum(b1));
  let eF = pF + 0.5 * fq.z;
  let eG = pG + 0.5 * gq.z;
  let wF = (1.0 - t + 0.05) * exp(-eF * 30.0);
  let wG = (t + 0.05) * exp(-eG * 30.0);
  var col = (wF * mix(a0, b0, t) + wG * mix(a1, b1, t)) / max(wF + wG, 1e-6);
  let bad = smoothstep(0.06, 0.16, min(eF, eG));
  var side: vec3f;
  if (t < 0.5) { side = select(a1, a0, fq.z <= gq.z); } else { side = select(b1, b0, fq.z <= gq.z); }
  col = mix(col, side, bad);
  bmErr2 = max(pF, pG);
  return vec4f(col, min(pF, pG));
}
`;

  // Per pair, on the mid-time grid (model size): which motion to trust where. Measures, over +-4 px
  // (5x5 samples), how well each explains the two frames at t = 0.5 (photometric error of its two
  // warps) along with local motion and structure cues, and turns them into r = weight of block
  // matching (a fitted logistic model), g = confidence (low where both disagree).
  const select = /* wgsl */ `
struct SU { fieldScale: vec2f, size: vec2f, useHalf: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> U: SU;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var texA: texture_2d<f32>;
@group(0) @binding(3) var texB: texture_2d<f32>;
@group(0) @binding(4) var flowT: texture_2d<f32>;
@group(0) @binding(5) var fieldF: texture_2d<f32>;
@group(0) @binding(6) var fieldG: texture_2d<f32>;
@group(0) @binding(7) var selOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var maskT: texture_2d<f32>;
@group(0) @binding(9) var flowH: texture_2d<f32>;
@group(0) @binding(10) var maskH: texture_2d<f32>;
@group(0) @binding(11) var<storage, read> meanA: array<u32, 2>;
@group(0) @binding(12) var<storage, read> meanB: array<u32, 2>;
@group(0) @binding(13) var<storage, read_write> stats: array<atomic<u32>, 4>;
${gain}
var<workgroup> cnt: array<atomic<u32>, 2>;

// photometric disagreement of a RIFE result's two warps at p: (error not counted where its mask
// picks one frame only (occlusion: the other warp is expected to disagree), one-sidedness, raw error)
fn rifeErr(p: vec2f, fT: texture_2d<f32>, mT: texture_2d<f32>) -> vec3f {
  let f = textureSampleLevel(fT, samp, p, 0.0);
  let side = abs(2.0 * textureSampleLevel(mT, samp, p, 0.0).r - 1.0);
  let e = abs(lum(Ag(p + f.xy)) - lum(Bg(p + f.zw)));
  return vec3f(e * (1.0 - side), side, e);
}

fn A(p: vec2f) -> vec3f { return textureSampleLevel(texA, samp, p, 0.0).rgb; }
fn B(p: vec2f) -> vec3f { return textureSampleLevel(texB, samp, p, 0.0).rgb; }
fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }
${bmInterp}
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) tid: u32) {
  // scene-change statistics (brightness-compensated): share of pixels whose two warps disagree
  setGain(0.5);
  if (tid < 2u) { atomicStore(&cnt[tid], 0u); }
  workgroupBarrier();
  let inside = f32(g.x) < U.size.x && f32(g.y) < U.size.y;
  if (inside && (g.x & 1u) == 0u && (g.y & 1u) == 0u) {
    let pc = (vec2f(g.xy) + 0.5) / U.size;
    let fc = textureSampleLevel(flowT, samp, pc, 0.0);
    atomicAdd(&cnt[1], 1u);
    if (abs(lum(Ag(pc + fc.xy)) - lum(Bg(pc + fc.zw))) > 0.1) { atomicAdd(&cnt[0], 1u); }
  }
  workgroupBarrier();
  if (tid == 0u) {
    atomicAdd(&stats[0], atomicLoad(&cnt[0]));
    atomicAdd(&stats[1], atomicLoad(&cnt[1]));
  }
  if (!inside) { return; }
  let p0 = (vec2f(g.xy) + 0.5) / U.size;
  var eA = 0.0;
  var eAraw = 0.0;
  var eH = 0.0;
  var eB = 0.0;
  var eB2 = 0.0; // block matching, both directions
  var occ = 0.0;
  var sq = vec3f(0.0); // squared errors (RIFE masked, RIFE raw, block matching): local breakdowns
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let p = p0 + vec2f(f32(dx), f32(dy)) * 2.0 / U.size;
      let ra = rifeErr(p, flowT, maskT);
      let eb = bmInterp(p, 0.5).w;
      let eb2 = bmErr2;
      eA += ra.x;
      eAraw += ra.z;
      occ = max(occ, ra.y);
      if (U.useHalf > 0.0) { eH += rifeErr(p, flowH, maskH).x; }
      eB += eb;
      eB2 += eb2;
      sq += vec3f(ra.x * ra.x, ra.z * ra.z, eb2 * eb2);
    }
  }
  eA /= 25.0;
  eAraw /= 25.0;
  eH /= 25.0;
  eB /= 25.0;
  eB2 /= 25.0;
  // a true occlusion defeats block matching in one direction; where block matching explains both
  // frames in both directions, RIFE's one-sided mask is not an excuse and its full disagreement counts
  let bmGood = 1.0 - smoothstep(0.004, 0.015, eB2);
  eA = mix(eA, eAraw, bmGood);
  // local RMS errors (RIFE breaking down, e.g. ghosted text, shows as RMS well above its mean)
  let rmsA = sqrt(mix(sq.x, sq.y, bmGood) / 25.0);
  let rmsB = sqrt(sq.z / 25.0);
  // half-resolution RIFE where it explains the frames clearly better (large motion)
  let wH = smoothstep(0.002, 0.01, eA - eH) * U.useHalf;
  eA = mix(eA, eH, wH);
  // how much either motion varies within +-4 px (motion boundaries: block matching's blocks straddle
  // them) and how much structure the picture has (how informative the errors above are)
  var lo = vec4f(1e9);
  var hi = vec4f(-1e9);
  var tex = 0.0;
  let e1 = vec2f(1.0) / U.size;
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let p = p0 + vec2f(f32(dx), f32(dy)) * 2.0 / U.size;
      let f = textureSampleLevel(flowT, samp, p, 0.0);
      let m = vec4f(f.zw - f.xy, Fb(p).xy) * vec4f(U.size, U.size);
      lo = min(lo, m);
      hi = max(hi, m);
      tex += abs(lum(A(p + vec2f(e1.x, 0.0))) - lum(A(p - vec2f(e1.x, 0.0)))) + abs(lum(A(p + vec2f(0.0, e1.y))) - lum(A(p - vec2f(0.0, e1.y))));
    }
  }
  tex = smoothstep(0.01, 0.04, tex / 25.0); // 0 = flat, errors say little
  // Weight of block matching: logistic model fitted on the synthetic test scenes (maximising their mean
  // PSNR without making any scene worse than the previous hand-tuned rule). Pixel quantities are
  // measured on a 640-wide grid. Inputs: the errors above, RIFE's one-sidedness, structure, how much
  // either motion varies nearby, the motion lengths, RIFE's mask, block matching's match cost and the
  // distance to the picture edge in motions (content entering the picture).
  let px = 640.0 / U.size.x;
  let fc0 = textureSampleLevel(flowT, samp, p0, 0.0);
  let fb0 = Fb(p0);
  let magR = min(length((fc0.zw - fc0.xy) * U.size) * px, 64.0);
  let magB = min(length(fb0.xy * U.size) * px, 64.0);
  let edge = min(min(p0.x, 1.0 - p0.x) * U.size.x, min(p0.y, 1.0 - p0.y) * U.size.y) * px;
  let lg = vec4f(log(eA + 1e-3), log(eAraw + 1e-3), log(eB + 1e-3), log(eB2 + 1e-3));
  let lr = vec2f(log(rmsA + 1e-3), log(rmsB + 1e-3));
  let z = -4.5558 + dot(lg, vec4f(-0.5027, 1.2137, -0.1363, -0.7874)) + 0.8933 * occ + dot(lr, vec2f(0.3583, -0.5895))
    + 1.2700 * tex + 0.0366 * log(1.0 + max(hi.x - lo.x, hi.y - lo.y) * px) - 0.2750 * log(1.0 + max(hi.z - lo.z, hi.w - lo.w) * px)
    - 0.7804 * log(1.0 + magR) + 0.2947 * log(1.0 + magB)
    - 0.3030 * abs(2.0 * textureSampleLevel(maskT, samp, p0, 0.0).r - 1.0) - 0.2767 * log(min(fb0.z, 1.0) + 1e-3)
    + 0.5601 * log(1.0 + edge / (max(magR, magB) + 1.0));
  let wBM = 1.0 / (1.0 + exp(-z));
  let conf = 1.0 - smoothstep(0.05, 0.15, min(eA, eB));
  textureStore(selOut, g.xy, vec4f(wBM, conf, wH, 1.0));
}
`;

  // Interpolated frame at phase t, rendered straight into the output (canvas or test texture).
  // A point at q on the mid-time grid sits at q + f0 in A, q + f1 in B and, assuming piecewise
  // linear motion through q at t = 0.5, at q + (1 - 2t) f0 (t < 0.5) or q + (2t - 1) f1 (t > 0.5).
  // For each output pixel p the mid-time point is found by fixed-point iteration; both frames are
  // then sampled there and blended with RIFE's mask, shifted towards the nearer frame. At t = 0.5
  // this is exactly RIFE's output.
  // With block matching (hybrid > 0) its result replaces RIFE's where the selection map says so.
  // Masking: where neither motion is trusted, fall back towards the nearer original frame (no ghosting).
  // mode: 0 interpolate, 1 frame A, 2 frame B, 3 plain blend, 4 flow visualisation, 5 selection map
  const render = fullscreenVS + /* wgsl */ `
struct RU {
  rect: vec4f,
  fieldScale: vec2f,
  t: f32,
  mode: u32,
  masking: f32,
  sceneFrac: f32,
  split: f32,
  hybrid: f32, // weight of block matching where selected; < 0: no block matching / selection map
  tau: f32, // timestep of the extra RIFE anchor bound as flowX / maskX (0.5 = none)
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
@group(0) @binding(7) var fieldF: texture_2d<f32>;
@group(0) @binding(8) var fieldG: texture_2d<f32>;
@group(0) @binding(9) var selT: texture_2d<f32>;
@group(0) @binding(10) var flowX: texture_2d<f32>;
@group(0) @binding(11) var maskX: texture_2d<f32>;
@group(0) @binding(12) var flowH: texture_2d<f32>;
@group(0) @binding(13) var maskH: texture_2d<f32>;
@group(0) @binding(14) var<storage, read> meanA: array<u32, 2>;
@group(0) @binding(15) var<storage, read> meanB: array<u32, 2>;

fn A(p: vec2f) -> vec3f { return textureSampleLevel(texA, samp, p, 0.0).rgb; }
fn B(p: vec2f) -> vec3f { return textureSampleLevel(texB, samp, p, 0.0).rgb; }
// Catmull-Rom (bicubic) sample from 9 bilinear taps: keeps warped frames as sharp as the originals
// (bilinear warping softens them, which shows as a pulsing sharpness at high frame rates).
fn sharp(tex: texture_2d<f32>, uv: vec2f) -> vec3f {
  let size = vec2f(textureDimensions(tex));
  let sp = uv * size;
  let tc = floor(sp - 0.5) + 0.5;
  let f = sp - tc;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let t0 = (tc - 1.0) / size;
  let t3 = (tc + 2.0) / size;
  let t12 = (tc + w2 / w12) / size;
  var r = textureSampleLevel(tex, samp, vec2f(t0.x, t0.y), 0.0).rgb * (w0.x * w0.y);
  r += textureSampleLevel(tex, samp, vec2f(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y);
  r += textureSampleLevel(tex, samp, vec2f(t3.x, t0.y), 0.0).rgb * (w3.x * w0.y);
  r += textureSampleLevel(tex, samp, vec2f(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y);
  r += textureSampleLevel(tex, samp, vec2f(t12.x, t12.y), 0.0).rgb * (w12.x * w12.y);
  r += textureSampleLevel(tex, samp, vec2f(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y);
  r += textureSampleLevel(tex, samp, vec2f(t0.x, t3.y), 0.0).rgb * (w0.x * w3.y);
  r += textureSampleLevel(tex, samp, vec2f(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
  r += textureSampleLevel(tex, samp, vec2f(t3.x, t3.y), 0.0).rgb * (w3.x * w3.y);
  return clamp(r, vec3f(0.0), vec3f(1.0));
}
fn Fl(p: vec2f) -> vec4f { return textureSampleLevel(flowT, samp, p, 0.0); }
fn Mk(p: vec2f) -> vec4f { return textureSampleLevel(maskT, samp, p, 0.0); }
fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }
${gain}
${bmInterp}
fn hue(h: f32) -> vec3f {
  return clamp(abs(fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}

// motion of a mid-time point towards the nearer frame (f0 for t < 0.5, f1 after)
fn toward(f: vec4f, lo: bool) -> vec2f { return select(f.zw, f.xy, lo); }

// RIFE result (flow fT, mask mT) for timestep tau, drawn at time t. A point q there is at q + f0 in
// A and q + f1 in B, and (piecewise linear) at q + (tau - t) / tau * f0 before tau,
// q + (t - tau) / (1 - tau) * f1 after. Find the q that is at p at time t: q + c * toward(q) = p.
// The fixed point is iterated from three starts (no motion, and one step along / against the motion
// at p); the candidate that lands closest to p wins, where a point RIFE's mask says is only visible
// in the farther frame (it appears from behind something) loses against one that is visible on our
// side (this only ranks the candidates). Returns (colour, miss): how far (model pixels) the chosen
// candidate lands from p.
fn rifeAt(p: vec2f, t: f32, tau: f32, fT: texture_2d<f32>, mT: texture_2d<f32>) -> vec4f {
  let lo = t < tau;
  let c = select((t - tau) / (1.0 - tau), (tau - t) / tau, lo);
  // t mapped so that tau -> 0.5 (RIFE's own blend at t = tau), 0 -> 0, 1 -> 1
  let s = select(0.5 + 0.5 * (t - tau) / (1.0 - tau), 0.5 * t / tau, lo);
  let px = vec2f(textureDimensions(fT));
  let m0 = c * toward(textureSampleLevel(fT, samp, p, 0.0), lo);
  var starts = array<vec2f, 3>(p, p - m0, p + m0);
  var q = p;
  var best = 1e9;
  var miss = 1e9;
  for (var k = 0; k < 3; k++) {
    var r = starts[k];
    for (var i = 0; i < 5; i++) { r = p - c * toward(textureSampleLevel(fT, samp, r, 0.0), lo); }
    let res = length((r + c * toward(textureSampleLevel(fT, samp, r, 0.0), lo) - p) * px);
    let m = textureSampleLevel(mT, samp, r, 0.0).r;
    let hidden = smoothstep(0.75, 0.95, select(m, 1.0 - m, lo)) * c;
    let cost = res + 3.0 * hidden;
    if (cost < best) { best = cost; miss = res; q = r; }
  }
  let f = textureSampleLevel(fT, samp, q, 0.0);
  let m = textureSampleLevel(mT, samp, q, 0.0).r;
  let a = sharp(texA, q + f.xy) * gA;
  let b = sharp(texB, q + f.zw) * gB;
  // RIFE's blend weight m, shifted towards the nearer frame (squared time weights: equal at s = 0.5,
  // so t = tau is RIFE's own output; where m is one-sided (occlusion) it still wins)
  let wa = m * (1.0 - s) * (1.0 - s);
  let wb = (1.0 - m) * s * s;
  return vec4f((wa * a + wb * b) / max(wa + wb, 1e-6), miss);
}

fn interp(p: vec2f, t: f32) -> vec3f {
  // From RIFE at t = 0.5 where it extrapolates cleanly (it is RIFE's most accurate timestep); where it
  // misses, from the extra anchor nearer to t (if bound), which has far less to extrapolate.
  var r = rifeAt(p, t, 0.5, flowT, maskT);
  var sel = vec4f(0.0, 1.0, 0.0, 0.0);
  if (U.hybrid >= 0.0) { sel = textureSampleLevel(selT, samp, p, 0.0); }
  // half-resolution RIFE where the selection map prefers it (large motion)
  if (sel.b > 0.002) { r = mix(r, rifeAt(p, t, 0.5, flowH, maskH), sel.b); }
  if (U.tau != 0.5 && r.w > 0.5) {
    let rx = rifeAt(p, t, U.tau, flowX, maskX);
    let w = smoothstep(0.5, 1.5, r.w) * step(rx.w, r.w);
    r = mix(r, rx, w);
  }
  var col = r.rgb;
  let best = r.w;
  let f = Fl(p);
  var conf = 1.0 - smoothstep(0.08, 0.25, abs(lum(Ag(p + f.xy)) - lum(Bg(p + f.zw))));
  let lo = t < 0.5;
  // No point of either RIFE result lands here: background that is uncovered at time t but hidden at
  // the RIFE timestep (behind something moving). Fill it from the nearer frame, along block
  // matching's motion if there.
  let hole = smoothstep(1.0, 2.5, best);
  var fill = select(Bg(p), Ag(p), lo);
  if (U.hybrid >= 0.0) {
    let bm = bmInterp(p, t).rgb;
    fill = bm;
    let w = sel.r * U.hybrid;
    if (w > 0.002) { col = mix(col, bm, w); }
    conf = sel.g;
  }
  col = mix(col, fill, hole);
  // Optional artifact masking: where no motion is trusted, show the nearer original frame instead.
  if (U.masking > 0.0) {
    let near = select(Bg(p), Ag(p), t < 0.5);
    col = mix(col, near, U.masking * (1.0 - conf));
  }
  return col;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let p = (in.pos.xy - U.rect.xy) / U.rect.zw;
  if (p.x < 0.0 || p.y < 0.0 || p.x > 1.0 || p.y > 1.0) { return vec4f(0.0); }
  let t = U.t;
  setGain(t);
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
    case 5u: {
      // selection map: red = block matching used, green = half-resolution RIFE used, blue = low confidence
      let s = select(vec4f(0.0, 1.0, 0.0, 0.0), textureSampleLevel(selT, samp, p, 0.0), U.hybrid >= 0.0);
      col = mix(vec3f(lum(A(p)) * 0.6), vec3f(1.0, 0.1, 0.1), s.r);
      col = mix(col, vec3f(0.1, 1.0, 0.2), s.b);
      col = mix(col, vec3f(0.1, 0.3, 1.0), (1.0 - s.g) * 0.7);
    }
    default: { col = interp(p, t); }
  }
  if (U.split > 0.0 && abs(in.pos.x - U.split) < 1.0) { col = vec3f(1.0); }
  return vec4f(col, 1.0);
}
`;

  S.WGSL = { ingest, toTensor, prep, conv, convReduce, deconv, deconvPlain, accum, pack, select, render };
})();
