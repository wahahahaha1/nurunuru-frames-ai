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

  // Block input at the block's resolution: bilinear downscale (PyTorch align_corners=False) of
  //   first block: img0, img1, feat0, feat1, timestep                                   (15 channels)
  //   later:  warp(img0, f0), warp(img1, f1), warp(feat0, f0), warp(feat1, f1), timestep,
  //           mask, block features, (f0, f1) / scale                                    (28 channels)
  // Frame tensors hold RGB (planes 0-2) and the Head features (planes 3-6). Warping = bilinear
  // sampling at x + flow with border clamp (grid_sample, align_corners=True).
  const prep = /* wgsl */ `
struct PU { pw: u32, ph: u32, bw: u32, bh: u32, scale: f32, first: u32, pad0: u32, pad1: u32 };
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
  v[14] = 0.5;
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

  S.WGSL = { ingest, toTensor, prep, conv, convReduce, deconv, deconvPlain, accum, pack, render };
})();
