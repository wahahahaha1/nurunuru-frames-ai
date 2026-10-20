// Block-matching motion estimation (from NuruNuru Frames): luma pyramid per frame, hierarchical
// block matching in both directions, 3x3 vector median. Used next to RIFE: where its vectors
// explain the two frames clearly better (pans, fine texture), the renderer uses them instead.
//   * Forward field F: blocks on frame A, vector v with A(x) ~ B(x + v).
//     Backward field G: blocks on frame B, vector w with B(x) ~ A(x + w).
//   * Fields are rgba16float textures (one texel per block): (v.x, v.y) in UV units, z = SAD per pixel.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});
  const BU = GPUBufferUsage;
  const TU = GPUTextureUsage;

  // Measured on RTX 3080, 1080p source: about 2.2 ms per source frame.
  S.BM_PARAMS = {
    meWidth: 960, // luma width used for block matching
    bs: 8,
    ov: 4,
    topRadius: 8,
    radius: 2,
    subpel: 1,
    lambda: 0.003, // penalty per pixel of deviation from the predictor
    lambdaZero: 0.0005, // slight preference for short vectors
  };

  // Frame texture -> luma at motion-estimation resolution (box filter of taps x taps bilinear samples).
  const luma = /* wgsl */ `
struct LU { w: u32, h: u32, taps: u32, pad: u32 };
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outL: array<f32>;
@group(0) @binding(3) var<uniform> U: LU;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.w || g.y >= U.h) { return; }
  let inv = vec2f(1.0 / f32(U.w), 1.0 / f32(U.h));
  let base = vec2f(g.xy) * inv;
  let n = U.taps;
  var s = 0.0;
  for (var j = 0u; j < n; j++) {
    for (var i = 0u; i < n; i++) {
      let uv = base + (vec2f(f32(i), f32(j)) + 0.5) / f32(n) * inv;
      s += dot(textureSampleLevel(img, samp, uv, 0.0).rgb, vec3f(0.299, 0.587, 0.114));
    }
  }
  outL[g.y * U.w + g.x] = s / f32(n * n);
}
`;

  const downsample = /* wgsl */ `
struct DU { iw: u32, ih: u32, ow: u32, oh: u32 };
@group(0) @binding(0) var<storage, read> inL: array<f32>;
@group(0) @binding(1) var<storage, read_write> outL: array<f32>;
@group(0) @binding(2) var<uniform> U: DU;
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.ow || g.y >= U.oh) { return; }
  let x0 = min(g.x * 2u, U.iw - 1u);
  let x1 = min(g.x * 2u + 1u, U.iw - 1u);
  let y0 = min(g.y * 2u, U.ih - 1u);
  let y1 = min(g.y * 2u + 1u, U.ih - 1u);
  outL[g.y * U.ow + g.x] = 0.25 * (inL[y0 * U.iw + x0] + inL[y0 * U.iw + x1] + inL[y1 * U.iw + x0] + inL[y1 * U.iw + x1]);
}
`;

  // Hierarchical block matching, one workgroup per block.
  //  top level  (useParent = 0): exhaustive search within +-radius (+ temporal candidate)
  //  lower levels (useParent = 1): parent 3x3 neighbourhood (x2), temporal, zero -> best,
  //                                then full search within +-radius around it
  //  subpel > 0: that many rounds of 8-neighbour sub-pixel refinement (1/2, 1/4 ...)
  const motion = /* wgsl */ `
struct MEP {
  w: u32, h: u32, nbx: u32, nby: u32,
  pnbx: u32, pnby: u32, bs: u32, ov: u32,
  radius: i32, useParent: u32, useTemporal: u32, subpel: u32,
  lambda: f32, lambdaZero: f32, pad0: f32, pad1: f32,
};
@group(0) @binding(0) var<uniform> P: MEP;
@group(0) @binding(1) var<storage, read> cur: array<f32>;
@group(0) @binding(2) var<storage, read> refL: array<f32>;
@group(0) @binding(3) var<storage, read> parent: array<vec4f>;
@group(0) @binding(4) var<storage, read> temporal: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> outv: array<vec4f>;

var<workgroup> blk: array<f32, 1024>;
var<workgroup> win: array<f32, 1600>;
var<workgroup> ps: array<f32, 64>;
var<workgroup> rc: array<f32, 64>;
var<workgroup> rv: array<vec2f, 64>;
var<workgroup> pred: vec2f;

// Each candidate's SAD is split over a group of 4 lanes (lane l sums rows l, l+4, ...);
// the 64 threads therefore evaluate 16 candidates at a time.

fn refAt(x: i32, y: i32) -> f32 {
  let cx = clamp(x, 0, i32(P.w) - 1);
  let cy = clamp(y, 0, i32(P.h) - 1);
  return refL[u32(cy) * P.w + u32(cx)];
}

fn sadGlobalPart(v: vec2i, x0: i32, y0: i32, ws: i32, lane: i32) -> f32 {
  var s = 0.0;
  for (var j = lane; j < ws; j += 4) {
    for (var i = 0; i < ws; i++) {
      s += abs(blk[j * ws + i] - refAt(x0 + i + v.x, y0 + j + v.y));
    }
  }
  return s;
}

// d = position (window pixels) of the block's top-left inside the shared search window.
fn sadWinPart(d: vec2i, ws: i32, side: i32, lane: i32) -> f32 {
  var s = 0.0;
  for (var j = lane; j < ws; j += 4) {
    let row = (j + d.y) * side + d.x;
    for (var i = 0; i < ws; i++) {
      s += abs(blk[j * ws + i] - win[row + i]);
    }
  }
  return s;
}

fn sadWinSubPart(d: vec2f, ws: i32, side: i32, lane: i32) -> f32 {
  let fd = floor(d);
  let a = d - fd;
  let o = vec2i(fd);
  var s = 0.0;
  for (var j = lane; j < ws; j += 4) {
    let r0 = (j + o.y) * side + o.x;
    let r1 = r0 + side;
    for (var i = 0; i < ws; i++) {
      let top = mix(win[r0 + i], win[r0 + i + 1], a.x);
      let bot = mix(win[r1 + i], win[r1 + i + 1], a.x);
      s += abs(blk[j * ws + i] - mix(top, bot, a.y));
    }
  }
  return s;
}

// Sum of the 4 partial SADs of this thread's group (call from uniform control flow).
fn groupSum(tid: u32, val: f32) -> f32 {
  ps[tid] = val;
  workgroupBarrier();
  let b = tid & 0xFFFFFFFCu;
  let s = ps[b] + ps[b + 1u] + ps[b + 2u] + ps[b + 3u];
  workgroupBarrier();
  return s;
}

fn cost(v: vec2f, sad: f32, npix: f32) -> f32 {
  let d = abs(v - pred);
  return sad / npix + P.lambda * (d.x + d.y) + P.lambdaZero * (abs(v.x) + abs(v.y));
}

fn reduceBest(tid: u32) {
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (tid < s && rc[tid + s] < rc[tid]) {
      rc[tid] = rc[tid + s];
      rv[tid] = rv[tid + s];
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) tid: u32) {
  let bx = wg.x;
  let by = wg.y;
  let lane = i32(tid & 3u);
  let cand = i32(tid >> 2u);
  let ws = i32(P.bs + 2u * P.ov);
  let x0 = i32(bx * P.bs) - i32(P.ov);
  let y0 = i32(by * P.bs) - i32(P.ov);
  for (var k = tid; k < u32(ws * ws); k += 64u) {
    let cx = clamp(x0 + i32(k) % ws, 0, i32(P.w) - 1);
    let cy = clamp(y0 + i32(k) / ws, 0, i32(P.h) - 1);
    blk[k] = cur[u32(cy) * P.w + u32(cx)];
  }
  if (tid == 0u) {
    var pr = vec2f(0.0);
    if (P.useParent == 1u) {
      let pbx = min(bx / 2u, P.pnbx - 1u);
      let pby = min(by / 2u, P.pnby - 1u);
      pr = parent[pby * P.pnbx + pbx].xy * 2.0;
    } else if (P.useTemporal == 1u) {
      pr = temporal[by * P.nbx + bx].xy;
    }
    pred = pr;
  }
  workgroupBarrier();
  let npix = f32(ws * ws);
  var bestC = 1e30;
  var bestV = vec2f(0.0);

  // Stage 1: exhaustive search (top level) or predictor candidates (lower levels).
  if (P.useParent == 0u) {
    let r = P.radius;
    let n = 2 * r + 1;
    let total = n * n + 1; // + temporal predictor
    for (var base = 0; base < total; base += 16) {
      let idx = base + cand;
      var valid = idx < total;
      var v = round(pred);
      if (idx < n * n) {
        v = vec2f(f32(idx % n - r), f32(idx / n - r));
      } else {
        valid = valid && P.useTemporal == 1u;
      }
      var part = 0.0;
      if (valid) { part = sadGlobalPart(vec2i(v), x0, y0, ws, lane); }
      let sad = groupSum(tid, part);
      if (valid && lane == 0) {
        let c = cost(v, sad, npix);
        if (c < bestC) { bestC = c; bestV = v; }
      }
    }
  } else {
    // 9 parent neighbourhood vectors (x2), temporal, zero
    var v = vec2f(0.0);
    var valid = cand < 11;
    if (cand < 9) {
      let qx = clamp(i32(min(bx / 2u, P.pnbx - 1u)) + cand % 3 - 1, 0, i32(P.pnbx) - 1);
      let qy = clamp(i32(min(by / 2u, P.pnby - 1u)) + cand / 3 - 1, 0, i32(P.pnby) - 1);
      v = parent[u32(qy) * P.pnbx + u32(qx)].xy * 2.0;
    } else if (cand == 9) {
      valid = P.useTemporal == 1u;
      v = temporal[by * P.nbx + bx].xy;
    }
    v = round(v);
    var part = 0.0;
    if (valid) { part = sadGlobalPart(vec2i(v), x0, y0, ws, lane); }
    let sad = groupSum(tid, part);
    if (valid && lane == 0) {
      bestC = cost(v, sad, npix);
      bestV = v;
    }
  }
  rc[tid] = bestC;
  rv[tid] = bestV;
  reduceBest(tid);

  // Stage 2: load a search window around the best integer vector into shared memory.
  let r = select(0, P.radius, P.useParent == 1u);
  let m = r + 1;
  let side = ws + 2 * m;
  let c0 = rv[0];
  let cc0 = rc[0];
  workgroupBarrier();
  if (r > 0 || P.subpel > 0u) {
    let ox = x0 + i32(c0.x) - m;
    let oy = y0 + i32(c0.y) - m;
    for (var k = i32(tid); k < side * side; k += 64) {
      win[k] = refAt(ox + k % side, oy + k / side);
    }
    workgroupBarrier();
  }

  // Stage 3: full search within +-r around c0.
  if (r > 0) {
    let n = 2 * r + 1;
    bestC = 1e30;
    bestV = c0;
    if (tid == 0u) { bestC = cc0; }
    for (var base = 0; base < n * n; base += 16) {
      let idx = base + cand;
      let valid = idx < n * n;
      let d = vec2i(idx % n - r, idx / n - r);
      var part = 0.0;
      if (valid) { part = sadWinPart(d + vec2i(m), ws, side, lane); }
      let sad = groupSum(tid, part);
      if (valid && lane == 0) {
        let v = c0 + vec2f(d);
        let c = cost(v, sad, npix);
        if (c < bestC) { bestC = c; bestV = v; }
      }
    }
    rc[tid] = bestC;
    rv[tid] = bestV;
    reduceBest(tid);
  }

  // Stage 4: sub-pixel refinement (bilinear) inside the window, 8 neighbours per round.
  var step = 0.5;
  for (var it = 0u; it < P.subpel; it++) {
    let cb = rv[0];
    let ccb = rc[0];
    workgroupBarrier();
    bestC = 1e30;
    bestV = cb;
    if (tid == 0u) { bestC = ccb; }
    let valid = cand < 8;
    let k = select(cand, cand + 1, cand >= 4);
    let v = cb + vec2f(f32(k % 3 - 1), f32(k / 3 - 1)) * step;
    let d = clamp(v - c0 + f32(m), vec2f(0.0), vec2f(f32(2 * m)));
    var part = 0.0;
    if (valid) { part = sadWinSubPart(d, ws, side, lane); }
    let sad = groupSum(tid, part);
    if (valid && lane == 0) {
      let c = cost(v, sad, npix);
      if (c < bestC) { bestC = c; bestV = v; }
    }
    rc[tid] = bestC;
    rv[tid] = bestV;
    reduceBest(tid);
    step = step * 0.5;
  }

  // Final: plain SAD of the chosen vector (group 0) for reliability.
  let vf = rv[0];
  var part = 0.0;
  if (cand == 0) {
    if (r > 0 || P.subpel > 0u) {
      part = sadWinSubPart(clamp(vf - c0 + f32(m), vec2f(0.0), vec2f(f32(2 * m))), ws, side, lane);
    } else {
      part = sadGlobalPart(vec2i(vf), x0, y0, ws, lane);
    }
  }
  let sad = groupSum(tid, part);
  if (tid == 0u) {
    outv[by * P.nbx + bx] = vec4f(vf, sad / npix, 0.0);
  }
}
`;

  // 3x3 component-wise vector median -> rgba16float field texture (UV units).
  const finalize = /* wgsl */ `
struct FU { nbx: u32, nby: u32, w: u32, h: u32 };
@group(0) @binding(0) var<storage, read> vin: array<vec4f>;
@group(0) @binding(1) var fieldOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> U: FU;

fn med9(a: array<f32, 9>) -> f32 {
  var v = a;
  for (var i = 1; i < 9; i++) {
    let x = v[i];
    var j = i - 1;
    while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; j--; }
    v[j + 1] = x;
  }
  return v[4];
}

@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= U.nbx || g.y >= U.nby) { return; }
  let c = vin[g.y * U.nbx + g.x];
  var xs: array<f32, 9>;
  var ys: array<f32, 9>;
  var n = 0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let qx = clamp(i32(g.x) + dx, 0, i32(U.nbx) - 1);
      let qy = clamp(i32(g.y) + dy, 0, i32(U.nby) - 1);
      let q = vin[u32(qy) * U.nbx + u32(qx)];
      xs[n] = q.x;
      ys[n] = q.y;
      n++;
    }
  }
  textureStore(fieldOut, g.xy, vec4f(med9(xs) / f32(U.w), med9(ys) / f32(U.h), c.z, 0.0));
}
`;

  // Duplicate-frame test (anime drawn on twos / threes, repeated frames): counts the 8x8 blocks of a
  // coarse luma level whose mean absolute difference to the previous frame exceeds a threshold.
  // A frame with no such block is a repeat of the previous drawing (codec noise stays below it,
  // while even a small real change, like a mouth, makes at least one block exceed it).
  const diff = /* wgsl */ `
struct XU { w: u32, h: u32, thr: f32, pad: u32 };
@group(0) @binding(0) var<uniform> U: XU;
@group(0) @binding(1) var<storage, read> la: array<f32>;
@group(0) @binding(2) var<storage, read> lb: array<f32>;
@group(0) @binding(3) var<storage, read_write> outC: array<atomic<u32>, 4>;

var<workgroup> part: array<f32, 64>;

@compute @workgroup_size(8, 8) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) tid: u32) {
  let x = wg.x * 8u + lid.x;
  let y = wg.y * 8u + lid.y;
  var d = 0.0;
  if (x < U.w && y < U.h) { d = abs(la[y * U.w + x] - lb[y * U.w + x]); }
  part[tid] = d;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (tid < s) { part[tid] += part[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) {
    atomicAdd(&outC[1], 1u);
    if (part[0] / 64.0 > U.thr) { atomicAdd(&outC[0], 1u); }
  }
}
`;
  const DIFF_THRESHOLD = 0.012;

  // Mean luma of a frame (coarse level): [sum * 256, pixel count], for brightness changes (fades).
  const mean = /* wgsl */ `
struct MU { w: u32, h: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> U: MU;
@group(0) @binding(1) var<storage, read> lum: array<f32>;
@group(0) @binding(2) var<storage, read_write> outM: array<atomic<u32>, 2>;

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) tid: u32) {
  var v = 0.0;
  var n = 0u;
  if (g.x < U.w && g.y < U.h) { v = lum[g.y * U.w + g.x]; n = 1u; }
  part[tid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) { part[tid] += part[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) { atomicAdd(&outM[0], u32(part[0] * 256.0)); }
  if (n == 1u) { atomicAdd(&outM[1], 1u); }
}
`; // mean |luma difference| of an 8x8 block (0..1) that counts as a change

  class BlockMatcher {
    constructor(device, sampler) {
      this.device = device;
      this.samp = sampler;
      const d = device;
      const comp = (code, label) =>
        d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code, label }), entryPoint: 'main' } });
      this.lumaPipe = comp(luma, 'bm-luma');
      this.downPipe = comp(downsample, 'bm-downsample');
      this.mePipe = comp(motion, 'bm-motion');
      this.finPipe = comp(finalize, 'bm-finalize');
      this.diffPipe = comp(diff, 'bm-diff');
      this.meanPipe = comp(mean, 'bm-mean');
      this.zeroBuf = d.createBuffer({ size: 64, usage: BU.STORAGE });
      this.res = null;
    }

    // Allocate the pyramid / vector buffers / fields of every frame record (frame textures fw x fh,
    // source srcW x srcH).
    configure(frames, fw, srcW, srcH) {
      this.destroyResources();
      const d = this.device;
      const p = S.BM_PARAMS;
      const uniforms = [];
      const uni = (data) => {
        const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: BU.UNIFORM | BU.COPY_DST });
        d.queue.writeBuffer(b, 0, data);
        uniforms.push(b);
        return b;
      };
      const ms = Math.min(1, p.meWidth / srcW);
      const mw = Math.max(32, Math.round(srcW * ms));
      const mh = Math.max(32, Math.round(srcH * ms));
      const levels = Math.max(2, Math.min(5, Math.floor(Math.log2(mw / 100)) + 1));
      const lw = [mw];
      const lh = [mh];
      for (let k = 1; k < levels; k++) {
        lw.push(Math.ceil(lw[k - 1] / 2));
        lh.push(Math.ceil(lh[k - 1] / 2));
      }
      const nbx = lw.map((w) => Math.ceil(w / p.bs));
      const nby = lh.map((h) => Math.ceil(h / p.bs));
      const lumaU = uni(new Uint32Array([mw, mh, Math.max(1, Math.min(4, Math.round(fw / mw))), 0]));
      const downU = [null];
      for (let k = 1; k < levels; k++) downU.push(uni(new Uint32Array([lw[k - 1], lh[k - 1], lw[k], lh[k]])));
      const meU = [];
      for (let k = 0; k < levels; k++) {
        const top = k === levels - 1;
        meU.push(
          [0, 1].map((useT) => {
            const buf = new ArrayBuffer(64);
            const u = new Uint32Array(buf);
            const i = new Int32Array(buf);
            const f = new Float32Array(buf);
            u.set([lw[k], lh[k], nbx[k], nby[k], top ? 1 : nbx[k + 1], top ? 1 : nby[k + 1], p.bs, p.ov]);
            i[8] = top ? p.topRadius : p.radius;
            u[9] = top ? 0 : 1;
            u[10] = useT;
            u[11] = k === 0 ? p.subpel : 0;
            f[12] = p.lambda;
            f[13] = p.lambdaZero;
            return uni(buf);
          })
        );
      }
      const finU = uni(new Uint32Array([nbx[0], nby[0], mw, mh]));
      // duplicate test on the level closest to 240 px wide
      let dl = 0;
      while (dl + 1 < levels && lw[dl] > 320) dl++;
      const xbuf = new ArrayBuffer(16);
      new Uint32Array(xbuf).set([lw[dl], lh[dl]]);
      new Float32Array(xbuf)[2] = DIFF_THRESHOLD;
      const diffU = uni(xbuf);
      const meanU = uni(new Uint32Array([lw[dl], lh[dl], 0, 0]));
      for (const fr of frames) {
        const bm = {
          luma: lw.map((w, k) => d.createBuffer({ size: w * lh[k] * 4, usage: BU.STORAGE })),
          vecF: lw.map((_, k) => d.createBuffer({ size: nbx[k] * nby[k] * 16, usage: BU.STORAGE })),
          vecG: lw.map((_, k) => d.createBuffer({ size: nbx[k] * nby[k] * 16, usage: BU.STORAGE })),
          fieldF: d.createTexture({ size: [nbx[0], nby[0]], format: 'rgba16float', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING }),
          fieldG: d.createTexture({ size: [nbx[0], nby[0]], format: 'rgba16float', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING }),
          hasPair: false, // fields for (id-1 -> id) are submitted
          pairPrevId: -1,
          diff: d.createBuffer({ size: 16, usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC }),
          mean: d.createBuffer({ size: 16, usage: BU.STORAGE | BU.COPY_DST }), // [luma sum * 256, count]
        };
        bm.fieldFView = bm.fieldF.createView();
        bm.fieldGView = bm.fieldG.createView();
        bm.lumaBG = d.createBindGroup({
          layout: this.lumaPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.samp },
            { binding: 1, resource: fr.view },
            { binding: 2, resource: { buffer: bm.luma[0] } },
            { binding: 3, resource: { buffer: lumaU } },
          ],
        });
        bm.downBG = [null];
        for (let k = 1; k < levels; k++) {
          bm.downBG.push(
            d.createBindGroup({
              layout: this.downPipe.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: bm.luma[k - 1] } },
                { binding: 1, resource: { buffer: bm.luma[k] } },
                { binding: 2, resource: { buffer: downU[k] } },
              ],
            })
          );
        }
        bm.meanBG = d.createBindGroup({
          layout: this.meanPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: meanU } },
            { binding: 1, resource: { buffer: bm.luma[dl] } },
            { binding: 2, resource: { buffer: bm.mean } },
          ],
        });
        fr.bm = bm;
      }
      this.res = { mw, mh, levels, lw, lh, nbx, nby, meU, finU, diffU, dl, uniforms, frames };
    }

    // Encode the duplicate test of frame b against a (after both luma pyramids); b.bm.diff gets
    // [changed blocks, blocks]. Clear the buffer before the pass (clearDiff).
    clearDiff(enc, b) {
      enc.clearBuffer(b.bm.diff);
    }

    encodeDiff(cp, a, b) {
      const r = this.res;
      const k = r.dl;
      cp.setPipeline(this.diffPipe);
      cp.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: this.diffPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: r.diffU } },
            { binding: 1, resource: { buffer: a.bm.luma[k] } },
            { binding: 2, resource: { buffer: b.bm.luma[k] } },
            { binding: 3, resource: { buffer: b.bm.diff } },
          ],
        })
      );
      cp.dispatchWorkgroups(Math.ceil(r.lw[k] / 8), Math.ceil(r.lh[k] / 8));
    }

    // Read the duplicate test of b (after encodeDiff was submitted): resolves to true for a repeat.
    async readDup(b) {
      const d = this.device;
      const rb = d.createBuffer({ size: 16, usage: BU.COPY_DST | BU.MAP_READ });
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(b.bm.diff, 0, rb, 0, 16);
      d.queue.submit([enc.finish()]);
      try {
        await rb.mapAsync(GPUMapMode.READ);
        const v = new Uint32Array(rb.getMappedRange().slice(0));
        return v[1] > 0 && v[0] === 0;
      } finally {
        rb.destroy();
      }
    }

    destroyResources() {
      if (!this.res) return;
      for (const fr of this.res.frames) {
        const bm = fr.bm;
        if (!bm) continue;
        bm.fieldF.destroy();
        bm.fieldG.destroy();
        bm.diff.destroy();
        bm.mean.destroy();
        for (const b of [...bm.luma, ...bm.vecF, ...bm.vecG]) b.destroy();
        fr.bm = null;
      }
      for (const b of this.res.uniforms) b.destroy();
      this.res = null;
    }

    destroy() {
      this.destroyResources();
      this.zeroBuf.destroy();
    }

    // Build the luma pyramid and mean luma of a frame (inside a compute pass; clear fr.bm.mean before
    // the pass with clearMean).
    clearMean(enc, fr) {
      enc.clearBuffer(fr.bm.mean);
    }

    encodeLuma(cp, fr) {
      const r = this.res;
      fr.bm.hasPair = false;
      fr.bm.pairPrevId = -1;
      cp.setPipeline(this.lumaPipe);
      cp.setBindGroup(0, fr.bm.lumaBG);
      cp.dispatchWorkgroups(Math.ceil(r.mw / 16), Math.ceil(r.mh / 16));
      cp.setPipeline(this.downPipe);
      for (let k = 1; k < r.levels; k++) {
        cp.setBindGroup(0, fr.bm.downBG[k]);
        cp.dispatchWorkgroups(Math.ceil(r.lw[k] / 16), Math.ceil(r.lh[k] / 16));
      }
      cp.setPipeline(this.meanPipe);
      cp.setBindGroup(0, fr.bm.meanBG);
      cp.dispatchWorkgroups(Math.ceil(r.lw[r.dl] / 16), Math.ceil(r.lh[r.dl] / 16));
    }

    // Steps estimating both fields of the pair (a -> b); the temporal predictor is a's pair, if valid.
    steps(a, b) {
      const d = this.device;
      const r = this.res;
      const A = a.bm;
      const B = b.bm;
      const useT = A.hasPair && A.pairPrevId === a.id - 1 ? 1 : 0;
      const steps = [];
      const dirs = [
        { name: 'F', cur: A, ref: B, out: B.vecF, tmp: A.vecF },
        { name: 'G', cur: B, ref: A, out: B.vecG, tmp: A.vecG },
      ];
      for (const dir of dirs) {
        for (let k = r.levels - 1; k >= 0; k--) {
          const top = k === r.levels - 1;
          steps.push({
            key: `bm${k}`,
            pipe: this.mePipe,
            block: -2,
            kind: 'bm',
            bg: d.createBindGroup({
              layout: this.mePipe.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: r.meU[k][useT] } },
                { binding: 1, resource: { buffer: dir.cur.luma[k] } },
                { binding: 2, resource: { buffer: dir.ref.luma[k] } },
                { binding: 3, resource: { buffer: top ? this.zeroBuf : dir.out[k + 1] } },
                { binding: 4, resource: { buffer: useT ? dir.tmp[k] : this.zeroBuf } },
                { binding: 5, resource: { buffer: dir.out[k] } },
              ],
            }),
            wg: [r.nbx[k], r.nby[k], 1],
            cost: r.nbx[k] * r.nby[k] * 2e4,
          });
        }
      }
      for (const [vec, field] of [[B.vecF[0], B.fieldFView], [B.vecG[0], B.fieldGView]]) {
        steps.push({
          key: 'bmf',
          pipe: this.finPipe,
          block: -2,
          kind: 'bm',
          bg: d.createBindGroup({
            layout: this.finPipe.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: vec } },
              { binding: 1, resource: field },
              { binding: 2, resource: { buffer: r.finU } },
            ],
          }),
          wg: [Math.ceil(r.nbx[0] / 8), Math.ceil(r.nby[0] / 8), 1],
          cost: r.nbx[0] * r.nby[0] * 100,
          after: () => {
            B.hasPair = true;
            B.pairPrevId = a.id;
          },
        });
      }
      return steps;
    }

    // UV scale of the fields: block grid covers slightly more than the picture.
    fieldScale() {
      const r = this.res;
      return [r.mw / (S.BM_PARAMS.bs * r.nbx[0]), r.mh / (S.BM_PARAMS.bs * r.nby[0])];
    }
  }

  S.BlockMatcher = BlockMatcher;
})();
