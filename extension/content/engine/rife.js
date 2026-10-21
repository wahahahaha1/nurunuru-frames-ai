// RIFE (IFNet, Practical-RIFE by hzwer, MIT) 4.25 family implemented on WebGPU compute shaders.
// Per frame: Head (3 convs + deconv) -> 4 feature channels, stored in the frame tensor.
// Per source pair, one inference at timestep 0.5: five coarse-to-fine blocks, each = block input
// (warp + downscale) -> 2 strided convs -> 8 residual convs -> deconv + pixel shuffle -> upsample
// into the running (flow, mask, features). The work is returned as a list of steps (one dispatch
// each, with a key for timing) so the caller can spread it over several display frames.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});
  const BU = GPUBufferUsage;

  // scales = input downscale of each block; modulo = what the padded input sides must be multiples
  // of (coarsest block: 1 / scale, then two stride-2 convs)
  S.RIFE_MODELS = {
    'rife-4.25': { file: 'models/rife-v4.25.bin', scales: [16, 8, 4, 2, 1], modulo: 64 },
    'rife-4.25-lite': { file: 'models/rife-v4.25-lite.bin', scales: [32, 16, 8, 4, 1], modulo: 128 },
  };
  const NC = 13; // block output channels: flow 4, mask 1, features 8
  const FRAME_PLANES = 7; // frame tensor: RGB + 4 Head features

  const CONV_CFG = {
    conv1: { stride: 1, tx: 16, ty: 8, px: 2, py: 2, ci: 4 },
    conv1s: { stride: 1, tx: 8, ty: 8, px: 1, py: 1, ci: 8 }, // small feature maps (coarse blocks)
    conv2: { stride: 2, tx: 8, ty: 8, px: 2, py: 2, ci: 2 },
  };
  // Workgroups a dispatch should offer to fill a large GPU; small layers get split-K below this.
  const TARGET_WG = 512;

  // Shader variant, workgroup grid and split-K factor of a 3x3 conv layer.
  function convPlan(L, stride, ow, oh) {
    const tiles = (cfg) => Math.ceil(ow / (cfg.tx * cfg.px)) * Math.ceil(oh / (cfg.ty * cfg.py));
    const ncog = Math.ceil(L.cout / 16);
    let key = stride === 1 ? 'conv1' : 'conv2';
    if (stride === 1 && tiles(CONV_CFG.conv1) * ncog < TARGET_WG / 2) key = 'conv1s';
    const cfg = CONV_CFG[key];
    const wgs = tiles(cfg) * ncog;
    let ks = Math.max(1, Math.min(8, Math.floor(L.cin / (cfg.ci * 4)), Math.ceil(TARGET_WG / wgs)));
    const cinPer = Math.ceil(L.cin / ks / cfg.ci) * cfg.ci;
    ks = Math.ceil(L.cin / cinPer);
    const wg = [Math.ceil(ow / (cfg.tx * cfg.px)), Math.ceil(oh / (cfg.ty * cfg.py)), ncog * ks];
    return { key, ks, cinPer, ncog, wg };
  }

  // Split-K factor of a block deconv (output 4fw x 4fh, one thread per pixel).
  function deconvPlan(cin, fw, fh) {
    const wgs = Math.ceil((fw * 4) / 16) * Math.ceil((fh * 4) / 16);
    let ks = Math.max(1, Math.min(16, Math.floor(cin / 8), Math.ceil(TARGET_WG / wgs)));
    const cinPer = Math.ceil(cin / ks);
    ks = Math.ceil(cin / cinPer);
    return { ks, cinPer };
  }

  function half2float(h) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    if (e === 0) return s * f * 2 ** -24;
    if (e === 31) return f ? NaN : s * Infinity;
    return s * (1 + f / 1024) * 2 ** (e - 15);
  }

  // Weight file (tools/export_weights.py): "NNFW", u32 version, u32 header length, JSON header, fp16 data.
  S.loadRifeWeights = async (url) => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`モデルを読み込めません (${res.status})`);
    const buf = await res.arrayBuffer();
    const dv = new DataView(buf);
    if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== 'NNFW') throw new Error('モデルファイルの形式が違います');
    const hlen = dv.getUint32(8, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, hlen)));
    const data = new Uint16Array(buf, 12 + hlen);
    const lut = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) lut[i] = half2float(i);
    const tensors = new Map();
    for (const t of header.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = lut[data[t.offset + i]];
      tensors.set(t.name, { shape: t.shape, data: out });
    }
    return { model: header.model, tensors };
  };

  // Rearrange PyTorch weights into the shader layouts (see shaders.js) in one Float32Array.
  function packWeights(tensors, nBlocks) {
    const parts = [];
    let size = 0; // in vec4 units
    const add = (arr) => {
      const off = size;
      parts.push(arr);
      size += arr.length / 4;
      return off;
    };
    const get = (name) => {
      const t = tensors.get(name);
      if (!t) throw new Error('モデルに ' + name + ' がありません');
      return t;
    };
    // 3x3 conv, output channels padded to groups of 16; ResConv's beta folded into weights and bias
    const convW = (wName, bName, betaName) => {
      const w = get(wName);
      const b = get(bName).data;
      const beta = betaName ? get(betaName).data : null;
      const [cout, cin] = w.shape;
      const groups = Math.ceil(cout / 16);
      const W = new Float32Array(groups * cin * 9 * 16);
      for (let co = 0; co < cout; co++) {
        const g = co >> 4;
        const lane = co & 15; // = v * 4 + e
        const k = beta ? beta[co] : 1;
        for (let ci = 0; ci < cin; ci++) {
          for (let tap = 0; tap < 9; tap++) {
            W[((g * cin + ci) * 9 + tap) * 16 + lane] = w.data[(co * cin + ci) * 9 + tap] * k;
          }
        }
      }
      const B = new Float32Array(groups * 16);
      for (let co = 0; co < cout; co++) B[co] = b[co] * (beta ? beta[co] : 1);
      return { cin, cout, wOff: add(W), bOff: add(B) };
    };
    // block deconv [cin, 4 * NC, 4, 4] + pixel shuffle: 4 vec4 (16 slots, NC used) per (ci, sub, tap)
    const deconvW = (wName, bName) => {
      const w = get(wName);
      const b = get(bName).data;
      const [cin, cout] = w.shape;
      if (cout !== 4 * NC) throw new Error('想定外のモデル構造です: ' + wName);
      const W = new Float32Array(cin * 4 * 16 * 16);
      for (let ci = 0; ci < cin; ci++) {
        for (let sub = 0; sub < 4; sub++) {
          for (let tap = 0; tap < 16; tap++) {
            const base = ((ci * 4 + sub) * 16 + tap) * 16;
            for (let c = 0; c < NC; c++) W[base + c] = w.data[(ci * cout + c * 4 + sub) * 16 + tap];
          }
        }
      }
      const B = new Float32Array(4 * 16);
      for (let sub = 0; sub < 4; sub++) for (let c = 0; c < NC; c++) B[sub * 16 + c] = b[c * 4 + sub];
      return { cin, wOff: add(W), bOff: add(B) };
    };
    // plain deconv [cin, 4, 4, 4]: one vec4 per (ci, tap)
    const deconvPlainW = (wName, bName) => {
      const w = get(wName);
      const [cin, cout] = w.shape;
      const W = new Float32Array(cin * 16 * 4);
      for (let ci = 0; ci < cin; ci++) {
        for (let tap = 0; tap < 16; tap++) for (let c = 0; c < cout; c++) W[(ci * 16 + tap) * 4 + c] = w.data[(ci * cout + c) * 16 + tap];
      }
      const B = new Float32Array(4);
      B.set(get(bName).data);
      return { cin, wOff: add(W), bOff: add(B) };
    };
    const head = {
      convs: [0, 1, 2].map((i) => convW(`encode.cnn${i}.weight`, `encode.cnn${i}.bias`)),
      deconv: deconvPlainW('encode.cnn3.weight', 'encode.cnn3.bias'),
    };
    const blocks = [...Array(nBlocks).keys()].map((i) => {
      const p = `block${i}.`;
      return {
        conv0: [convW(p + 'conv0.0.0.weight', p + 'conv0.0.0.bias'), convW(p + 'conv0.1.0.weight', p + 'conv0.1.0.bias')],
        res: [...Array(8).keys()].map((j) =>
          convW(`${p}convblock.${j}.conv.weight`, `${p}convblock.${j}.conv.bias`, `${p}convblock.${j}.beta`)
        ),
        deconv: deconvW(p + 'lastconv.0.weight', p + 'lastconv.0.bias'),
      };
    });
    const all = new Float32Array(size * 4);
    let o = 0;
    for (const a of parts) {
      all.set(a, o);
      o += a.length;
    }
    return { head, blocks, data: all };
  }

  class RifeNet {
    constructor(device, weights) {
      const spec = S.RIFE_MODELS[weights.model];
      if (!spec) throw new Error('未対応のモデルです: ' + weights.model);
      this.device = device;
      this.model = weights.model;
      this.spec = spec;
      const d = device;
      const W = S.WGSL;
      const comp = (code, label) =>
        d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code, label }), entryPoint: 'main' } });
      this.pipes = {
        prep: comp(W.prep, 'rife-prep'),
        conv1: comp(W.conv(CONV_CFG.conv1), 'rife-conv-s1'),
        conv1s: comp(W.conv(CONV_CFG.conv1s), 'rife-conv-s1-small'),
        conv2: comp(W.conv(CONV_CFG.conv2), 'rife-conv-s2'),
        reduce: comp(W.convReduce, 'rife-conv-reduce'),
        deconv: comp(W.deconv(NC), 'rife-deconv'),
        deconvPlain: comp(W.deconvPlain, 'rife-deconv-plain'),
        accum: comp(W.accum, 'rife-accum'),
        sumK: comp(W.sumK, 'rife-sum-k'),
        pack: comp(W.pack, 'rife-pack'),
      };
      const { head, blocks, data } = packWeights(weights.tensors, spec.scales.length);
      this.head = head;
      this.layers = blocks;
      this.wbuf = d.createBuffer({ size: data.byteLength, usage: BU.STORAGE | BU.COPY_DST });
      d.queue.writeBuffer(this.wbuf, 0, data);
      this.res = null;
      this.keyPrefix = '';
    }

    // A second network sharing weights and pipelines, with its own size and scratch buffers
    // (step keys prefixed with `prefix` so their timings are kept apart).
    sibling(prefix) {
      const n = Object.create(RifeNet.prototype);
      Object.assign(n, this, { res: null, keyPrefix: prefix, parent: this });
      return n;
    }

    // Picture size (cw x ch, sides multiples of 8) and padded model input size, about `width` wide.
    // Portrait pictures (vertical video) get the area of a 16:9 picture of that width instead (the
    // same cost; at full width they would cost several times more).
    static modelSize(srcW, srcH, width, modulo) {
      const cw = Math.max(64, Math.round(Math.min(width * RifeNet.widthFactor(srcW, srcH), srcW) / 8) * 8);
      const ch = Math.max(32, Math.round((cw * srcH) / srcW / 8) * 8);
      return { cw, ch, pw: Math.ceil(cw / modulo) * modulo, ph: Math.ceil(ch / modulo) * modulo };
    }

    // Model width / `width` for a srcW x srcH picture (1 up to square, less for portrait).
    static widthFactor(srcW, srcH) {
      const a = srcH / srcW;
      return a > 1 ? Math.sqrt(9 / 16 / a) : 1;
    }

    // Bytes of the largest scratch buffer at padded size pw x ph (the last block's input).
    static scratchBytes(pw, ph) {
      return 28 * 4 * pw * ph;
    }

    // Floats in one frame tensor.
    static tensorSize(pw, ph) {
      return FRAME_PLANES * pw * ph;
    }

    // Allocate scratch buffers and the static part of the step list for a cw x ch picture.
    configure(cw, ch) {
      if (this.res && this.res.cw === cw && this.res.ch === ch) return;
      this.destroyResources();
      const d = this.device;
      const pw = Math.ceil(cw / this.spec.modulo) * this.spec.modulo;
      const ph = Math.ceil(ch / this.spec.modulo) * this.spec.modulo;
      const uniforms = [];
      const uni = (data) => {
        const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: BU.UNIFORM | BU.COPY_DST });
        d.queue.writeBuffer(b, 0, data);
        uniforms.push(b);
        return b;
      };
      const plan = this.spec.scales.map((scale, i) => {
        const L = this.layers[i];
        const bw = pw / scale;
        const bh = ph / scale;
        return { i, scale, L, bw, bh, h1: [bw / 2, bh / 2], h2: [bw / 4, bh / 4], inC: i === 0 ? 15 : 28 };
      });
      // scratch sizes (floats)
      let maxIn = 0;
      let maxMid = 0;
      let maxFeat = 0;
      let maxPart = 0;
      let maxTmp = 0;
      const notePart = (L, stride, ow, oh) => {
        const c = convPlan(L, stride, ow, oh);
        if (c.ks > 1) maxPart = Math.max(maxPart, c.ks * L.cout * ow * oh);
      };
      for (const p of plan) {
        maxIn = Math.max(maxIn, p.inC * p.bw * p.bh);
        maxMid = Math.max(maxMid, p.L.conv0[0].cout * p.h1[0] * p.h1[1]);
        maxFeat = Math.max(maxFeat, p.L.conv0[1].cout * p.h2[0] * p.h2[1]);
        notePart(p.L.conv0[0], 2, ...p.h1);
        notePart(p.L.conv0[1], 2, ...p.h2);
        notePart(p.L.res[0], 1, ...p.h2);
        const dp = deconvPlan(p.L.deconv.cin, ...p.h2);
        maxTmp = Math.max(maxTmp, dp.ks * NC * p.h2[0] * 4 * p.h2[1] * 4);
      }
      const hw = pw / 2;
      const hh = ph / 2;
      notePart(this.head.convs[0], 2, hw, hh);
      notePart(this.head.convs[1], 1, hw, hh);
      const sbuf = (n, extra = 0) => d.createBuffer({ size: Math.max(16, n * 4), usage: BU.STORAGE | extra });
      const bufs = {
        x: sbuf(maxIn),
        mid: sbuf(maxMid),
        f0: sbuf(maxFeat),
        f1: sbuf(maxFeat),
        part: sbuf(maxPart),
        tmp: sbuf(maxTmp),
        fm: sbuf(NC * pw * ph, BU.COPY_SRC), // COPY_SRC: read back by tests
        h0: sbuf(16 * hw * hh),
        h1: sbuf(16 * hw * hh),
      };
      const bg = (pipe, entries) =>
        d.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries: entries.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
        });
      // Conv layer -> 1 step, or 2 with split-K (partial sums, then reduce + bias + residual + act).
      const convSteps = (key, block, L, stride, inBuf, iw, ih, outBuf, ow, oh, resBuf) => {
        const c = convPlan(L, stride, ow, oh);
        const pipe = this.pipes[c.key];
        const split = c.ks > 1;
        const u = uni(new Uint32Array([L.cin, L.cout, iw, ih, ow, oh, L.wOff, L.bOff, resBuf ? 1 : 0, 1, c.ncog, c.cinPer, split ? 1 : 0, 0, 0, 0]));
        const kind = stride === 1 ? 'conv1' : 'conv2';
        const variant = `${c.key} ${L.cin}→${L.cout} ${ow}×${oh}${split ? ` k${c.ks}` : ''}`; // (for profiling)
        const conv = { key: this.keyPrefix + key, pipe, block, kind, variant, bg: bg(pipe, [u, inBuf, this.wbuf, resBuf || inBuf, split ? bufs.part : outBuf]), wg: c.wg, cost: L.cin * L.cout * 9 * ow * oh };
        if (!split) return [conv];
        const n = L.cout * ow * oh;
        const ru = uni(new Uint32Array([n, ow * oh, c.ks, resBuf ? 1 : 0, L.bOff, 1, 0, 0]));
        const red = this.pipes.reduce;
        return [conv, { key: this.keyPrefix + key + 'r', pipe: red, block, kind, bg: bg(red, [ru, bufs.part, this.wbuf, resBuf || bufs.part, outBuf]), wg: [Math.ceil(n / 256), 1, 1], cost: n * c.ks }];
      };
      const steps = []; // static steps (block inputs are filled per job)
      for (const p of plan) {
        const { L, i: b } = p;
        p.prepIndex = steps.length;
        steps.push(null);
        steps.push(...convSteps(`b${b}c0`, b, L.conv0[0], 2, bufs.x, p.bw, p.bh, bufs.mid, ...p.h1));
        steps.push(...convSteps(`b${b}c1`, b, L.conv0[1], 2, bufs.mid, ...p.h1, bufs.f0, ...p.h2));
        let cur = bufs.f0;
        let other = bufs.f1;
        L.res.forEach((R, j) => {
          steps.push(...convSteps(`b${b}r${j}`, b, R, 1, cur, ...p.h2, other, ...p.h2, cur));
          [cur, other] = [other, cur];
        });
        const [fw, fh] = p.h2;
        const dp = deconvPlan(L.deconv.cin, fw, fh);
        const du = uni(new Uint32Array([L.deconv.cin, fw, fh, L.deconv.wOff, L.deconv.bOff, dp.cinPer, 0, 0]));
        steps.push({
          key: `${this.keyPrefix}b${b}d`,
          pipe: this.pipes.deconv,
          block: b,
          kind: 'deconv',
          bg: bg(this.pipes.deconv, [du, cur, this.wbuf, bufs.tmp]),
          wg: [Math.ceil((fw * 4) / 16), Math.ceil((fh * 4) / 16), dp.ks],
          cost: L.deconv.cin * NC * fw * fh * 4,
          variant: `deconv ${L.deconv.cin}→${NC} ${fw * 4}×${fh * 4}${dp.ks > 1 ? ` k${dp.ks}` : ''}`,
        });
        if (dp.ks > 1) {
          // split-K slices summed once here, not per full-resolution pixel in accum
          const n = NC * fw * 4 * fh * 4;
          steps.push({
            key: `${this.keyPrefix}b${b}k`,
            pipe: this.pipes.sumK,
            block: b,
            kind: 'deconv',
            bg: bg(this.pipes.sumK, [uni(new Uint32Array([n, dp.ks, 0, 0])), bufs.tmp]),
            wg: [Math.min(65535, Math.ceil(n / 256)), Math.ceil(n / 256 / 65535), 1],
            cost: n * dp.ks,
          });
        }
        const abuf = new ArrayBuffer(32);
        new Uint32Array(abuf).set([pw, ph, fw * 4, fh * 4, 0, b === 0 ? 1 : 0, 1, NC]);
        new Float32Array(abuf)[4] = p.scale;
        steps.push({
          key: `${this.keyPrefix}b${b}a`,
          pipe: this.pipes.accum,
          block: b,
          kind: 'other',
          bg: bg(this.pipes.accum, [uni(abuf), bufs.tmp, bufs.fm]),
          wg: [Math.ceil(pw / 16), Math.ceil(ph / 16), 1],
          cost: pw * ph * NC * 4,
        });
        p.prepU = new Map(); // timestep -> uniform buffer (made on first use)
        p.prepCost = p.bw * p.bh * (b === 0 ? 30 : 400);
      }
      this.res = {
        cw, ch, pw, ph, plan, bufs, steps, uniforms, bg, convSteps, uni,
        packU: uni(new Uint32Array([cw, ch, pw, ph])),
        headCache: new Map(),
      };
    }

    // Head steps for a frame tensor: features into planes 3-6 (cached per tensor buffer).
    headSteps(tensor) {
      const r = this.res;
      const hit = r.headCache.get(tensor);
      if (hit) return hit.steps;
      const nu = r.uniforms.length;
      let steps;
      const { pw, ph, bufs } = r;
      const hw = pw / 2;
      const hh = ph / 2;
      const [c0, c1, c2] = this.head.convs;
      steps = [
        ...r.convSteps('h0', -1, c0, 2, tensor, pw, ph, bufs.h0, hw, hh),
        ...r.convSteps('h1', -1, c1, 1, bufs.h0, hw, hh, bufs.h1, hw, hh),
        ...r.convSteps('h2', -1, c2, 1, bufs.h1, hw, hh, bufs.h0, hw, hh),
      ];
      const D = this.head.deconv;
      const eu = r.uni(new Uint32Array([D.cin, hw, hh, D.wOff, D.bOff, 3 * pw * ph, 0, 0]));
      steps.push({
        key: this.keyPrefix + 'h3',
        pipe: this.pipes.deconvPlain,
        block: -1,
        kind: 'deconv',
        bg: r.bg(this.pipes.deconvPlain, [eu, bufs.h0, this.wbuf, tensor]),
        wg: [Math.ceil(pw / 16), Math.ceil(ph / 16), 1],
        cost: D.cin * 4 * 4 * pw * ph,
      });
      r.headCache.set(tensor, { steps, uniforms: r.uniforms.splice(nu) });
      return steps;
    }

    // Drop the cached Head steps of frame tensors that are being freed.
    forgetHeads(tensors) {
      const r = this.res;
      if (!r) return;
      for (const t of tensors) {
        const hit = r.headCache.get(t);
        if (!hit) continue;
        for (const b of hit.uniforms) b.destroy();
        r.headCache.delete(t);
      }
    }

    // Step list for the pair (tensors a -> b, Head features already computed) at timestep ts; the
    // last step writes out's flow / mask textures and scene statistics.
    job(tensorA, tensorB, out, ts = 0.5) {
      const r = this.res;
      const steps = r.steps.slice();
      for (const p of r.plan) {
        if (!p.prepU.has(ts)) {
          const pb = new ArrayBuffer(32);
          new Uint32Array(pb).set([r.pw, r.ph, p.bw, p.bh, 0, p.i === 0 ? 1 : 0, 0, 0]);
          const f = new Float32Array(pb);
          f[4] = p.scale;
          f[6] = ts;
          p.prepU.set(ts, r.uni(pb));
        }
        steps[p.prepIndex] = {
          key: `${this.keyPrefix}b${p.i}p`,
          pipe: this.pipes.prep,
          block: p.i,
          kind: 'other',
          bg: r.bg(this.pipes.prep, [p.prepU.get(ts), tensorA, tensorB, r.bufs.fm, r.bufs.x]),
          wg: [Math.ceil(p.bw / 16), Math.ceil(p.bh / 16), 1],
          cost: p.prepCost,
        };
      }
      steps.push({
        key: this.keyPrefix + 'pack',
        pipe: this.pipes.pack,
        block: r.plan.length - 1,
        kind: 'other',
        bg: r.bg(this.pipes.pack, [r.packU, r.bufs.fm, tensorA, tensorB, out.flowView, out.maskView, out.stats]),
        wg: [Math.ceil(r.cw / 16), Math.ceil(r.ch / 16), 1],
        cost: r.cw * r.ch * 30,
      });
      return steps;
    }

    destroyResources() {
      if (!this.res) return;
      for (const b of Object.values(this.res.bufs)) b.destroy();
      for (const b of this.res.uniforms) b.destroy();
      for (const hit of this.res.headCache.values()) for (const b of hit.uniforms) b.destroy();
      this.res = null;
    }

    destroy() {
      this.destroyResources();
      if (!this.parent) this.wbuf.destroy();
    }
  }

  S.RifeNet = RifeNet;
})();
