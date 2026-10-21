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
    conv2: { stride: 2, tx: 16, ty: 8, px: 2, py: 1, ci: 2, co: 32 }, // fastest of the benchmark's sweep on an RTX 3080
  };
  // Alternative tilings per kind, tried by the auto-tuner (index 0 = the default above). Which is
  // fastest depends on the GPU and the layer's shape.
  const CONV_VARIANTS = {
    conv1: [CONV_CFG.conv1, { stride: 1, tx: 16, ty: 8, px: 2, py: 2, ci: 4, co: 32 }, { stride: 1, tx: 16, ty: 16, px: 2, py: 1, ci: 4 }],
    conv1s: [CONV_CFG.conv1s, { stride: 1, tx: 8, ty: 8, px: 1, py: 1, ci: 4, co: 32 }, CONV_CFG.conv1],
    conv2: [CONV_CFG.conv2, { stride: 2, tx: 8, ty: 8, px: 2, py: 2, ci: 2 }, { stride: 2, tx: 8, ty: 8, px: 2, py: 2, ci: 2, co: 32 }],
  };
  // each also without split-K (appended, so stored indices keep their meaning)
  for (const k of Object.keys(CONV_VARIANTS)) CONV_VARIANTS[k] = [...CONV_VARIANTS[k], ...CONV_VARIANTS[k].map((c) => ({ ...c, nosplit: true }))];
  // shared memory of a tiling (input tile + weights)
  const smemBytes = (c) =>
    4 * (c.ci * ((c.tx * c.px - 1) * c.stride + 3) * ((c.ty * c.py - 1) * c.stride + 3) + ((c.co || 16) / 16) * c.ci * 36 * 4);
  // Workgroups a dispatch should offer to fill a large GPU; small layers get split-K below this.
  const TARGET_WG = 512;

  // Shader variant, workgroup grid and split-K factor of a 3x3 conv layer.
  function convPlan(L, stride, ow, oh, cfgs = CONV_CFG) {
    const tiles = (cfg) => Math.ceil(ow / (cfg.tx * cfg.px)) * Math.ceil(oh / (cfg.ty * cfg.py));
    const groups = (cfg) => Math.ceil(L.cout / (cfg.co || 16));
    let key = stride === 1 ? 'conv1' : 'conv2';
    if (stride === 1 && tiles(cfgs.conv1) * groups(cfgs.conv1) < TARGET_WG / 2) key = 'conv1s';
    const cfg = cfgs[key];
    const ncog = groups(cfg);
    const wgs = tiles(cfg) * ncog;
    // (nosplit: a variant of the auto-tuner; on some GPUs the split costs more than it gains)
    let ks = cfg.nosplit ? 1 : Math.max(1, Math.min(8, Math.floor(L.cin / (cfg.ci * 4)), Math.ceil(TARGET_WG / wgs)));
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
      this.convPipes = new Map(); // tiling -> pipeline (shared with siblings)
      this.choice = new Map(); // 'layer key@size' -> index into CONV_VARIANTS (auto-tuner; shared with siblings)
      this.res = null;
      this.keyPrefix = '';
      this.batch = 1; // inferences run together (items of every buffer one after another)
    }

    pipeFor(cfg) {
      const { nosplit, ...shader } = cfg; // (nosplit only changes the dispatch)
      const k = JSON.stringify(shader);
      let p = this.convPipes.get(k);
      if (!p) {
        p = this.device.createComputePipeline({ layout: 'auto', compute: { module: this.device.createShaderModule({ code: S.WGSL.conv(shader) }), entryPoint: 'main' } });
        this.convPipes.set(k, p);
      }
      return p;
    }

    // Time the tilings of every conv layer of the blocks at size cw x ch (own scratch buffers, so
    // that inference in flight is not disturbed) and remember for each layer the fastest, where it is
    // at least 5 % faster than the default. Needs timestamp queries. Returns the number of layers
    // that use a non-default tiling.
    async tune(cw, ch) {
      const d = this.device;
      if (!d.features.has('timestamp-query')) return 0;
      const t = this.sibling('T');
      t.configure(cw, ch);
      const lim = d.limits.maxComputeWorkgroupStorageSize;
      const tests = [];
      for (const a of t.res.convCalls) {
        const [key, , L, stride, , , , , ow, oh] = a;
        const kind = convPlan(L, stride, ow, oh).key;
        CONV_VARIANTS[kind].forEach((cfg, idx) => {
          if (smemBytes(cfg) <= lim) tests.push({ lk: `${key}@${ow}x${oh}`, idx, steps: t.res.convSteps(...a.slice(0, 10), a[10], idx), ms: [] });
        });
      }
      const REPEAT = 3;
      for (let round = -1; round < 5; round++) {
        // (round -1: warm-up, pipeline compilation)
        const qs = d.createQuerySet({ type: 'timestamp', count: 2 * tests.length });
        const enc = d.createCommandEncoder();
        tests.forEach((x, i) => {
          const cp = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
          for (let r = 0; r < REPEAT; r++) {
            for (const s of x.steps) {
              cp.setPipeline(s.pipe);
              cp.setBindGroup(0, s.bg);
              cp.dispatchWorkgroups(...s.wg);
            }
          }
          cp.end();
        });
        const size = 16 * tests.length;
        const resolve = d.createBuffer({ size, usage: BU.QUERY_RESOLVE | BU.COPY_SRC });
        const read = d.createBuffer({ size, usage: BU.COPY_DST | BU.MAP_READ });
        enc.resolveQuerySet(qs, 0, 2 * tests.length, resolve, 0);
        enc.copyBufferToBuffer(resolve, 0, read, 0, size);
        d.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const ts = new BigInt64Array(read.getMappedRange());
        if (round >= 0) tests.forEach((x, i) => x.ms.push(Number(ts[2 * i + 1] - ts[2 * i]) / 1e6 / REPEAT));
        read.unmap();
        [qs, resolve, read].forEach((o) => o.destroy());
      }
      t.destroyResources();
      const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
      const byLayer = new Map();
      for (const x of tests) {
        if (!byLayer.has(x.lk)) byLayer.set(x.lk, []);
        byLayer.get(x.lk).push({ idx: x.idx, ms: med(x.ms) });
      }
      for (const [lk, rs] of byLayer) {
        const def = rs.find((r) => r.idx === 0);
        const best = rs.reduce((a, b) => (b.ms < a.ms ? b : a));
        if (def && best.idx !== 0 && best.ms < 0.95 * def.ms) this.choice.set(lk, best.idx);
        else this.choice.delete(lk);
      }
      return [...byLayer.keys()].filter((lk) => this.choice.has(lk)).length;
    }

    // A second network sharing weights and pipelines, with its own size and scratch buffers
    // (step keys prefixed with `prefix` so their timings are kept apart).
    sibling(prefix, batch = 1) {
      const n = Object.create(RifeNet.prototype);
      Object.assign(n, this, { res: null, keyPrefix: prefix, parent: this, batch });
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
        // (any tiling the auto-tuner may pick)
        for (const cfg of CONV_VARIANTS[convPlan(L, stride, ow, oh).key]) {
          const c = convPlan(L, stride, ow, oh, { conv1: cfg, conv1s: cfg, conv2: cfg });
          if (c.ks > 1) maxPart = Math.max(maxPart, c.ks * L.cout * ow * oh);
        }
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
      const B = this.batch;
      const bufs = {
        x: sbuf(B * maxIn),
        mid: sbuf(B * maxMid),
        f0: sbuf(B * maxFeat),
        f1: sbuf(B * maxFeat),
        part: sbuf(B * maxPart),
        tmp: sbuf(B * maxTmp),
        fm: sbuf(B * NC * pw * ph, BU.COPY_SRC), // COPY_SRC: read back by tests
        h0: sbuf(16 * hw * hh),
        h1: sbuf(16 * hw * hh),
      };
      const bg = (pipe, entries) =>
        d.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries: entries.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
        });
      // Conv layer -> 1 step, or 2 with split-K (partial sums, then reduce + bias + residual + act).
      // (variant: index into CONV_VARIANTS; default: the auto-tuner's choice for this layer, else 0)
      // (nb: batch items; the Head runs one frame at a time)
      const convSteps = (key, block, L, stride, inBuf, iw, ih, outBuf, ow, oh, resBuf, variant, nb = B) => {
        const base = convPlan(L, stride, ow, oh);
        const vi = variant != null ? variant : this.choice.get(`${key}@${ow}x${oh}`) || 0;
        const cfg = CONV_VARIANTS[base.key][vi];
        const c = vi === 0 ? base : { ...convPlan(L, stride, ow, oh, { conv1: cfg, conv1s: cfg, conv2: cfg }), key: base.key };
        const pipe = vi === 0 ? this.pipes[c.key] : this.pipeFor(cfg);
        const split = c.ks > 1;
        const u = uni(new Uint32Array([L.cin, L.cout, iw, ih, ow, oh, L.wOff, L.bOff, resBuf ? 1 : 0, 1, c.ncog, c.cinPer, split ? 1 : 0, c.ks, 0, 0]));
        const kind = stride === 1 ? 'conv1' : 'conv2';
        const desc = `${c.key}${vi ? `/${vi}` : ''} ${L.cin}→${L.cout} ${ow}×${oh}${split ? ` k${c.ks}` : ''}`; // (for profiling)
        const layer = { L, stride, iw, ih, ow, oh }; // (for the benchmark's kernel sweep)
        const conv = { key: this.keyPrefix + key, pipe, block, kind, variant: desc, layer, bg: bg(pipe, [u, inBuf, this.wbuf, resBuf || inBuf, split ? bufs.part : outBuf]), wg: [c.wg[0], c.wg[1], c.wg[2] * nb], cost: nb * L.cin * L.cout * 9 * ow * oh };
        if (!split) return [conv];
        const n = L.cout * ow * oh;
        const ru = uni(new Uint32Array([n, ow * oh, c.ks, resBuf ? 1 : 0, L.bOff, 1, 0, 0]));
        const red = this.pipes.reduce;
        return [conv, { key: this.keyPrefix + key + 'r', pipe: red, block, kind, bg: bg(red, [ru, bufs.part, this.wbuf, resBuf || bufs.part, outBuf]), wg: [Math.ceil(n / 256), nb, 1], cost: nb * n * c.ks }];
      };
      const steps = []; // static steps (block inputs are filled per job)
      const convCalls = []; // arguments of the blocks' conv layers (for the auto-tuner)
      const blockConv = (...a) => {
        convCalls.push(a);
        return convSteps(...a);
      };
      for (const p of plan) {
        const { L, i: b } = p;
        p.prepIndex = steps.length;
        steps.push(null);
        steps.push(...blockConv(`b${b}c0`, b, L.conv0[0], 2, bufs.x, p.bw, p.bh, bufs.mid, ...p.h1));
        steps.push(...blockConv(`b${b}c1`, b, L.conv0[1], 2, bufs.mid, ...p.h1, bufs.f0, ...p.h2));
        let cur = bufs.f0;
        let other = bufs.f1;
        L.res.forEach((R, j) => {
          steps.push(...blockConv(`b${b}r${j}`, b, R, 1, cur, ...p.h2, other, ...p.h2, cur));
          [cur, other] = [other, cur];
        });
        const [fw, fh] = p.h2;
        const dp = deconvPlan(L.deconv.cin, fw, fh);
        const du = uni(new Uint32Array([L.deconv.cin, fw, fh, L.deconv.wOff, L.deconv.bOff, dp.cinPer, dp.ks, 0]));
        steps.push({
          key: `${this.keyPrefix}b${b}d`,
          pipe: this.pipes.deconv,
          block: b,
          kind: 'deconv',
          bg: bg(this.pipes.deconv, [du, cur, this.wbuf, bufs.tmp]),
          wg: [Math.ceil(fw / 8), Math.ceil(fh / 8), 4 * dp.ks * B],
          cost: B * L.deconv.cin * NC * fw * fh * 4,
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
            wg: [Math.min(65535, Math.ceil(n / 256)), Math.ceil(n / 256 / 65535), B],
            cost: B * n * dp.ks,
          });
        }
        const abuf = new ArrayBuffer(48);
        new Uint32Array(abuf).set([pw, ph, fw * 4, fh * 4, 0, b === 0 ? 1 : 0, 1, NC, dp.ks * NC * fw * 4 * fh * 4, NC * pw * ph, 0, 0]);
        new Float32Array(abuf)[4] = p.scale;
        steps.push({
          key: `${this.keyPrefix}b${b}a`,
          pipe: this.pipes.accum,
          block: b,
          kind: 'other',
          bg: bg(this.pipes.accum, [uni(abuf), bufs.tmp, bufs.fm]),
          wg: [Math.ceil(pw / 16), Math.ceil(ph / 16), B],
          cost: B * pw * ph * NC * 4,
        });
        p.prepU = new Map(); // 'timestep/item' -> uniform buffer (made on first use)
        p.prepCost = p.bw * p.bh * (b === 0 ? 30 : 400);
      }
      this.res = {
        cw, ch, pw, ph, plan, bufs, steps, uniforms, bg, convSteps, convCalls, uni,
        packU: [...Array(B).keys()].map((k) => uni(new Uint32Array([cw, ch, pw, ph, k * NC * pw * ph, 0, 0, 0]))),
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
        ...r.convSteps('h0', -1, c0, 2, tensor, pw, ph, bufs.h0, hw, hh, undefined, undefined, 1),
        ...r.convSteps('h1', -1, c1, 1, bufs.h0, hw, hh, bufs.h1, hw, hh, undefined, undefined, 1),
        ...r.convSteps('h2', -1, c2, 1, bufs.h1, hw, hh, bufs.h0, hw, hh, undefined, undefined, 1),
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
    // last step writes out's flow / mask textures and scene statistics. With batch > 1, `items` lists
    // the pairs ({ a, b, out }) inferred together: job(items, ts).
    job(tensorA, tensorB, out, ts = 0.5) {
      const items = Array.isArray(tensorA) ? tensorA : [{ a: tensorA, b: tensorB, out }];
      if (Array.isArray(tensorA)) ts = tensorB != null ? tensorB : 0.5;
      const r = this.res;
      const steps = [];
      const prepAt = new Map(r.plan.map((p) => [p.prepIndex, p]));
      r.steps.forEach((s, k) => {
        if (s) return steps.push(s);
        // block input of each item (its own frame tensors)
        const p = prepAt.get(k);
        items.forEach((it, n) => {
          const key = `${ts}/${n}`;
          if (!p.prepU.has(key)) {
            const pb = new ArrayBuffer(48);
            new Uint32Array(pb).set([r.pw, r.ph, p.bw, p.bh, 0, p.i === 0 ? 1 : 0, 0, 0, n * p.inC * p.bw * p.bh, n * NC * r.pw * r.ph, 0, 0]);
            const f = new Float32Array(pb);
            f[4] = p.scale;
            f[6] = ts;
            p.prepU.set(key, r.uni(pb));
          }
          steps.push({
            key: `${this.keyPrefix}b${p.i}p`,
            pipe: this.pipes.prep,
            block: p.i,
            kind: 'other',
            bg: r.bg(this.pipes.prep, [p.prepU.get(key), it.a, it.b, r.bufs.fm, r.bufs.x]),
            wg: [Math.ceil(p.bw / 16), Math.ceil(p.bh / 16), 1],
            cost: p.prepCost,
          });
        });
      });
      items.forEach((it, n) =>
        steps.push({
          key: this.keyPrefix + 'pack',
          pipe: this.pipes.pack,
          block: r.plan.length - 1,
          kind: 'other',
          bg: r.bg(this.pipes.pack, [r.packU[n], r.bufs.fm, it.a, it.b, it.out.flowView, it.out.maskView, it.out.stats]),
          wg: [Math.ceil(r.cw / 16), Math.ceil(r.ch / 16), 1],
          cost: r.cw * r.ch * 30,
        })
      );
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
  S.convPlan = convPlan;
  S.CONV_CFG = CONV_CFG;
})();
