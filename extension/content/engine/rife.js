// RIFE v4.6 (IFNet, Practical-RIFE by hzwer, MIT) implemented on WebGPU compute shaders.
// One inference per source pair at timestep 0.5: four coarse-to-fine blocks (scales 8, 4, 2, 1),
// each = block input (warp + downscale) -> 2 strided convs -> 8 residual convs -> deconv + pixel
// shuffle -> upsample and accumulate (flow, mask). The work is returned as a list of steps (one
// dispatch each, with a rough cost) so the caller can spread it over several display frames.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});
  const BU = GPUBufferUsage;

  const BLOCKS = [
    { c: 192, inC: 7, scale: 8 },
    { c: 128, inC: 12, scale: 4 },
    { c: 96, inC: 12, scale: 2 },
    { c: 64, inC: 12, scale: 1 },
  ];
  // Model input sides must be multiples of this (block 0 works at 1/8 scale, then strides 2 x 2).
  const ALIGN = 32;

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
    const ncog = L.cout / 16;
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

  // Split-K factor of a deconv (output 4fw x 4fh, one thread per pixel).
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
  function packWeights(tensors) {
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
    const convW = (wName, bName, betaName) => {
      const w = get(wName);
      const b = get(bName).data;
      const beta = betaName ? get(betaName).data : null;
      const [cout, cin] = w.shape;
      if (cout % 16) throw new Error('出力チャンネル数が 16 の倍数ではありません: ' + wName);
      const groups = cout / 16;
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
      const B = new Float32Array(cout);
      for (let co = 0; co < cout; co++) B[co] = b[co] * (beta ? beta[co] : 1);
      return { cin, cout, wOff: add(W), bOff: add(B) };
    };
    const deconvW = (wName, bName) => {
      const w = get(wName); // [cin, 24, 4, 4]
      const b = get(bName).data;
      const cin = w.shape[0];
      const W = new Float32Array(cin * 4 * 16 * 8);
      for (let ci = 0; ci < cin; ci++) {
        for (let sub = 0; sub < 4; sub++) {
          for (let tap = 0; tap < 16; tap++) {
            const base = ((ci * 4 + sub) * 16 + tap) * 8;
            for (let c = 0; c < 5; c++) W[base + c] = w.data[(ci * 24 + c * 4 + sub) * 16 + tap];
          }
        }
      }
      const B = new Float32Array(32);
      for (let sub = 0; sub < 4; sub++) for (let c = 0; c < 5; c++) B[sub * 8 + c] = b[c * 4 + sub];
      return { cin, wOff: add(W), bOff: add(B) };
    };
    const blocks = BLOCKS.map((_, i) => {
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
    return { blocks, data: all };
  }

  class RifeNet {
    constructor(device, weights) {
      this.device = device;
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
        deconv: comp(W.deconv, 'rife-deconv'),
        accum: comp(W.accum, 'rife-accum'),
        pack: comp(W.pack, 'rife-pack'),
      };
      const { blocks, data } = packWeights(weights.tensors);
      this.layers = blocks;
      this.wbuf = d.createBuffer({ size: data.byteLength, usage: BU.STORAGE | BU.COPY_DST });
      d.queue.writeBuffer(this.wbuf, 0, data);
      this.res = null;
    }

    static modelSize(srcW, srcH, width) {
      const w = Math.max(ALIGN * 4, Math.round(Math.min(width, srcW) / ALIGN) * ALIGN);
      const h = Math.max(ALIGN * 2, Math.round((w * srcH) / srcW / ALIGN) * ALIGN);
      return [w, h];
    }

    // Allocate scratch buffers and the static part of the step list for an mw x mh model input.
    configure(mw, mh) {
      if (this.res && this.res.mw === mw && this.res.mh === mh) return;
      this.destroyResources();
      const d = this.device;
      const uniforms = [];
      const uni = (data) => {
        const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: BU.UNIFORM | BU.COPY_DST });
        d.queue.writeBuffer(b, 0, data);
        uniforms.push(b);
        return b;
      };
      const plan = BLOCKS.map((b, i) => {
        const bw = mw / b.scale;
        const bh = mh / b.scale;
        const h1 = [bw / 2, bh / 2];
        const h2 = [bw / 4, bh / 4];
        return { ...b, i, bw, bh, h1, h2, prepCost: bw * bh * (i === 0 ? 20 : 120) };
      });
      let maxIn = 0;
      let maxMid = 0;
      let maxFeat = 0;
      let maxPart = 0;
      let maxTmp = 0;
      for (const p of plan) {
        const L = this.layers[p.i];
        maxIn = Math.max(maxIn, p.inC * p.bw * p.bh);
        maxMid = Math.max(maxMid, (p.c / 2) * p.h1[0] * p.h1[1]);
        maxFeat = Math.max(maxFeat, p.c * p.h2[0] * p.h2[1]);
        const part = (Lc, stride, ow, oh) => {
          const c = convPlan(Lc, stride, ow, oh);
          if (c.ks > 1) maxPart = Math.max(maxPart, c.ks * Lc.cout * ow * oh);
        };
        part(L.conv0[0], 2, p.h1[0], p.h1[1]);
        part(L.conv0[1], 2, p.h2[0], p.h2[1]);
        part(L.res[0], 1, p.h2[0], p.h2[1]);
        const dp = deconvPlan(L.deconv.cin, p.h2[0], p.h2[1]);
        maxTmp = Math.max(maxTmp, dp.ks * 5 * p.h2[0] * 4 * p.h2[1] * 4);
      }
      const sbuf = (n, extra = 0) => d.createBuffer({ size: Math.max(16, n * 4), usage: BU.STORAGE | extra });
      const bufs = {
        x: sbuf(maxIn),
        mid: sbuf(maxMid),
        f0: sbuf(maxFeat),
        f1: sbuf(maxFeat),
        tmp: sbuf(maxTmp),
        part: sbuf(maxPart),
        fm: sbuf(5 * mw * mh, BU.COPY_SRC), // COPY_SRC: read back by tests
      };
      const steps = []; // static steps (everything except the frame-dependent block inputs)
      const bg = (pipe, entries) =>
        d.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries: entries.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })),
        });
      // Conv layer -> 1 step, or 2 with split-K (partial sums, then reduce + bias + residual + act).
      const convSteps = (block, L, stride, inBuf, iw, ih, outBuf, ow, oh, resBuf) => {
        const c = convPlan(L, stride, ow, oh);
        const pipe = this.pipes[c.key];
        const split = c.ks > 1;
        const u = uni(new Uint32Array([L.cin, L.cout, iw, ih, ow, oh, L.wOff, L.bOff, resBuf ? 1 : 0, 1, c.ncog, c.cinPer, split ? 1 : 0, 0, 0, 0]));
        const kind = stride === 1 ? 'conv1' : 'conv2';
        const conv = { pipe, block, kind, bg: bg(pipe, [u, inBuf, this.wbuf, resBuf || inBuf, split ? bufs.part : outBuf]), wg: c.wg, cost: L.cin * L.cout * 9 * ow * oh };
        if (!split) return [conv];
        const n = L.cout * ow * oh;
        const ru = uni(new Uint32Array([n, ow * oh, c.ks, resBuf ? 1 : 0, L.bOff, 1, 0, 0]));
        const red = this.pipes.reduce;
        return [conv, { pipe: red, block, kind, bg: bg(red, [ru, bufs.part, this.wbuf, resBuf || bufs.part, outBuf]), wg: [Math.ceil(n / 256), 1, 1], cost: n * c.ks }];
      };
      for (const p of plan) {
        const L = this.layers[p.i];
        const b = p.i;
        p.prepIndex = steps.length;
        steps.push(null); // block input, filled per job
        steps.push(...convSteps(b, L.conv0[0], 2, bufs.x, p.bw, p.bh, bufs.mid, p.h1[0], p.h1[1]));
        steps.push(...convSteps(b, L.conv0[1], 2, bufs.mid, p.h1[0], p.h1[1], bufs.f0, p.h2[0], p.h2[1]));
        let cur = bufs.f0;
        let other = bufs.f1;
        for (const R of L.res) {
          steps.push(...convSteps(b, R, 1, cur, p.h2[0], p.h2[1], other, p.h2[0], p.h2[1], cur));
          [cur, other] = [other, cur];
        }
        const [fw, fh] = p.h2;
        const dp = deconvPlan(L.deconv.cin, fw, fh);
        const du = uni(new Uint32Array([L.deconv.cin, fw, fh, L.deconv.wOff, L.deconv.bOff, dp.cinPer, 0, 0]));
        steps.push({
          pipe: this.pipes.deconv,
          block: b,
          kind: 'deconv',
          bg: bg(this.pipes.deconv, [du, cur, this.wbuf, bufs.tmp]),
          wg: [Math.ceil((fw * 4) / 16), Math.ceil((fh * 4) / 16), dp.ks],
          cost: L.deconv.cin * 20 * fw * fh * 4,
        });
        const abuf = new ArrayBuffer(32);
        new Uint32Array(abuf).set([mw, mh, fw * 4, fh * 4, 0, p.i === 0 ? 1 : 0, dp.ks, 0]);
        new Float32Array(abuf)[4] = p.scale;
        steps.push({
          pipe: this.pipes.accum,
          block: b,
          kind: 'other',
          bg: bg(this.pipes.accum, [uni(abuf), bufs.tmp, bufs.fm]),
          wg: [Math.ceil(mw / 16), Math.ceil(mh / 16), 1],
          cost: mw * mh * 40,
        });
        const pb = new ArrayBuffer(32);
        new Uint32Array(pb).set([mw, mh, p.bw, p.bh, 0, p.i === 0 ? 1 : 0, 0, 0]);
        new Float32Array(pb)[4] = p.scale;
        p.prepU = uni(pb);
      }
      const packCost = mw * mh * 30;
      let totalCost = packCost;
      for (const s of steps) if (s) totalCost += s.cost;
      for (const p of plan) totalCost += p.prepCost;
      this.res = { mw, mh, plan, bufs, steps, uniforms, packU: uni(new Uint32Array([mw, mh, 0, 0])), packCost, totalCost };
    }

    // Full step list for the pair (tensors a -> b); the last step writes the frame's flow / mask
    // textures and scene statistics.
    job(tensorA, tensorB, out) {
      const d = this.device;
      const r = this.res;
      const steps = r.steps.slice();
      for (const p of r.plan) {
        steps[p.prepIndex] = {
          pipe: this.pipes.prep,
          block: p.i,
          kind: 'other',
          bg: d.createBindGroup({
            layout: this.pipes.prep.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: p.prepU } },
              { binding: 1, resource: { buffer: tensorA } },
              { binding: 2, resource: { buffer: tensorB } },
              { binding: 3, resource: { buffer: r.bufs.fm } },
              { binding: 4, resource: { buffer: r.bufs.x } },
            ],
          }),
          wg: [Math.ceil(p.bw / 16), Math.ceil(p.bh / 16), 1],
          cost: p.prepCost,
        };
      }
      steps.push({
        pipe: this.pipes.pack,
        block: 3,
        kind: 'other',
        bg: d.createBindGroup({
          layout: this.pipes.pack.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: r.packU } },
            { binding: 1, resource: { buffer: r.bufs.fm } },
            { binding: 2, resource: { buffer: tensorA } },
            { binding: 3, resource: { buffer: tensorB } },
            { binding: 4, resource: out.flowView },
            { binding: 5, resource: out.maskView },
            { binding: 6, resource: { buffer: out.stats } },
          ],
        }),
        wg: [Math.ceil(r.mw / 16), Math.ceil(r.mh / 16), 1],
        cost: r.packCost,
        clearStats: out.stats,
      });
      return steps;
    }

    destroyResources() {
      if (!this.res) return;
      for (const b of Object.values(this.res.bufs)) b.destroy();
      for (const b of this.res.uniforms) b.destroy();
      this.res = null;
    }

    destroy() {
      this.destroyResources();
      this.wbuf.destroy();
    }
  }

  S.RifeNet = RifeNet;
  S.RIFE_ALIGN = ALIGN;
})();
