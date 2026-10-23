// GPU frame interpolator (AI): frame ingest, RIFE inference and motion-compensated rendering.
// Frames live in a ring of RING slots; the RIFE result of the pair (previous frame -> frame) is
// stored with the later frame. Inference is queued as a job per pair and submitted in chunks by
// pump(), called once per display frame after rendering, so that a long inference never holds up
// the display frames queued behind it.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});
  const RING = 8; // enough for pairs of drawings on threes plus the frames queued behind them
  const BU = GPUBufferUsage;
  const TU = GPUTextureUsage;
  const CHUNK_SHARE = 0.45; // share of a display frame one inference chunk should take
  const STATS_BYTES = 48; // per-pair statistics: scene change (0-1), directing effects (2-8)
  const MAX_INFLIGHT = 2; // inference submissions the GPU may lag behind before pump() waits
  const MEASURE_EVERY = 8; // every n-th inference job is timed step by step
  // Extra RIFE timesteps per pair (besides 0.5): output frames are drawn from the nearest one, so
  // large motion is extrapolated over a quarter of the pair at most instead of half of it.
  const ANCHORS = [0.25, 0.75];
  const TUNE_VERSION = 2; // version of the auto-tuner's set of tilings

  const sameArch = (a, b) => a.modulo === b.modulo && a.scales.join() === b.scales.join();

  class Interpolator {
    constructor(device, net, params) {
      this.device = device;
      this.net = net;
      this.netH = net.sibling('H'); // the same RIFE at half the resolution (large motion)
      // the ensemble's second run, with its own scratch so that the two runs' steps can be interleaved
      // (independent work the GPU can overlap: the low-resolution blocks alone leave it mostly idle)
      this.netS = net.sibling('S');
      // the ensemble's runs inferred together as one batch (fewer, larger dispatches), when memory allows
      this.netE = null;
      // another version of RIFE (same architecture, other weights) for the ensemble's swapped run:
      // the two versions err differently, so their average beats either's own ensemble (setCross)
      this.netX = null;
      this.params = { ...S.TUNING, ...S.PRESETS.standard, ...params };
      const W = S.WGSL;
      const mod = (code, label) => device.createShaderModule({ code, label });
      this.samp = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
      const ingestMod = mod(W.ingest, 'ingest');
      this.ingestPipe = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: ingestMod, entryPoint: 'vs' },
        fragment: { module: ingestMod, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
      });
      this.tensorPipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod(W.toTensor, 'to-tensor'), entryPoint: 'main' } });
      // selection map: models fitted for the ensemble / for RIFE alone
      this.selectPipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod(W.select, 'select'), entryPoint: 'main' } });
      this.selectPipePlain = device.createComputePipeline({ layout: 'auto', compute: { module: mod(W.selectPlain, 'select-plain'), entryPoint: 'main' } });
      this.selectPipeEns4 = device.createComputePipeline({ layout: 'auto', compute: { module: mod(W.selectEns4, 'select-ens4'), entryPoint: 'main' } });
      this.combinePipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod(W.combine, 'combine'), entryPoint: 'main' } });
      this.combine4Pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod(W.combine4, 'combine4'), entryPoint: 'main' } });
      // directing effects (sudden large motion, shake, flash): statistics and decision per pair
      this.fxStatsPipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod(W.fxStats, 'fx-stats'), entryPoint: 'main' } });
      this.fxDecidePipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod(W.fxDecide, 'fx-decide'), entryPoint: 'main' } });
      this.fxCounts = [0, 0, 0]; // pairs shown as drawn for each effect (with fxMask set)
      this.fxReadbacks = [];
      this.bm = new S.BlockMatcher(device, this.samp);
      this.renderMod = mod(W.render, 'render');
      this.renderPipes = new Map();
      this.renderU = device.createBuffer({ size: 64, usage: BU.UNIFORM | BU.COPY_DST });
      this.zeroTex = device.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: TU.TEXTURE_BINDING });
      this.zeroStats = device.createBuffer({ size: STATS_BYTES, usage: BU.STORAGE });
      this.res = null;
      this.seq = 0;
      this.jobs = [];
      this.chunks = 4; // chunks per inference job (pump() keeps this up to date)
      // Measured GPU ms per step key (layers differ a lot in efficiency, so the work is split by
      // measured time, not by arithmetic cost); empty until the first measured job completes.
      this.stepMs = new Map();
      this.jobCount = 0;
      this.inflight = 0; // submitted inference chunks the GPU has not finished yet
      this.pendingRetune = false; // the auto-tuner changed tilings: rebuild the networks between jobs
      // pairs that needed inference (started, or skipped for being behind) / were completed, for the
      // player's load and starvation checks (repeated drawings need none)
      this.demand = 0;
      this.done = 0;
      // GPU timing (sampled): onTiming('me' | 'render', ms) reports inference per source frame and render cost.
      this.onTiming = null;
      this.timing = device.features.has('timestamp-query');
      this.renderCount = 0;
      if (this.timing) {
        this.querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
        this.resolveBuf = device.createBuffer({ size: 512, usage: BU.QUERY_RESOLVE | BU.COPY_SRC });
        this.readbacks = [0, 1, 2, 3].map(() => device.createBuffer({ size: 16, usage: BU.COPY_DST | BU.MAP_READ }));
      }
    }

    // Returns a free readback buffer if this call should be measured, else null.
    sample(counter, every) {
      if (!this.timing || counter % every !== 0) return null;
      return this.readbacks.pop() || null;
    }

    // Resolve query pair (first, first+1) into rb and pass the elapsed GPU ms to done() once mapped.
    finishSample(enc, rb, first, done) {
      const off = first === 0 ? 0 : 256;
      enc.resolveQuerySet(this.querySet, first, 2, this.resolveBuf, off);
      enc.copyBufferToBuffer(this.resolveBuf, off, rb, 0, 16);
      return () =>
        rb.mapAsync(GPUMapMode.READ).then(
          () => {
            const t = new BigInt64Array(rb.getMappedRange());
            const ms = Number(t[1] - t[0]) / 1e6;
            rb.unmap();
            this.readbacks.push(rb);
            if (ms > 0 && ms < 1000) done(ms * (S.debugGpuScale || 1));
          },
          () => {} // device lost / destroyed
        );
    }

    // GPU ms of one inference (one Head + the pair) at the current model size (null until measured).
    jobMs() {
      if (!this.stepMs.size || !this.res) return null;
      let ms = 0;
      for (const k of this.typicalKeys()) ms += this.stepMs.get(k) || 0;
      return ms;
    }

    // Estimated GPU ms of a step (before the first measurement: 1 G multiply-adds per ms).
    stepEst(step) {
      const m = this.stepMs.get(step.key);
      return m != null ? m : step.cost / 1e6 + 0.02;
    }

    // Fold the step times of a measured job into the running estimate and report the job time.
    noteJobTimes(steps, ms) {
      const sum = new Map();
      for (let i = 0; i < steps.length; i++) {
        const [t, n] = sum.get(steps[i].key) || [0, 0];
        sum.set(steps[i].key, [t + ms[i], n + 1]);
      }
      for (const [k, [t, n]] of sum) {
        const v = t / n;
        const old = this.stepMs.get(k);
        this.stepMs.set(k, old == null ? v : old + 0.25 * (v - old));
      }
      if (this.onTiming) this.onTiming('me', this.jobMs());
    }

    // Switch to another network (model change); everything is reallocated.
    setNet(net) {
      if (net === this.net) return;
      const src = this.res && [this.res.srcW, this.res.srcH];
      this.destroyResources();
      this.netH.destroyResources();
      this.netS.destroyResources();
      this.dropEnsNet();
      if (this.netX) this.netX.destroyResources();
      this.net.destroyResources(); // the previous model's scratch (reallocated if switched back)
      this.net = net;
      this.netH = net.sibling('H');
      this.netS = net.sibling('S');
      if (src) this.configure(...src);
    }

    // The second version for the ensemble (a network loaded by the player), or null.
    setCross(net) {
      if (this.netX && net && this.netX.parent === net) return;
      if (this.netX) {
        if (this.res) this.netX.forgetHeads(this.res.frames.filter((f) => f.tensorX).map((f) => f.tensorX));
        this.netX.destroyResources();
      }
      this.netX = net ? net.sibling('X') : null;
      if (this.res) for (const f of this.res.frames) f.featX = false;
    }

    setParams(params) {
      const old = this.params;
      this.params = { ...this.params, ...params };
      const keys = ['maxFrameWidth', 'modelWidth', 'bufferFrames'];
      if (this.res && keys.some((k) => old[k] !== this.params[k])) {
        const { srcW, srcH } = this.res;
        this.destroyResources();
        this.configure(srcW, srcH);
      }
    }

    // (Re)allocate everything for a source of srcW x srcH. Returns true when resources changed.
    configure(srcW, srcH) {
      if (this.res && this.res.srcW === srcW && this.res.srcH === srcH) return false;
      this.destroyResources();
      const d = this.device;
      const p = this.params;
      const fs = Math.min(1, p.maxFrameWidth / srcW);
      const fw = Math.max(16, Math.round(srcW * fs));
      const fh = Math.max(16, Math.round(srcH * fs));
      // (smaller where the largest scratch buffer would exceed the device's binding size)
      let size = S.RifeNet.modelSize(srcW, srcH, p.modelWidth, this.net.spec.modulo);
      for (let w = p.modelWidth; S.RifeNet.scratchBytes(size.pw, size.ph) > d.limits.maxStorageBufferBindingSize && w > 128; ) {
        w *= 0.9;
        size = S.RifeNet.modelSize(srcW, srcH, w, this.net.spec.modulo);
      }
      const { cw: mw, ch: mh, pw, ph } = size;
      this.net.configure(mw, mh);
      const half = S.RifeNet.modelSize(srcW, srcH, Math.round(mw / 2) / S.RifeNet.widthFactor(srcW, srcH), this.net.spec.modulo);
      this.netH.configure(half.cw, half.ch);
      this.netS.destroyResources(); // (configured when first needed)
      if (this.netX) this.netX.destroyResources();
      this.dropEnsNet();
      const uni = (data) => {
        const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: BU.UNIFORM | BU.COPY_DST });
        d.queue.writeBuffer(b, 0, data);
        return b;
      };
      const ingestScale = fw / srcW;
      const ingestU = uni(new Float32Array([1 / fw, 1 / fh, ingestScale < 0.75 ? 2 : 1, 0]));
      const tensorU = uni(new Uint32Array([mw, mh, pw, ph, Math.max(1, Math.min(4, Math.round(fw / mw))), 0, 0, 0]));
      const tensorUM = uni(new Uint32Array([mw, mh, pw, ph, Math.max(1, Math.min(4, Math.round(fw / mw))), 1, 0, 0])); // mirrored
      const tensorUH = uni(new Uint32Array([half.cw, half.ch, half.pw, half.ph, Math.max(1, Math.min(4, Math.round(fw / half.cw))), 0, 0, 0]));
      const frames = [];
      for (let s = 0; s < this.ringSize; s++) {
        const tex = d.createTexture({
          size: [fw, fh],
          format: 'rgba8unorm',
          usage: TU.TEXTURE_BINDING | TU.RENDER_ATTACHMENT | TU.COPY_SRC | TU.COPY_DST,
        });
        const fr = {
          slot: s,
          tex,
          view: tex.createView(),
          tensor: d.createBuffer({ size: S.RifeNet.tensorSize(pw, ph) * 4, usage: BU.STORAGE }),
          flow: d.createTexture({ size: [mw, mh], format: 'rgba16float', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING }),
          mask: d.createTexture({ size: [mw, mh], format: 'rgba8unorm', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING }),
          sel: d.createTexture({ size: [mw, mh], format: 'rgba8unorm', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING | TU.COPY_SRC }),
          tensorH: d.createBuffer({ size: S.RifeNet.tensorSize(half.pw, half.ph) * 4, usage: BU.STORAGE }),
          flowH: d.createTexture({ size: [half.cw, half.ch], format: 'rgba16float', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING }),
          maskH: d.createTexture({ size: [half.cw, half.ch], format: 'rgba8unorm', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING }),
          tensorM: null, // mirrored tensor (4-way ensemble), allocated when first needed
          tensorX: null, // tensor with the second version's Head features (allocated when first needed)
          featX: false, // its Head features are submitted
          hasTensorX: false, // it holds this frame
          featM: false, // Head features of the mirrored tensor are submitted
          hasTensorM: false, // the mirrored tensor holds this frame
          featH: false, // Head features of the half-resolution tensor are submitted
          hasTensorH: false, // the half-resolution tensor holds this frame
          stats: d.createBuffer({ size: STATS_BYTES, usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC }),
          fxU: d.createBuffer({ size: 16, usage: BU.UNIFORM | BU.COPY_DST }),
          fxDU: d.createBuffer({ size: 16, usage: BU.UNIFORM | BU.COPY_DST }),
          id: -1, // sequence number of the frame currently held
          hasPair: false, // RIFE result for (id-1 -> id) is complete
          hasSel: false, // ... together with block matching and the selection map
          feat: false, // Head features of the current frame are submitted
          pairPrevId: -1,
        };
        fr.flowView = fr.flow.createView();
        fr.maskView = fr.mask.createView();
        fr.selView = fr.sel.createView();
        fr.flowHView = fr.flowH.createView();
        fr.maskHView = fr.maskH.createView();
        fr.tensorHBG = d.createBindGroup({
          layout: this.tensorPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.samp },
            { binding: 1, resource: fr.view },
            { binding: 2, resource: { buffer: fr.tensorH } },
            { binding: 3, resource: { buffer: tensorUH } },
          ],
        });
        fr.anchors = ANCHORS.map((ts) => ({ ts, flow: null, mask: null, ready: false })); // textures allocated when first needed
        fr.tensorBG = d.createBindGroup({
          layout: this.tensorPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.samp },
            { binding: 1, resource: fr.view },
            { binding: 2, resource: { buffer: fr.tensor } },
            { binding: 3, resource: { buffer: tensorU } },
          ],
        });
        frames.push(fr);
      }
      this.bm.configure(frames, fw, srcW, srcH);
      const [fsx, fsy] = this.bm.fieldScale();
      const selectU = uni(new Float32Array([fsx, fsy, mw, mh, 0, 0, 0, 0]));
      const selectUH = uni(new Float32Array([fsx, fsy, mw, mh, 1, 0, 0, 0]));
      this.jobs = [];
      this.stepMs = new Map();
      const scratchStats = d.createBuffer({ size: 16, usage: BU.STORAGE });
      // ensemble: the two RIFE runs of a pair before they are combined into the pair's flow / mask
      const ens = {};
      for (const k of ['N', 'S', 'M', 'MS']) {
        ens['flow' + k] = d.createTexture({ size: [mw, mh], format: 'rgba16float', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING });
        ens['mask' + k] = d.createTexture({ size: [mw, mh], format: 'rgba8unorm', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING });
        ens['flow' + k + 'View'] = ens['flow' + k].createView();
        ens['mask' + k + 'View'] = ens['mask' + k].createView();
      }
      this.res = { srcW, srcH, fw, fh, mw, mh, pw, ph, half, ingestU, tensorU, tensorUM, tensorUH, selectU, selectUH, scratchStats, ens, frames, next: 0, jobKeys: new Map() };
      if (this.params.autotune) this.scheduleTune();
      return true;
    }

    // Auto-tuning of the conv tilings for this GPU, model and size: from storage if done before,
    // else measured a few seconds after playback starts (and stored). The networks are rebuilt
    // with the result between two jobs (pump()).
    scheduleTune() {
      const r = this.res;
      const net = this.net;
      const info = this.device.adapterInfo || {};
      // (TUNE_VERSION: bumped when the tilings tried change, so that older results are measured again)
      const key = [TUNE_VERSION, info.vendor, info.architecture, info.device, info.description, net.model, `${r.mw}x${r.mh}`, `${r.half.cw}x${r.half.ch}`].join('|');
      if (this.tuneKey === key) return;
      this.tuneKey = key;
      const current = () => this.tuneKey === key && this.res && this.net === net;
      S.loadTune(key).then((stored) => {
        if (!current()) return;
        if (stored) {
          for (const [k, v] of Object.entries(stored)) net.choice.set(k, v);
          this.pendingRetune = true;
          return;
        }
        setTimeout(async () => {
          if (!current()) return;
          try {
            await net.tune(r.mw, r.mh);
            await net.tune(r.half.cw, r.half.ch);
            if (!current()) return;
            S.saveTune(key, Object.fromEntries(net.choice));
            this.pendingRetune = true;
          } catch (e) {
            // (device lost or the like: keep the default tilings)
          }
        }, 3000);
      });
    }

    rebuildNets() {
      for (const n of [this.net, this.netS, this.netH, this.netE, this.netX]) {
        if (!n || !n.res) continue;
        const { cw, ch } = n.res;
        n.destroyResources();
        n.configure(cw, ch);
      }
    }

    // Step keys of a typical job (one Head + the pair) with the current options, for jobMs().
    typicalKeys() {
      const r = this.res;
      const o = this.options();
      const opt = `${o.useBM}/${o.useHalf}/${o.useAnchors}/${o.useEns}/${o.useCross}`;
      if (!r.jobKeys.has(opt)) {
        const heads = [
          ...this.net.headSteps(r.frames[0].tensor),
          ...(o.useEns === 4 ? this.net.headSteps(this.mirrored(r.frames[0])) : []),
          ...(o.useCross ? this.netX.headSteps(this.crossTensor(r.frames[0])) : []),
          ...(o.useHalf ? this.netH.headSteps(r.frames[0].tensorH) : []),
        ];
        const steps = [...heads, ...this.pairSteps(r.frames[0], r.frames[1], o)];
        r.jobKeys.set(opt, steps.map((st) => st.key));
      }
      return r.jobKeys.get(opt);
    }

    // Which parts a job runs with the current parameters: block matching (hybrid), half-resolution
    // RIFE (multiscale, needs the hybrid's selection map), extra anchors, ensemble (useEns = 2: RIFE
    // also on the swapped pair; 4: and on the mirrored pair both ways), the swapped run by the second
    // version (useCross: two-way ensemble with a second version of the same architecture loaded).
    options() {
      const useBM = this.params.hybrid !== false;
      const e = this.params.ensemble;
      const useEns = e === 4 ? 4 : e ? 2 : 0;
      const useCross = useEns === 2 && !!this.netX && this.params.crossModel !== false && sameArch(this.netX.spec, this.net.spec);
      return { useBM, useHalf: useBM && this.params.multiscale !== false, useAnchors: !!this.params.anchors, useEns, useCross };
    }

    // The frame tensor for the second version (its own Head features; allocated on first use).
    crossTensor(f) {
      if (!this.netX.res) this.netX.configure(this.res.mw, this.res.mh);
      if (!f.tensorX) {
        const r = this.res;
        f.tensorX = this.device.createBuffer({ size: S.RifeNet.tensorSize(r.pw, r.ph) * 4, usage: BU.STORAGE });
        f.tensorXBG = this.device.createBindGroup({
          layout: this.tensorPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.samp },
            { binding: 1, resource: f.view },
            { binding: 2, resource: { buffer: f.tensorX } },
            { binding: 3, resource: { buffer: r.tensorU } },
          ],
        });
      }
      return f.tensorX;
    }

    // The network inferring the ensemble's `items` runs as one batch at the model size, or null where
    // batching is off or its scratch buffers would exceed the device's limits.
    ensNet(items) {
      const r = this.res;
      if (this.params.batchEns === false) return null;
      const lim = this.device.limits;
      const bytes = items * S.RifeNet.scratchBytes(r.pw, r.ph);
      if (bytes > lim.maxStorageBufferBindingSize || bytes > lim.maxBufferSize) return null;
      if (this.netE && this.netE.batch !== items) this.dropEnsNet();
      if (!this.netE) this.netE = this.net.sibling('E' + items, items);
      if (!this.netE.res) this.netE.configure(r.mw, r.mh);
      return this.netE;
    }

    dropEnsNet() {
      if (this.netE) this.netE.destroyResources();
      this.netE = null;
    }

    // The mirrored tensor of a frame record (allocated on first use).
    mirrored(f) {
      if (!f.tensorM) {
        const r = this.res;
        f.tensorM = this.device.createBuffer({ size: S.RifeNet.tensorSize(r.pw, r.ph) * 4, usage: BU.STORAGE });
        f.tensorMBG = this.device.createBindGroup({
          layout: this.tensorPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.samp },
            { binding: 1, resource: f.view },
            { binding: 2, resource: { buffer: f.tensorM } },
            { binding: 3, resource: { buffer: r.tensorUM } },
          ],
        });
      }
      return f.tensorM;
    }

    // Steps of the pair (a -> b) after the Head features: block matching, RIFE at t = 0.5 (full and
    // half resolution), selection map, then the extra anchor timesteps (each part if used).
    // steps.mainEnd = index after which the pair can be drawn.
    pairSteps(a, b, { useBM, useHalf, useAnchors, useEns, useCross }) {
      const steps = [];
      const r = this.res;
      // with the selection map, scene statistics come from it (brightness-compensated)
      const stats = useBM ? r.scratchStats : b.stats;
      // Independent runs (block matching, the RIFE runs of the ensemble and at half resolution, each
      // with its own scratch) are interleaved step by step so that the GPU can overlap them (the
      // low-resolution blocks alone leave it mostly idle).
      const interleave = (runs) => {
        const out = [];
        for (let k = 0; k < Math.max(...runs.map((x) => x.length)); k++) for (const x of runs) if (k < x.length) out.push(x[k]);
        return out;
      };
      const e = r.ens;
      const runs = [];
      if (useBM) runs.push(this.bm.steps(a, b));
      const netE = useEns && !useCross ? this.ensNet(useEns) : null;
      if (useCross) {
        // (two networks: not batched)
        runs.push(this.net.job(a.tensor, b.tensor, { flowView: e.flowNView, maskView: e.maskNView, stats }));
        runs.push(this.netX.job(this.crossTensor(b), this.crossTensor(a), { flowView: e.flowSView, maskView: e.maskSView, stats: r.scratchStats }));
      } else if (netE) {
        const items = [
          { a: a.tensor, b: b.tensor, out: { flowView: e.flowNView, maskView: e.maskNView, stats } },
          { a: b.tensor, b: a.tensor, out: { flowView: e.flowSView, maskView: e.maskSView, stats: r.scratchStats } },
        ];
        if (useEns === 4) {
          const am = this.mirrored(a);
          const bm = this.mirrored(b);
          items.push(
            { a: am, b: bm, out: { flowView: e.flowMView, maskView: e.maskMView, stats: r.scratchStats } },
            { a: bm, b: am, out: { flowView: e.flowMSView, maskView: e.maskMSView, stats: r.scratchStats } }
          );
        }
        runs.push(netE.job(items, 0.5));
      } else if (useEns) {
        if (!this.netS.res) this.netS.configure(r.mw, r.mh);
        runs.push(this.net.job(a.tensor, b.tensor, { flowView: e.flowNView, maskView: e.maskNView, stats }));
        runs.push(this.netS.job(b.tensor, a.tensor, { flowView: e.flowSView, maskView: e.maskSView, stats: r.scratchStats }));
      } else {
        runs.push(this.net.job(a.tensor, b.tensor, { flowView: b.flowView, maskView: b.maskView, stats }));
      }
      if (useHalf) runs.push(this.netH.job(a.tensorH, b.tensorH, { flowView: b.flowHView, maskView: b.maskHView, stats: r.scratchStats }));
      steps.push(...interleave(runs));
      if (useEns) {
        const views = [e.flowNView, e.maskNView, e.flowSView, e.maskSView];
        if (useEns === 4 && !netE) {
          const am = this.mirrored(a);
          const bm = this.mirrored(b);
          steps.push(...interleave([
            this.net.job(am, bm, { flowView: e.flowMView, maskView: e.maskMView, stats: r.scratchStats }),
            this.netS.job(bm, am, { flowView: e.flowMSView, maskView: e.maskMSView, stats: r.scratchStats }),
          ]));
        }
        if (useEns === 4) views.push(e.flowMView, e.maskMView, e.flowMSView, e.maskMSView);
        const pipe = useEns === 4 ? this.combine4Pipe : this.combinePipe;
        steps.push({
          key: 'ens',
          pipe,
          block: this.net.spec.scales.length - 1,
          kind: 'other',
          bg: this.device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [...views, b.flowView, b.maskView].map((v, i) => ({ binding: i, resource: v })),
          }),
          wg: [Math.ceil(r.mw / 16), Math.ceil(r.mh / 16), 1],
          cost: r.mw * r.mh * 10,
        });
      }
      if (useBM) {
        const pipe = useEns === 4 ? this.selectPipeEns4 : useEns ? this.selectPipe : this.selectPipePlain;
        steps.push({
          key: 'sel',
          pipe,
          block: -2,
          kind: 'bm',
          bg: this.device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: useHalf ? r.selectUH : r.selectU } },
              { binding: 1, resource: this.samp },
              { binding: 2, resource: a.view },
              { binding: 3, resource: b.view },
              { binding: 4, resource: b.flowView },
              { binding: 5, resource: b.bm.fieldFView },
              { binding: 6, resource: b.bm.fieldGView },
              { binding: 7, resource: b.selView },
              { binding: 8, resource: b.maskView },
              { binding: 9, resource: b.flowHView },
              { binding: 10, resource: b.maskHView },
              { binding: 11, resource: { buffer: a.bm.mean } },
              { binding: 12, resource: { buffer: b.bm.mean } },
              { binding: 13, resource: { buffer: b.stats } },
            ],
          }),
          wg: [Math.ceil(r.mw / 16), Math.ceil(r.mh / 16), 1],
          cost: r.mw * r.mh * 300,
        });
      }
      steps.push(...this.fxSteps(a, b, useHalf));
      steps.mainEnd = steps.length;
      if (useAnchors) {
        for (const an of b.anchors) {
          if (!an.flow) {
            an.flow = this.device.createTexture({ size: [r.mw, r.mh], format: 'rgba16float', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING });
            an.mask = this.device.createTexture({ size: [r.mw, r.mh], format: 'rgba8unorm', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING });
            an.flowView = an.flow.createView();
            an.maskView = an.mask.createView();
          }
          const js = this.net.job(a.tensor, b.tensor, { flowView: an.flowView, maskView: an.maskView, stats: this.res.scratchStats }, an.ts);
          const id = b.id;
          js[js.length - 1].after = () => {
            if (b.id === id) an.ready = true;
          };
          steps.push(...js);
        }
      }
      return steps;
    }

    // Directing-effect statistics and decision of the pair (a -> b), into b.stats (after its flow is
    // final). Thresholds from params.fxSensitivity (0..100, higher = detected more readily).
    fxSteps(a, b, useHalf) {
      const d = this.device;
      const r = this.res;
      const k = Math.pow(2, (50 - (this.params.fxSensitivity ?? 50)) / 50);
      d.queue.writeBuffer(b.fxU, 0, new Float32Array([r.mw, r.mh, useHalf ? 1 : 0, 0.05 * k]));
      d.queue.writeBuffer(b.fxDU, 0, new Float32Array([0.06 * k, 1 + 1.5 * k, 0.012 * k, 0.3 * k]));
      const bg = (pipe, res) => d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: res.map((x, i) => ({ binding: i, resource: x instanceof GPUBuffer ? { buffer: x } : x })) });
      return [
        {
          key: 'fx',
          pipe: this.fxStatsPipe,
          block: -2,
          kind: 'other',
          bg: bg(this.fxStatsPipe, [b.fxU, this.samp, a.view, b.view, b.flowView, useHalf ? b.flowHView : b.flowView, b.stats]),
          wg: [Math.ceil(r.mw / 32), Math.ceil(r.mh / 32), 1],
          cost: r.mw * r.mh * 4,
        },
        {
          key: 'fxd',
          pipe: this.fxDecidePipe,
          block: -2,
          kind: 'other',
          // (the previous pair's statistics are a's, valid if a's pair was inferred: decided on the GPU)
          bg: bg(this.fxDecidePipe, [b.fxDU, a === b ? this.zeroStats : a.stats, b.stats, a.bm.mean, b.bm.mean]),
          wg: [1, 1, 1],
          cost: 1000,
        },
      ];
    }

    destroyResources() {
      if (!this.res) return;
      this.net.forgetHeads(this.res.frames.map((f) => f.tensor));
      this.net.forgetHeads(this.res.frames.filter((f) => f.tensorM).map((f) => f.tensorM));
      if (this.netX) this.netX.forgetHeads(this.res.frames.filter((f) => f.tensorX).map((f) => f.tensorX));
      this.netH.forgetHeads(this.res.frames.map((f) => f.tensorH));
      for (const f of this.res.frames) {
        f.tex.destroy();
        f.tensor.destroy();
        f.flow.destroy();
        f.mask.destroy();
        f.sel.destroy();
        f.tensorH.destroy();
        if (f.tensorM) f.tensorM.destroy();
        if (f.tensorX) f.tensorX.destroy();
        f.flowH.destroy();
        f.maskH.destroy();
        f.stats.destroy();
        f.fxU.destroy();
        f.fxDU.destroy();
        for (const an of f.anchors) {
          if (an.flow) an.flow.destroy();
          if (an.mask) an.mask.destroy();
        }
      }
      this.res.scratchStats.destroy();
      for (const k of ['flowN', 'maskN', 'flowS', 'maskS', 'flowM', 'maskM', 'flowMS', 'maskMS']) this.res.ens[k].destroy();
      this.bm.destroyResources();
      this.res.ingestU.destroy();
      this.res.tensorU.destroy();
      this.res.tensorUH.destroy();
      this.res.tensorUM.destroy();
      this.res.selectUH.destroy();
      this.res.selectU.destroy();
      this.jobs = [];
      this.res = null;
    }

    destroy() {
      this.destroyResources();
      this.netH.destroyResources();
      this.netS.destroyResources();
      this.dropEnsNet();
      if (this.netX) this.netX.destroyResources();
      this.bm.destroy();
      this.renderU.destroy();
      this.zeroTex.destroy();
      this.zeroStats.destroy();
      for (const b of this.fxReadbacks) b.destroy();
      this.onTiming = null;
      if (this.timing) {
        this.querySet.destroy();
        this.resolveBuf.destroy();
        for (const b of this.readbacks) b.destroy();
      }
    }

    // Upload a video frame (HTMLVideoElement or VideoFrame) into the next ring slot and convert it
    // to the model's input tensor. If `prev` (a frame returned earlier) is given, an inference job
    // for (prev -> this frame) is queued. With detectDup, the frame is also compared with prev:
    // f.dupState becomes 'dup' (a repeat of the previous drawing, f.sameAs = the frame first showing
    // it) or 'distinct' a moment later; jobs wait for it and pair distinct drawings only.
    // Returns the frame record.
    ingest(source, prev, detectDup = false) {
      const d = this.device;
      const r = this.res;
      const f = r.frames[r.next];
      const ext = d.importExternalTexture({ source });
      const enc = d.createCommandEncoder();
      const bg = d.createBindGroup({
        layout: this.ingestPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.samp },
          { binding: 1, resource: ext },
          { binding: 2, resource: { buffer: r.ingestU } },
        ],
      });
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view: f.view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      rp.setPipeline(this.ingestPipe);
      rp.setBindGroup(0, bg);
      rp.draw(3);
      rp.end();
      return this.process(enc, prev, detectDup);
    }

    // Everything after the next ring slot's frame texture is written by `enc`: tensor, luma pyramid,
    // duplicate test, inference job. (Tests write the texture themselves and call this.)
    process(enc, prev, detectDup = false) {
      const d = this.device;
      const r = this.res;
      const f = r.frames[r.next];
      r.next = (r.next + 1) % r.frames.length;
      f.id = ++this.seq;
      f.hasPair = false;
      f.hasSel = false;
      f.hasHalf = false;
      f.pairPrevId = -1;
      f.feat = false;
      f.featH = false;
      f.hasTensorH = this.options().useHalf;
      f.hasTensorM = this.options().useEns === 4;
      f.featM = false;
      f.hasTensorX = this.options().useCross;
      f.featX = false;
      f.sameAs = null;
      f.drawId = f.id; // id of the frame that first showed this drawing
      f.pairDrawId = -1;
      for (const an of f.anchors) an.ready = false;
      const live = !!prev && prev.id > 0 && this.isLive(prev);
      const test = detectDup && live;
      f.dupState = test ? 'unknown' : 'distinct';
      if (test) this.bm.clearDiff(enc, f);
      this.bm.clearMean(enc, f);
      const cp = enc.beginComputePass();
      cp.setPipeline(this.tensorPipe);
      cp.setBindGroup(0, f.tensorBG);
      cp.dispatchWorkgroups(Math.ceil(r.pw / 16), Math.ceil(r.ph / 16));
      if (this.options().useHalf) {
        cp.setBindGroup(0, f.tensorHBG);
        cp.dispatchWorkgroups(Math.ceil(r.half.pw / 16), Math.ceil(r.half.ph / 16));
      }
      if (f.hasTensorM) {
        this.mirrored(f);
        cp.setBindGroup(0, f.tensorMBG);
        cp.dispatchWorkgroups(Math.ceil(r.pw / 16), Math.ceil(r.ph / 16));
      }
      if (f.hasTensorX) {
        this.crossTensor(f);
        cp.setBindGroup(0, f.tensorXBG);
        cp.dispatchWorkgroups(Math.ceil(r.pw / 16), Math.ceil(r.ph / 16));
      }
      this.bm.encodeLuma(cp, f);
      if (test) this.bm.encodeDiff(cp, prev, f);
      cp.end();
      enc.clearBuffer(f.stats);
      d.queue.submit([enc.finish()]);
      if (test) {
        const id = f.id;
        this.bm.readDup(f).then(
          (dup) => {
            if (f.id !== id) return;
            f.dupState = dup ? 'dup' : 'distinct';
            if (dup) {
              f.sameAs = prev.dupState === 'dup' && prev.sameAs ? prev.sameAs : prev;
              f.drawId = prev.drawId;
            }
          },
          () => {}
        );
      }
      if (live) this.queueJob(prev, f);
      return f;
    }

    // Queue the inference for the pair (a -> b); pump() submits it.
    queueJob(a, b) {
      const measure = this.jobCount++ % MEASURE_EVERY === 0;
      this.jobs.push({ a, aId: a.id, b, bId: b.id, steps: null, next: 0, measure, ms: null, pending: 0, marks: [] });
    }

    // Steps of a job, built when it starts: Head features of a and b where missing, then the pair.
    startJob(job) {
      const steps = [];
      const o = this.options();
      // (half resolution was switched on after one of the frames came in: its half tensor is stale)
      if (!job.a.hasTensorH || !job.b.hasTensorH) o.useHalf = false;
      if (o.useEns === 4 && (!job.a.hasTensorM || !job.b.hasTensorM)) o.useEns = 2;
      if (o.useCross && (!job.a.hasTensorX || !job.b.hasTensorX)) o.useCross = false;
      this.demand++;
      for (const f of [job.a, job.b]) {
        if (!f.feat) {
          steps.push(...this.net.headSteps(f.tensor));
          job.marks.push([steps.length, f, 'feat']); // features of f are submitted once steps[0, n) are
        }
        if (o.useEns === 4 && !f.featM) {
          steps.push(...this.net.headSteps(this.mirrored(f)));
          job.marks.push([steps.length, f, 'featM']);
        }
        if (o.useCross && !f.featX) {
          steps.push(...this.netX.headSteps(this.crossTensor(f)));
          job.marks.push([steps.length, f, 'featX']);
        }
        if (o.useHalf && !f.featH) {
          steps.push(...this.netH.headSteps(f.tensorH));
          job.marks.push([steps.length, f, 'featH']);
        }
      }
      job.useBM = o.useBM;
      job.useHalf = o.useHalf;
      const ps = this.pairSteps(job.a, job.b, o);
      job.mainEnd = steps.length + ps.mainEnd;
      steps.push(...ps);
      job.steps = steps;
      if (job.measure) job.ms = new Array(steps.length).fill(0);
    }

    // Ring slots: the base plus the frames of the user's buffer (params.bufferFrames).
    get ringSize() {
      return RING + (this.params.bufferFrames || 0);
    }

    // A frame record stays usable until its slot is reused.
    isLive(frame) {
      return !!this.res && this.res.frames[frame.slot] === frame && frame.id > 0;
    }

    // Forget queued inference (seek, new source).
    cancelJobs() {
      this.jobs = [];
    }

    // Submit the next chunk of queued inference. vsyncMs = display frame interval, srcMs = source
    // frame interval. A job is split into chunks of about CHUNK_SHARE of a display frame of GPU time
    // each, but into no more chunks than there are display frames per source frame (or it falls
    // behind). One pump never spans two jobs.
    pump(vsyncMs, srcMs) {
      const d = this.device;
      this.jobs = this.jobs.filter((j) => j.a.id === j.aId && j.b.id === j.bId && this.isLive(j.a) && this.isLive(j.b));
      // behind: skip pairs rather than lag further (beyond what the user's buffer allows), but finish
      // the one already under way
      while (this.jobs.length > 2 + (this.params.bufferFrames || 0)) {
        const [j] = this.jobs.splice(this.jobs[0].next > 0 ? 1 : 0, 1);
        if (j.b.dupState !== 'dup') this.demand++;
      }
      if (this.pendingRetune && !this.jobs.some((j) => j.steps)) {
        // (no job holds steps of the old networks)
        this.pendingRetune = false;
        this.rebuildNets();
      }
      if (!this.jobs.length || !this.res || this.inflight >= MAX_INFLIGHT) return;
      const job = this.jobs[0];
      if (!job.steps) {
        // repeated drawings: wait for the duplicate test, drop pairs ending on a repeat and start
        // pairs from the frame that first showed a repeated drawing
        if (job.a.dupState === 'unknown' || job.b.dupState === 'unknown') return;
        if (job.b.dupState === 'dup') {
          this.jobs.shift();
          return this.pump(vsyncMs, srcMs);
        }
        if (job.a.dupState === 'dup') {
          // the frame that first showed the drawing, or (gone from the ring) this repeat of it
          const s = job.a.sameAs;
          if (s && this.isLive(s) && s.drawId === job.a.drawId) {
            job.a = s;
            job.aId = s.id;
          }
        }
        this.startJob(job);
      }
      const steps = job.steps;
      let jobMs = 0;
      for (const st of steps) jobMs += this.stepEst(st);
      const maxChunks = Math.max(1, Math.floor((0.75 * srcMs) / vsyncMs));
      const target = Math.max(CHUNK_SHARE * vsyncMs, jobMs / maxChunks);
      // chunk count of a whole job at this target (for the player's latency allowance)
      let chunks = 1;
      for (let i = 0, acc = 0; i < steps.length; i++) {
        const e = this.stepEst(steps[i]);
        if (acc > 0 && acc + e > target) {
          chunks++;
          acc = 0;
        }
        acc += e;
      }
      this.chunks = chunks;
      const first = job.next;
      let last = first;
      for (let acc = 0; last < steps.length; last++) {
        const e = this.stepEst(steps[last]);
        if (last > first && acc + e > target) break;
        acc += e;
      }
      const enc = d.createCommandEncoder();
      const timed = this.timing && job.measure;
      let qs = null;
      if (timed) qs = d.createQuerySet({ type: 'timestamp', count: 2 * (last - first) });
      let cp = null;
      for (let i = first; i < last; i++) {
        const s = steps[i];
        if (!cp || timed) {
          if (cp) cp.end();
          cp = enc.beginComputePass(
            timed ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * (i - first), endOfPassWriteIndex: 2 * (i - first) + 1 } } : undefined
          );
        }
        cp.setPipeline(s.pipe);
        cp.setBindGroup(0, s.bg);
        cp.dispatchWorkgroups(...s.wg);
        if (s.after) s.after();
      }
      cp.end();
      job.next = last;
      for (const [n, f, flag] of job.marks) if (last >= n) f[flag] = true;
      let read = null;
      let fxRead = null; // the pair's directing-effect decision, read back for the player's statistics
      if (timed) {
        const size = 16 * (last - first);
        const resolve = d.createBuffer({ size, usage: BU.QUERY_RESOLVE | BU.COPY_SRC });
        read = d.createBuffer({ size, usage: BU.COPY_DST | BU.MAP_READ });
        enc.resolveQuerySet(qs, 0, 2 * (last - first), resolve, 0);
        enc.copyBufferToBuffer(resolve, 0, read, 0, size);
        read.resolve = resolve;
      }
      if (!job.mainDone && last >= job.mainEnd) {
        // the pair can be drawn (extra anchors may still follow)
        job.mainDone = true;
        job.b.hasPair = true;
        job.b.hasSel = job.useBM;
        job.b.hasHalf = job.useHalf;
        job.b.pairPrevId = job.aId;
        job.b.pairDrawId = job.a.drawId;
        this.done++;
        if (this.params.fxMask) fxRead = this.fxReadbacks.pop() || d.createBuffer({ size: STATS_BYTES, usage: BU.COPY_DST | BU.MAP_READ });
        if (fxRead) enc.copyBufferToBuffer(job.b.stats, 0, fxRead, 0, STATS_BYTES);
      }
      if (last === steps.length) this.jobs.shift();
      d.queue.submit([enc.finish()]);
      const done = () => {
        if (--job.pending === 0 && job.next === steps.length && job.ms) this.noteJobTimes(steps, job.ms);
      };
      if (job.ms) job.pending++;
      if (fxRead) {
        fxRead.mapAsync(GPUMapMode.READ).then(
          () => {
            const flags = new Uint32Array(fxRead.getMappedRange())[7];
            fxRead.unmap();
            for (let i = 0; i < 3; i++) if (flags & this.params.fxMask & (1 << i)) this.fxCounts[i]++;
            if (this.fxReadbacks.length < 4) this.fxReadbacks.push(fxRead);
            else fxRead.destroy();
          },
          () => fxRead.destroy()
        );
      }
      if (read) {
        read.mapAsync(GPUMapMode.READ).then(
          () => {
            const t = new BigInt64Array(read.getMappedRange());
            for (let i = first; i < last; i++) {
              const ms = Number(t[2 * (i - first) + 1] - t[2 * (i - first)]) / 1e6;
              job.ms[i] = ms >= 0 && ms < 1000 ? ms * (S.debugGpuScale || 1) : 0;
            }
            read.unmap();
            [qs, read.resolve, read].forEach((r) => r.destroy());
            done();
          },
          () => {} // device lost / destroyed
        );
      }
      const t0 = performance.now();
      this.inflight++;
      d.queue.onSubmittedWorkDone().then(
        () => {
          this.inflight--;
          if (!this.timing && job.ms) {
            // no timestamp queries: spread the chunk's completion latency (an overestimate) over its steps by cost
            let cost = 0;
            for (let i = first; i < last; i++) cost += steps[i].cost + 1e5;
            const ms = performance.now() - t0;
            for (let i = first; i < last; i++) job.ms[i] = (ms * (steps[i].cost + 1e5)) / cost;
            done();
          }
        },
        () => {}
      );
    }

    renderPipe(format) {
      let pipe = this.renderPipes.get(format);
      if (!pipe) {
        pipe = this.device.createRenderPipeline({
          layout: 'auto',
          vertex: { module: this.renderMod, entryPoint: 'vs' },
          fragment: { module: this.renderMod, entryPoint: 'fs', targets: [{ format }] },
        });
        this.renderPipes.set(format, pipe);
      }
      return pipe;
    }

    // Is the RIFE result for (a -> b) ready? (a may be any frame showing the same drawing)
    paired(a, b) {
      return !!b && b !== a && this.isLive(b) && b.hasPair && (b.pairPrevId === a.id || (a.drawId != null && b.pairDrawId === a.drawId));
    }

    // Draw frame a..b at phase t into `view` (format `format`). rect = content rectangle
    // [x, y, w, h] in target pixels. opts: { mode, split, masking }.
    // mode: 0 interpolate, 1 frame A, 2 frame B, 3 plain blend, 4 flow, 5 selection map.
    render(view, format, rect, a, b, t, opts = {}) {
      const d = this.device;
      const p = this.params;
      if (!b || !this.isLive(b)) b = a;
      const paired = this.paired(a, b);
      let mode = opts.mode || 0;
      // no inference result (yet): show the nearer frame rather than a ghosting blend
      if (!paired && (mode === 0 || mode >= 4)) mode = b === a || t < 0.5 ? 1 : 2;
      const sel = paired && b.hasSel;
      // extra RIFE anchor nearer to t than 0.5 (ready, same pair), used where 0.5 extrapolates badly
      let anchor = null;
      if (paired) {
        for (const an of b.anchors) if (an.ready && Math.abs(an.ts - t) < Math.abs(0.5 - t) && (!anchor || Math.abs(an.ts - t) < Math.abs(anchor.ts - t))) anchor = an;
      }
      const buf = new ArrayBuffer(64);
      const f = new Float32Array(buf);
      const u = new Uint32Array(buf);
      f.set(rect, 0);
      f.set(this.bm.fieldScale(), 4);
      f[6] = t;
      u[7] = mode;
      f[8] = opts.masking ?? p.masking;
      f[9] = p.sceneFrac;
      f[10] = opts.split || 0;
      f[11] = sel ? (opts.hybrid ?? 1) : -1;
      f[12] = anchor ? anchor.ts : 0.5;
      u[13] = p.fxMask || 0;
      d.queue.writeBuffer(this.renderU, 0, buf);
      const pipe = this.renderPipe(format);
      const zero = this.zeroTex.createView();
      const bg = d.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.renderU } },
          { binding: 1, resource: this.samp },
          { binding: 2, resource: a.view },
          { binding: 3, resource: b.view },
          { binding: 4, resource: paired ? b.flowView : zero },
          { binding: 5, resource: paired ? b.maskView : zero },
          { binding: 6, resource: { buffer: paired ? b.stats : this.zeroStats } },
          { binding: 7, resource: sel ? b.bm.fieldFView : zero },
          { binding: 8, resource: sel ? b.bm.fieldGView : zero },
          { binding: 9, resource: sel ? b.selView : zero },
          { binding: 10, resource: anchor ? anchor.flowView : zero },
          { binding: 11, resource: anchor ? anchor.maskView : zero },
          { binding: 12, resource: sel && b.hasHalf ? b.flowHView : zero },
          { binding: 13, resource: sel && b.hasHalf ? b.maskHView : zero },
          { binding: 14, resource: { buffer: a.bm.mean } },
          { binding: 15, resource: { buffer: b.bm.mean } },
        ],
      });
      const enc = d.createCommandEncoder();
      const rb = this.onTiming ? this.sample(this.renderCount++, 20) : null;
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
        timestampWrites: rb ? { querySet: this.querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 } : undefined,
      });
      rp.setPipeline(pipe);
      rp.setBindGroup(0, bg);
      rp.draw(3);
      rp.end();
      const report = rb ? this.finishSample(enc, rb, 2, (ms) => this.onTiming && this.onTiming('render', ms)) : null;
      d.queue.submit([enc.finish()]);
      if (report) report();
      return paired;
    }

    // Scene-change statistics of the pair ending at frame b: { bad, total } (async, for OSD/tests).
    async readStats(b) {
      const d = this.device;
      const rb = d.createBuffer({ size: 16, usage: BU.COPY_DST | BU.MAP_READ });
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(b.stats, 0, rb, 0, 16);
      d.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const v = new Uint32Array(rb.getMappedRange().slice(0));
      rb.destroy();
      return { bad: v[0], total: v[1] };
    }
  }

  S.Interpolator = Interpolator;
})();
