// GPU frame interpolator (AI): frame ingest, RIFE inference and motion-compensated rendering.
// Frames live in a ring of RING slots; the RIFE result of the pair (previous frame -> frame) is
// stored with the later frame. Inference is queued as a job per pair and submitted in chunks by
// pump(), called once per display frame after rendering, so that a long inference never holds up
// the display frames queued behind it.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});
  const RING = 6;
  const BU = GPUBufferUsage;
  const TU = GPUTextureUsage;
  const CHUNK_SHARE = 0.45; // share of a display frame one inference chunk should take
  const MAX_INFLIGHT = 2; // inference submissions the GPU may lag behind before pump() waits
  const MEASURE_EVERY = 8; // every n-th inference job is timed step by step

  class Interpolator {
    constructor(device, net, params) {
      this.device = device;
      this.net = net;
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
      this.renderMod = mod(W.render, 'render');
      this.renderPipes = new Map();
      this.renderU = device.createBuffer({ size: 48, usage: BU.UNIFORM | BU.COPY_DST });
      this.zeroTex = device.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: TU.TEXTURE_BINDING });
      this.zeroStats = device.createBuffer({ size: 16, usage: BU.STORAGE });
      this.res = null;
      this.seq = 0;
      this.jobs = [];
      this.chunks = 4; // chunks per inference job (pump() keeps this up to date)
      // Measured GPU ms per step key (layers differ a lot in efficiency, so the work is split by
      // measured time, not by arithmetic cost); empty until the first measured job completes.
      this.stepMs = new Map();
      this.jobCount = 0;
      this.inflight = 0; // submitted inference chunks the GPU has not finished yet
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
      for (const k of this.res.jobKeys) ms += this.stepMs.get(k) || 0;
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
      this.net = net;
      if (this.res) {
        const { srcW, srcH } = this.res;
        this.destroyResources();
        this.configure(srcW, srcH);
      }
    }

    setParams(params) {
      const old = this.params;
      this.params = { ...this.params, ...params };
      if (this.res && (old.maxFrameWidth !== this.params.maxFrameWidth || old.modelWidth !== this.params.modelWidth)) {
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
      const { cw: mw, ch: mh, pw, ph } = S.RifeNet.modelSize(srcW, srcH, p.modelWidth, this.net.spec.modulo);
      this.net.configure(mw, mh);
      const uni = (data) => {
        const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: BU.UNIFORM | BU.COPY_DST });
        d.queue.writeBuffer(b, 0, data);
        return b;
      };
      const ingestScale = fw / srcW;
      const ingestU = uni(new Float32Array([1 / fw, 1 / fh, ingestScale < 0.75 ? 2 : 1, 0]));
      const tensorU = uni(new Uint32Array([mw, mh, pw, ph, Math.max(1, Math.min(4, Math.round(fw / mw))), 0, 0, 0]));
      const frames = [];
      for (let s = 0; s < RING; s++) {
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
          stats: d.createBuffer({ size: 16, usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC }),
          id: -1, // sequence number of the frame currently held
          hasPair: false, // RIFE result for (id-1 -> id) is complete
          feat: false, // Head features of the current frame are submitted
          pairPrevId: -1,
        };
        fr.flowView = fr.flow.createView();
        fr.maskView = fr.mask.createView();
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
      this.jobs = [];
      this.stepMs = new Map();
      // step keys of a typical job (one Head + the pair), for jobMs()
      const jobKeys = [...this.net.headSteps(frames[0].tensor), ...this.net.job(frames[0].tensor, frames[1].tensor, frames[1])].map((st) => st.key);
      this.res = { srcW, srcH, fw, fh, mw, mh, pw, ph, ingestU, tensorU, frames, next: 0, jobKeys };
      return true;
    }

    destroyResources() {
      if (!this.res) return;
      for (const f of this.res.frames) {
        f.tex.destroy();
        f.tensor.destroy();
        f.flow.destroy();
        f.mask.destroy();
        f.stats.destroy();
      }
      this.res.ingestU.destroy();
      this.res.tensorU.destroy();
      this.jobs = [];
      this.res = null;
    }

    destroy() {
      this.destroyResources();
      this.renderU.destroy();
      this.zeroTex.destroy();
      this.zeroStats.destroy();
      this.onTiming = null;
      if (this.timing) {
        this.querySet.destroy();
        this.resolveBuf.destroy();
        for (const b of this.readbacks) b.destroy();
      }
    }

    // Upload a video frame (HTMLVideoElement or VideoFrame) into the next ring slot and convert it
    // to the model's input tensor. If `prev` (a frame returned earlier) is given, an inference job
    // for (prev -> this frame) is queued. Returns the frame record.
    ingest(source, prev) {
      const d = this.device;
      const r = this.res;
      const f = r.frames[r.next];
      r.next = (r.next + 1) % RING;
      f.id = ++this.seq;
      f.hasPair = false;
      f.pairPrevId = -1;
      f.feat = false;
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
      const cp = enc.beginComputePass();
      cp.setPipeline(this.tensorPipe);
      cp.setBindGroup(0, f.tensorBG);
      cp.dispatchWorkgroups(Math.ceil(r.pw / 16), Math.ceil(r.ph / 16));
      cp.end();
      enc.clearBuffer(f.stats);
      d.queue.submit([enc.finish()]);
      if (prev && prev.id > 0 && this.isLive(prev)) this.queueJob(prev, f);
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
      for (const f of [job.a, job.b]) {
        if (f.feat) continue;
        steps.push(...this.net.headSteps(f.tensor));
        job.marks.push([steps.length, f]); // features of f are submitted once steps[0, n) are
      }
      steps.push(...this.net.job(job.a.tensor, job.b.tensor, job.b));
      job.steps = steps;
      if (job.measure) job.ms = new Array(steps.length).fill(0);
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
      // behind: skip pairs rather than lag further, but finish the one already under way
      while (this.jobs.length > 2) this.jobs.splice(this.jobs[0].next > 0 ? 1 : 0, 1);
      if (!this.jobs.length || !this.res || this.inflight >= MAX_INFLIGHT) return;
      const job = this.jobs[0];
      if (!job.steps) this.startJob(job);
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
      }
      cp.end();
      job.next = last;
      for (const [n, f] of job.marks) if (last >= n) f.feat = true;
      let read = null;
      if (timed) {
        const size = 16 * (last - first);
        const resolve = d.createBuffer({ size, usage: BU.QUERY_RESOLVE | BU.COPY_SRC });
        read = d.createBuffer({ size, usage: BU.COPY_DST | BU.MAP_READ });
        enc.resolveQuerySet(qs, 0, 2 * (last - first), resolve, 0);
        enc.copyBufferToBuffer(resolve, 0, read, 0, size);
        read.resolve = resolve;
      }
      if (last === steps.length) {
        job.b.hasPair = true;
        job.b.pairPrevId = job.aId;
        this.jobs.shift();
      }
      d.queue.submit([enc.finish()]);
      const done = () => {
        if (--job.pending === 0 && job.next === steps.length && job.ms) this.noteJobTimes(steps, job.ms);
      };
      if (job.ms) job.pending++;
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

    // Is the RIFE result for (a -> b) ready?
    paired(a, b) {
      return !!b && b !== a && this.isLive(b) && b.hasPair && b.pairPrevId === a.id;
    }

    // Draw frame a..b at phase t into `view` (format `format`). rect = content rectangle
    // [x, y, w, h] in target pixels. opts: { mode, split, masking }.
    // mode: 0 interpolate, 1 frame A, 2 frame B, 3 plain blend, 4 flow.
    render(view, format, rect, a, b, t, opts = {}) {
      const d = this.device;
      const p = this.params;
      if (!b || !this.isLive(b)) b = a;
      const paired = this.paired(a, b);
      let mode = opts.mode || 0;
      // no inference result (yet): show the nearer frame rather than a ghosting blend
      if (!paired && (mode === 0 || mode === 4)) mode = b === a || t < 0.5 ? 1 : 2;
      const buf = new ArrayBuffer(48);
      const f = new Float32Array(buf);
      const u = new Uint32Array(buf);
      f.set(rect, 0);
      f[4] = t;
      u[5] = mode;
      f[6] = opts.masking ?? p.masking;
      f[7] = p.sceneFrac;
      f[8] = opts.split || 0;
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
