// Inference benchmark: RIFE job GPU time per preset for a 1920x1080 source, plus a per-kernel breakdown.
(() => {
  const S = globalThis.NNF;
  const $ = (id) => document.getElementById(id);
  const NAMES = { fast: '速度優先', standard: '標準', quality: '高画質' };
  const SRC = [1920, 1080];
  const RUNS = 8;

  const log = (s) => ($('log').textContent += s + '\n');

  const MODEL_NAMES = { 'rife-4.25': 'RIFE 4.25', 'rife-4.25-lite': 'RIFE 4.25 lite' };

  // Inputs and outputs of one job at mw x mh (content does not change the cost).
  function makeJob(device, net, mw, mh) {
    net.configure(mw, mh);
    const { pw, ph } = net.res;
    const tensor = (seed) => {
      const a = new Float32Array(S.RifeNet.tensorSize(pw, ph));
      let x = seed;
      for (let i = 0; i < a.length; i++) {
        x = (x * 1664525 + 1013904223) >>> 0;
        a[i] = (x >>> 8) / 16777216;
      }
      const b = device.createBuffer({ size: a.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, a);
      return b;
    };
    const flow = device.createTexture({ size: [mw, mh], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    const mask = device.createTexture({ size: [mw, mh], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    const stats = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const a = tensor(1);
    const b = tensor(2);
    const steps = [...net.headSteps(b), ...net.job(a, b, { flowView: flow.createView(), maskView: mask.createView(), stats })];
    return { steps, destroy: () => [a, b, flow, mask, stats].forEach((r) => r.destroy()) };
  }

  // GPU ms of each group of steps (one compute pass per group, timestamp queries when available).
  async function timeGroups(device, groups) {
    const ts = device.features.has('timestamp-query');
    const enc = device.createCommandEncoder();
    let qs = null;
    if (ts) qs = device.createQuerySet({ type: 'timestamp', count: groups.length * 2 });
    const t0 = performance.now();
    groups.forEach((g, i) => {
      const cp = enc.beginComputePass(ts ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } } : undefined);
      for (const s of g) {
        cp.setPipeline(s.pipe);
        cp.setBindGroup(0, s.bg);
        cp.dispatchWorkgroups(...s.wg);
      }
      cp.end();
    });
    if (!ts) {
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      return [performance.now() - t0];
    }
    const size = groups.length * 16;
    const resolve = device.createBuffer({ size, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.resolveQuerySet(qs, 0, groups.length * 2, resolve, 0);
    enc.copyBufferToBuffer(resolve, 0, read, 0, size);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(read.getMappedRange());
    const out = groups.map((_, i) => Number(t[2 * i + 1] - t[2 * i]) / 1e6);
    read.unmap();
    [qs, resolve, read].forEach((r) => r.destroy());
    return out;
  }

  const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
  const loadCell = (ms, fps) => {
    const pct = (ms * fps) / 10;
    const cls = pct <= 50 ? 'good' : pct <= 70 ? 'warn' : 'bad';
    return `<td class="${cls}">${pct.toFixed(0)}%</td>`;
  };

  const KIND = { conv1: '3×3 畳み込み（残差ブロック）', conv2: '3×3 畳み込み（ストライド 2）', deconv: '転置畳み込み', other: '入力の準備・ワープ・合成' };
  const blockName = (s, net) => (s.block < 0 ? '特徴抽出（Head）' : `段 ${s.block + 1}（1/${net.spec.scales[s.block]} 解像度）`);

  async function run() {
    $('run').disabled = true;
    $('log').textContent = '';
    const tbody = $('result').querySelector('tbody');
    tbody.innerHTML = '';
    try {
      const device = await S.getDevice();
      let err = null;
      device.onuncapturederror = (e) => (err = err || e.error.message);
      const info = (await navigator.gpu.requestAdapter())?.info || {};
      $('gpu').textContent =
        `GPU: ${[info.vendor, info.architecture, info.description].filter(Boolean).join(' / ') || '不明'}` +
        (device.features.has('timestamp-query') ? '' : '（タイムスタンプ非対応のため、待ち時間を含む概算です）');
      $('result').hidden = false;
      $('note').hidden = false;
      for (const preset of ['fast', 'standard', 'quality']) {
        const P = S.PRESETS[preset];
        const net = await S.getRifeNet(device, P.model);
        const { cw: mw, ch: mh } = S.RifeNet.modelSize(SRC[0], SRC[1], P.modelWidth, net.spec.modulo);
        const job = makeJob(device, net, mw, mh);
        const cost = job.steps.reduce((a, s) => a + s.cost, 0);
        await timeGroups(device, [job.steps]); // warm-up (pipeline compilation, caches)
        const times = [];
        for (let i = 0; i < RUNS; i++) times.push((await timeGroups(device, [job.steps]))[0]);
        const ms = median(times);
        tbody.insertAdjacentHTML(
          'beforeend',
          `<tr><td>${NAMES[preset]}</td><td>${MODEL_NAMES[P.model]} ${mw}×${mh}</td><td>${ms.toFixed(1)} ms</td>${loadCell(ms, 24)}${loadCell(ms, 30)}${loadCell(ms, 60)}</tr>`
        );
        log(`${preset} ${mw}x${mh}: median ${ms.toFixed(2)} ms (${times.map((t) => t.toFixed(1)).join(', ')}) / ${(cost / 1e9).toFixed(1)} G 積和 → ${((2 * cost) / ms / 1e6).toFixed(0)} GFLOPS 相当`);
        if (preset === 'standard' && device.features.has('timestamp-query')) {
          const per = await timeGroups(device, job.steps.map((s) => [s]));
          const total = per.reduce((a, b) => a + b, 0);
          const rows = (label) => {
            const sum = new Map();
            job.steps.forEach((s, i) => sum.set(label(s), (sum.get(label(s)) || 0) + per[i]));
            return [...sum].map(([k, v]) => `<tr><td>${k}</td><td>${v.toFixed(2)} ms</td><td>${((100 * v) / total).toFixed(0)}%</td></tr>`).join('');
          };
          const bd = $('breakdown').querySelector('tbody');
          bd.innerHTML = rows((s) => KIND[s.kind]) + '<tr><td colspan="3"></td></tr>' + rows((s) => blockName(s, net));
          log(`standard: ${job.steps.length} dispatches, sum of single-step passes ${total.toFixed(2)} ms`);
          $('breakdown').hidden = false;
          $('bdTitle').hidden = false;
        }
        job.destroy();
        await new Promise((r) => setTimeout(r, 50));
      }
      if (err) log('GPU エラー: ' + err);
      log(`${navigator.userAgent}`);
    } catch (e) {
      log('エラー: ' + e.message);
    }
    $('run').disabled = false;
  }

  $('run').onclick = run;
})();
