// Benchmark: GPU time of one interpolation job (Head + block matching + RIFE + selection map, as the
// player runs it) per preset for a 1920x1080 source, the cost of the optional extras, the render
// cost per output frame, and a per-kernel breakdown.
(() => {
  const S = globalThis.NNF;
  const $ = (id) => document.getElementById(id);
  const NAMES = { fast: '速度優先', standard: '標準', quality: '高画質' };
  const MODEL_NAMES = { 'rife-4.25': 'RIFE 4.25', 'rife-4.25-lite': 'RIFE 4.25 lite' };
  const q = new URLSearchParams(location.search);
  const SRC = [+(q.get('w') || 1920), +(q.get('h') || 1080)]; // smaller sizes for quick checks
  const RUNS = 8;

  const log = (s) => ($('log').textContent += s + '\n');

  // An engine for the source size with two synthetic frames ingested; returns the steps of the job
  // between them (one Head, as in steady playback) and the two frame records.
  function makeJob(device, net, params) {
    const eng = new S.Interpolator(device, net, params);
    eng.configure(SRC[0], SRC[1]);
    const r = eng.res;
    const frame = (seed) => {
      const px = new Uint8Array(r.fw * r.fh * 4);
      let x = seed;
      for (let i = 0; i < px.length; i++) {
        x = (x * 1664525 + 1013904223) >>> 0;
        px[i] = x >>> 24;
      }
      return px;
    };
    const put = (seed, prev) => {
      device.queue.writeTexture({ texture: r.frames[r.next].tex }, frame(seed), { bytesPerRow: r.fw * 4 }, [r.fw, r.fh]);
      return eng.process(device.createCommandEncoder(), prev, false);
    };
    const a = put(1, null);
    const b = put(2, a);
    a.feat = true; // its Head ran with the previous pair
    a.featH = true;
    const job = eng.jobs[0];
    eng.startJob(job);
    return { eng, a, b, job, steps: job.steps };
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

  async function timeJob(device, steps) {
    await timeGroups(device, [steps]); // warm-up (pipeline compilation, caches)
    const times = [];
    for (let i = 0; i < RUNS; i++) times.push((await timeGroups(device, [steps]))[0]);
    return { ms: median(times), times };
  }

  // GPU ms of drawing one 1080p output frame of the (now computed) pair.
  async function timeRender(device, j) {
    const { eng, a, b, job } = j;
    b.hasPair = true;
    b.hasSel = job.useBM;
    b.hasHalf = job.useHalf;
    b.pairPrevId = a.id;
    b.pairDrawId = a.drawId;
    const out = device.createTexture({ size: SRC, format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const times = [];
    eng.onTiming = (kind, ms) => kind === 'render' && times.push(ms);
    for (let i = 0; i < 60; i++) {
      eng.render(out.createView(), 'rgba8unorm', [0, 0, ...SRC], a, b, 0.3 + (i % 5) * 0.1, {});
      await device.queue.onSubmittedWorkDone();
    }
    await new Promise((r) => setTimeout(r, 100));
    out.destroy();
    return times.length ? median(times) : null;
  }

  const loadCell = (ms, fps) => {
    const pct = (ms * fps) / 10;
    const cls = pct <= 50 ? 'good' : pct <= 70 ? 'warn' : 'bad';
    return `<td class="${cls}">${pct.toFixed(0)}%</td>`;
  };

  const KIND = {
    conv1: '3×3 畳み込み（残差ブロック）',
    conv2: '3×3 畳み込み（ストライド 2）',
    deconv: '転置畳み込み',
    other: '入力の準備・ワープ・合成',
    bm: 'ブロックマッチング・選択マップ',
  };
  const blockName = (s, net) =>
    s.block === -2 ? 'ブロックマッチング・選択マップ' : s.block < 0 ? '特徴抽出（Head）' : `RIFE 段 ${s.block + 1}（1/${net.spec.scales[s.block]} 解像度）`;

  async function run() {
    $('run').disabled = true;
    $('log').textContent = '';
    const tbody = $('result').querySelector('tbody');
    const obody = $('options').querySelector('tbody');
    tbody.innerHTML = '';
    obody.innerHTML = '';
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
        const j = makeJob(device, net, { ...P, hybrid: true, multiscale: true });
        const { ms, times } = await timeJob(device, j.steps);
        const rms = await timeRender(device, j);
        const r = j.eng.res;
        tbody.insertAdjacentHTML(
          'beforeend',
          `<tr><td>${NAMES[preset]}</td><td>${MODEL_NAMES[P.model]} ${r.mw}×${r.mh}</td><td>${ms.toFixed(1)} ms</td>` +
            `${loadCell(ms, 12)}${loadCell(ms, 24)}${loadCell(ms, 30)}${loadCell(ms, 60)}<td>${rms == null ? '-' : rms.toFixed(2) + ' ms'}</td></tr>`
        );
        const cost = j.steps.reduce((a, s) => a + s.cost, 0);
        log(`${preset} ${r.mw}x${r.mh}: job median ${ms.toFixed(2)} ms (${times.map((t) => t.toFixed(1)).join(', ')}), ${j.steps.length} dispatches, ~${(cost / 1e9).toFixed(1)} G MAC; render ${rms == null ? '-' : rms.toFixed(3)} ms`);
        if (preset === 'standard') {
          // optional parts at this preset
          const variants = [
            ['半分の解像度での推論なし', { hybrid: true, multiscale: false }],
            ['ハイブリッドなし（RIFE のみ）', { hybrid: false }],
            ['＋ 大きな動きを高精度に（推論 3 回）', { hybrid: true, multiscale: true, anchors: true }],
          ];
          obody.insertAdjacentHTML('beforeend', `<tr><td>標準（既定の設定）</td><td>${ms.toFixed(1)} ms</td>${loadCell(ms, 24)}${loadCell(ms, 30)}</tr>`);
          for (const [label, opt] of variants) {
            const v = makeJob(device, net, { ...P, ...opt });
            const t = await timeJob(device, v.steps);
            obody.insertAdjacentHTML('beforeend', `<tr><td>${label}</td><td>${t.ms.toFixed(1)} ms</td>${loadCell(t.ms, 24)}${loadCell(t.ms, 30)}</tr>`);
            v.eng.destroy();
          }
          $('options').hidden = false;
          $('optTitle').hidden = false;
          if (device.features.has('timestamp-query')) {
            const per = await timeGroups(device, j.steps.map((s) => [s]));
            const total = per.reduce((a, b) => a + b, 0);
            const rows = (label) => {
              const sum = new Map();
              j.steps.forEach((s, i) => sum.set(label(s), (sum.get(label(s)) || 0) + per[i]));
              return [...sum].map(([k, v]) => `<tr><td>${k}</td><td>${v.toFixed(2)} ms</td><td>${((100 * v) / total).toFixed(0)}%</td></tr>`).join('');
            };
            const bd = $('breakdown').querySelector('tbody');
            bd.innerHTML = rows((s) => KIND[s.kind]) + '<tr><td colspan="3"></td></tr>' + rows((s) => blockName(s, net));
            log(`standard: sum of single-step passes ${total.toFixed(2)} ms`);
            $('breakdown').hidden = false;
            $('bdTitle').hidden = false;
          }
        }
        j.eng.destroy();
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
