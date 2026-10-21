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

  // Kernel variants of the 3x3 convolution, timed on the heaviest distinct conv layers of the job
  // (real weights, no split-K): which tiling suits this GPU.
  const CANDIDATES = {
    1: [
      { tx: 16, ty: 8, px: 2, py: 2, ci: 4, co: 16 },
      { tx: 16, ty: 8, px: 2, py: 2, ci: 4, co: 32 },
      { tx: 16, ty: 8, px: 2, py: 1, ci: 4, co: 32 },
      { tx: 8, ty: 8, px: 2, py: 2, ci: 4, co: 32 },
      { tx: 16, ty: 16, px: 2, py: 1, ci: 4, co: 16 },
      { tx: 16, ty: 16, px: 2, py: 1, ci: 4, co: 32 },
      { tx: 16, ty: 8, px: 2, py: 2, ci: 2, co: 32 },
      { tx: 8, ty: 8, px: 4, py: 2, ci: 4, co: 16 },
    ],
    2: [
      { tx: 8, ty: 8, px: 2, py: 2, ci: 2, co: 16 },
      { tx: 8, ty: 8, px: 2, py: 2, ci: 2, co: 32 },
      { tx: 8, ty: 8, px: 2, py: 2, ci: 4, co: 16 },
      { tx: 16, ty: 8, px: 2, py: 2, ci: 2, co: 16 },
      { tx: 16, ty: 8, px: 2, py: 1, ci: 2, co: 32 },
      { tx: 8, ty: 8, px: 1, py: 1, ci: 8, co: 32 },
    ],
  };
  const cfgName = (c) => `${c.tx}×${c.ty} 画素${c.px}×${c.py} 入力${c.ci} 出力${c.co}`;
  const smem = (c, stride) => 4 * (c.ci * ((c.tx * c.px - 1) * stride + 3) * ((c.ty * c.py - 1) * stride + 3) + (c.co / 16) * c.ci * 36 * 4);

  async function sweep(device, net, steps, per) {
    if (!device.features.has('timestamp-query')) return;
    const seen = new Set();
    const layers = steps
      .map((s, i) => ({ s, ms: per[i] }))
      .filter(({ s }) => s.layer)
      .sort((x, y) => y.ms - x.ms)
      .filter(({ s }) => {
        const k = `${s.layer.L.cin}/${s.layer.L.cout}/${s.layer.stride}/${s.layer.ow}/${s.layer.oh}`;
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      })
      .slice(0, 4);
    const tbody = $('sweep').querySelector('tbody');
    tbody.innerHTML = '';
    $('sweep').hidden = false;
    $('sweepTitle').hidden = false;
    const BU = GPUBufferUsage;
    const pipes = new Map();
    for (const { s } of layers) {
      const { L, stride, iw, ih, ow, oh } = s.layer;
      const inp = device.createBuffer({ size: L.cin * iw * ih * 4, usage: BU.STORAGE });
      const out = device.createBuffer({ size: L.cout * ow * oh * 4, usage: BU.STORAGE });
      const res = device.createBuffer({ size: L.cout * ow * oh * 4, usage: BU.STORAGE });
      for (const c of CANDIDATES[stride]) {
        if (smem(c, stride) > device.limits.maxComputeWorkgroupStorageSize) continue;
        const key = JSON.stringify({ ...c, stride });
        if (!pipes.has(key)) {
          pipes.set(key, device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: S.WGSL.conv({ ...c, stride }) }), entryPoint: 'main' } }));
        }
        const pipe = pipes.get(key);
        const ncog = Math.ceil(L.cout / c.co);
        const u = device.createBuffer({ size: 64, usage: BU.UNIFORM | BU.COPY_DST });
        device.queue.writeBuffer(u, 0, new Uint32Array([L.cin, L.cout, iw, ih, ow, oh, L.wOff, L.bOff, stride === 1 ? 1 : 0, 1, ncog, L.cin, 0, 0, 0, 0]));
        const bg = device.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries: [u, inp, net.wbuf, res, out].map((b, i) => ({ binding: i, resource: { buffer: b } })),
        });
        const wg = [Math.ceil(ow / (c.tx * c.px)), Math.ceil(oh / (c.ty * c.py)), ncog];
        const step = { pipe, bg, wg };
        await timeGroups(device, [[step]]);
        const times = [];
        for (let r = 0; r < 5; r++) times.push((await timeGroups(device, [[step, step, step, step]]))[0] / 4);
        const ms = median(times);
        const flops = (2 * L.cin * L.cout * 9 * ow * oh) / ms / 1e9;
        tbody.insertAdjacentHTML('beforeend', `<tr><td>${s.key} ${L.cin}→${L.cout} ${ow}×${oh}${stride === 2 ? ' 間引き' : ''}</td><td>${cfgName(c)}</td><td>${ms.toFixed(3)} ms</td><td>${flops.toFixed(2)} TFLOPS</td></tr>`);
        log(`sweep ${s.key} ${L.cin}-${L.cout} ${ow}x${oh} s${stride} ${c.tx}x${c.ty} p${c.px}x${c.py} ci${c.ci} co${c.co}: ${ms.toFixed(3)} ms`);
        u.destroy();
      }
      [inp, out, res].forEach((b) => b.destroy());
    }
  }

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
      {
        // warm-up: let the GPU reach its clocks before anything is measured (else the first numbers
        // are several times too high)
        const net = await S.getRifeNet(device, S.PRESETS.standard.model);
        const w = makeJob(device, net, { ...S.PRESETS.standard, hybrid: true, multiscale: true, ensemble: true });
        for (const t0 = performance.now(); performance.now() - t0 < 2000; ) await timeGroups(device, [w.steps]);
        w.eng.destroy();
      }
      for (const preset of ['fast', 'standard', 'quality']) {
        const P = S.PRESETS[preset];
        const net = await S.getRifeNet(device, P.model);
        const j = makeJob(device, net, { ...P, hybrid: true, multiscale: true, ensemble: true });
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
            ['（比較）推論をまとめない', { hybrid: true, multiscale: true, ensemble: true, batchEns: false }],
            ['＋ 左右反転でも推論（推論 4 回）', { hybrid: true, multiscale: true, ensemble: 4 }],
            ['（比較）左右反転・推論をまとめない', { hybrid: true, multiscale: true, ensemble: 4, batchEns: false }],
            ['入れ替えての推論なし', { hybrid: true, multiscale: true, ensemble: false }],
            ['半分の解像度での推論なし', { hybrid: true, multiscale: false, ensemble: true }],
            ['ハイブリッドなし（RIFE のみ）', { hybrid: false }],
            ['＋ 大きな動きを高精度に（推論 3 回）', { hybrid: true, multiscale: true, ensemble: true, anchors: true }],
          ];
          obody.insertAdjacentHTML('beforeend', `<tr><td>標準（既定の設定）</td><td>${ms.toFixed(1)} ms</td>${loadCell(ms, 24)}${loadCell(ms, 30)}</tr>`);
          for (const [label, opt] of variants) {
            const v = makeJob(device, net, { ...P, ...opt });
            const t = await timeJob(device, v.steps);
            obody.insertAdjacentHTML('beforeend', `<tr><td>${label}</td><td>${t.ms.toFixed(1)} ms</td>${loadCell(t.ms, 24)}${loadCell(t.ms, 30)}</tr>`);
            log(`${label}: ${t.ms.toFixed(2)} ms, ${v.steps.length} dispatches`);
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
            // the slowest steps, with the multiply-adds per second they achieve (where known)
            const hot = j.steps.map((s, i) => ({ s, ms: per[i] })).sort((x, y) => y.ms - x.ms).slice(0, 15);
            $('hot').querySelector('tbody').innerHTML = hot
              .map(({ s, ms }) => `<tr><td>${s.key}</td><td>${s.variant || KIND[s.kind]}</td><td>${ms.toFixed(2)} ms</td><td>${s.variant ? ((2 * s.cost) / ms / 1e9).toFixed(2) + ' TFLOPS' : ''}</td></tr>`)
              .join('');
            $('hot').hidden = false;
            $('hotTitle').hidden = false;
            log('hot: ' + hot.map(({ s, ms }) => `${s.key}=${ms.toFixed(2)}`).join(' '));
            await sweep(device, net, j.steps, per);
          }
        }
        j.eng.destroy();
        await new Promise((r) => setTimeout(r, 50));
      }
      // auto-tuning of the conv tilings (what the player does once per GPU / model / size)
      if (device.features.has('timestamp-query')) {
        const P = S.PRESETS.standard;
        const net = await S.getRifeNet(device, P.model);
        const opt = { ...P, hybrid: true, multiscale: true, ensemble: true };
        const a = makeJob(device, net, opt);
        const before = (await timeJob(device, a.steps)).ms;
        const r = a.eng.res;
        a.eng.destroy();
        const n1 = await net.tune(r.mw, r.mh);
        const n2 = await net.tune(r.half.cw, r.half.ch);
        net.destroyResources();
        const b = makeJob(device, net, opt);
        const after = (await timeJob(device, b.steps)).ms;
        b.eng.destroy();
        $('tune').textContent = `組み方を変えた層：${n1 + n2}　処理 1 回（標準・既定の設定）：${before.toFixed(1)} ms → ${after.toFixed(1)} ms`;
        $('tune').hidden = false;
        $('tuneTitle').hidden = false;
        log(`tune: ${n1}+${n2} layers changed, ${before.toFixed(2)} -> ${after.toFixed(2)} ms; ` + [...net.choice].map(([k, v]) => `${k}=${v}`).join(' '));
        net.choice.clear();
        net.destroyResources();
      }
      if (err) log('GPU エラー: ' + err);
      log(`${navigator.userAgent}`);
    } catch (e) {
      log('エラー: ' + e.message);
    }
    $('run').disabled = false;
    $('csv').hidden = false;
  }

  // All tables and the log as one CSV file (section, columns...), with a BOM so that Excel reads UTF-8.
  function saveCsv() {
    const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
    const lines = [[q('GPU'), q($('gpu').textContent)].join(',')];
    const sections = [
      ['result', 'プリセット'],
      ['options', 'オプション'],
      ['breakdown', '内訳'],
      ['hot', '時間のかかっている処理'],
      ['sweep', '組み方の比較'],
    ];
    for (const [id, name] of sections) {
      const t = $(id);
      if (!t || t.hidden) continue;
      for (const tr of t.querySelectorAll('tr')) {
        const cells = [...tr.children].map((c) => c.textContent.trim());
        if (cells.some((c) => c)) lines.push([q(name), ...cells.map(q)].join(','));
      }
    }
    if (!$('tune').hidden) lines.push([q('自動調整'), q($('tune').textContent)].join(','));
    for (const l of $('log').textContent.split('\n')) if (l.trim()) lines.push([q('記録'), q(l)].join(','));
    const blob = new Blob(['\ufeff' + lines.join('\r\n') + '\r\n'], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    a.download = `nnf-ai-bench-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }

  $('run').onclick = run;
  $('csv').onclick = saveCsv;
})();
