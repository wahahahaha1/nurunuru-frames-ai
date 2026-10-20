// Content-script entry: finds the site's main <video>, and drives capture (requestVideoFrameCallback),
// RIFE inference and per-display-frame rendering (requestAnimationFrame) on a canvas overlay.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});
  if (S.mainLoaded) return;
  S.mainLoaded = true;

  const audioByVideo = new WeakMap(); // a media element can be routed through WebAudio only once
  const PRESET_ORDER = ['fast', 'standard', 'quality'];
  const PRESET_NAMES = { fast: '速度優先', standard: '標準', quality: '高画質' };
  const LOAD_LIMIT = 0.7; // share of GPU time interpolation may use before the guard steps in
  const TAG = '[NuruNuru Frames AI]';

  class Player {
    constructor(video, settings, site) {
      this.video = video;
      this.site = site;
      this.settings = settings;
      this.clock = new S.FrameClock();
      this.overlay = new S.Overlay(video);
      this.osd = new S.OSD(this.overlay);
      if (!audioByVideo.has(video)) audioByVideo.set(video, new S.AudioDelay(video));
      this.audio = audioByVideo.get(video);
      this.active = true;
      this.ready = false;
      this.failed = null;
      this.reason = '初期化中';
      this.errors = 0;
      this.rafDiffs = [];
      this.displayHz = 60;
      this.dirty = true;
      this.renderedSize = '';
      this.stats = { frames: 0, interp: 0, since: performance.now(), outFps: 0, interpPct: 0 };
      // performance guard: measured GPU ms per source frame (me = inference) / output frame (render)
      this.perf = { me: null, render: null, over: 0, starve: 0, presetCap: null, outDiv: 1, note: '', stopped: false };
      this.rafCount = 0;
      this.onFrame = this.onFrame.bind(this);
      this.onRaf = this.onRaf.bind(this);
      this.handlers = {
        seeking: () => this.flush(),
        emptied: () => this.flush(true),
        loadstart: () => this.flush(true),
        ratechange: () => (this.clock.offset = null),
        play: () => {
          this.clock.offset = null;
          this.dirty = true;
        },
        pause: () => (this.dirty = true),
      };
      for (const [ev, fn] of Object.entries(this.handlers)) video.addEventListener(ev, fn);
      this.onVis = () => {
        if (!document.hidden) this.flush();
      };
      document.addEventListener('visibilitychange', this.onVis);
      this.init();
    }

    async init() {
      try {
        const device = await S.getDevice();
        if (!this.active) return;
        this.reason = 'AI モデルを読み込み中';
        const params = S.engineParams(this.settings);
        const net = await S.getRifeNet(device, params.model);
        if (!this.active) return;
        this.device = device;
        device.onuncapturederror = (e) => S.player && S.player.noteError(e.error.message);
        this.engine = new S.Interpolator(device, net, params);
        this.engine.onTiming = (kind, ms) => this.onTiming(kind, ms);
        this.ctx = this.overlay.canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.ctx.configure({ device, format: this.format, alphaMode: 'premultiplied' });
        this.ready = true;
        this.applySettings(this.settings);
        this.video.requestVideoFrameCallback(this.onFrame);
        this.rafId = requestAnimationFrame(this.onRaf);
      } catch (e) {
        this.fail(e.message);
      }
    }

    applySettings(s) {
      if (s.preset !== this.settings.preset) {
        // the user picked a preset: forget automatic downgrades and try again
        if (this.perf.stopped) this.failed = null;
        this.perf = { me: null, render: null, over: 0, starve: 0, presetCap: null, outDiv: 1, note: '', stopped: false };
      }
      this.settings = s;
      this.osd.show(!!s.osd);
      this.applyEngineParams();
      if (!s.audioSync) this.audio.set(0);
      this.dirty = true;
    }

    applyEngineParams() {
      if (!this.engine) return;
      const old = this.engine.res;
      const params = S.engineParams({ ...this.settings, preset: this.effectivePreset() });
      this.engine.setParams(params);
      if (this.engine.res !== old) this.flush();
      if (params.model !== this.engine.net.model) {
        // another model: load it (once per page), then switch
        S.getRifeNet(this.device, params.model).then(
          (net) => {
            if (!this.active || !this.engine || S.engineParams({ ...this.settings, preset: this.effectivePreset() }).model !== net.model) return;
            this.engine.setNet(net);
            this.flush();
          },
          (e) => this.fail(e.message)
        );
      }
    }

    effectivePreset() {
      const want = this.settings.preset;
      const cap = this.perf.presetCap;
      return cap && PRESET_ORDER.indexOf(cap) < PRESET_ORDER.indexOf(want) ? cap : want;
    }

    onTiming(kind, ms) {
      const p = this.perf;
      p[kind] = p[kind] == null ? ms : p[kind] + 0.2 * (ms - p[kind]);
    }

    srcRate() {
      return this.video.playbackRate / this.clock.frameDur;
    }

    // Requested output frame rate (before the performance guard), capped at the display rate.
    targetRate() {
      const s = this.settings;
      let r = this.displayHz;
      if (s.target === 'fixed') r = Math.max(1, +s.targetFps || 60);
      else if (s.target && s.target[0] === 'x') r = this.srcRate() * parseFloat(s.target.slice(1));
      return Math.min(r, this.displayHz);
    }

    outRate() {
      return this.targetRate() / this.perf.outDiv;
    }

    // Share of GPU time used by interpolation: motion per source frame + render per output frame.
    gpuLoad() {
      const p = this.perf;
      if (p.me == null) return null;
      return (p.me * this.srcRate() + (p.render || 0) * this.outRate()) / 1000;
    }

    // Called once per second while interpolating. After 3 s over the limit: lower the preset,
    // then halve the output frame rate, then give up (show the original video). No finished
    // inference for 5 s (a GPU too slow to even report its timing) counts as over the limit.
    checkPerf() {
      const load = this.gpuLoad();
      const p = this.perf;
      p.starve = this.stats.interpPct < 1 ? p.starve + 1 : 0;
      const starving = p.starve >= 5;
      if (load == null && !starving) return;
      p.over = starving || load > LOAD_LIMIT ? p.over + 1 : 0;
      if (p.over < 3) return;
      p.over = 0;
      p.starve = 0;
      const cur = this.effectivePreset();
      const srcFps = this.srcRate();
      if (cur !== 'fast') {
        p.presetCap = PRESET_ORDER[PRESET_ORDER.indexOf(cur) - 1];
        p.me = null;
        p.render = null;
        p.note = `GPU 負荷が高いため「${PRESET_NAMES[p.presetCap]}」に自動で下げました`;
        this.applyEngineParams();
      } else if (!starving && p.outDiv === 1 && this.targetRate() / 2 >= 1.5 * srcFps) {
        p.outDiv = 2;
        p.render = null;
        p.note = 'GPU 負荷が高いため出力を半分のフレームレートにしました';
      } else {
        p.stopped = true;
        p.note = '';
        this.fail('GPU の性能が足りないため補間を停止しました（プリセットを選び直すと再試行します）');
      }
    }

    fail(msg) {
      this.failed = msg;
      this.overlay.show(false);
      console.warn(TAG, msg);
    }

    noteError(msg) {
      this.errors++;
      this.lastError = msg;
      console.warn(TAG, msg);
      if (this.errors > 30) this.fail('GPU エラーが多発したため停止: ' + msg);
    }

    flush(newSource) {
      if (newSource) this.clock.resetAll();
      else this.clock.reset();
      if (this.engine) this.engine.cancelJobs();
      this.dirty = true;
      this.overlay.show(false);
    }

    bypassReason() {
      if (this.failed) return this.failed;
      const v = this.video;
      if (v.mediaKeys) return 'DRM 保護された動画のため補間できません';
      const c = this.clock;
      if (c.diffs.length >= 6) {
        const src = this.srcRate();
        const target = this.targetRate();
        if (src >= target * 0.9) {
          return target < this.displayHz * 0.97
            ? `ソース ${src.toFixed(1)} fps ≥ 目標 ${target.toFixed(0)} fps のため補間不要`
            : `ソース ${src.toFixed(1)} fps ≥ 表示 ${this.displayHz.toFixed(0)} Hz のため補間不要`;
        }
      }
      return null;
    }

    onFrame(now, md) {
      if (!this.active) return;
      this.video.requestVideoFrameCallback(this.onFrame);
      const m = md.mediaTime;
      const kind = this.clock.classify(m);
      if (kind === 'dup') return;
      if (kind === 'jump') this.flush();
      if (this.capture) this.capture.onFrame(m);
      this.clock.observe(m, md, now);
      const reason = this.bypassReason();
      let f = null;
      if (!reason) {
        const v = this.video;
        try {
          if (this.engine.configure(v.videoWidth, v.videoHeight)) this.clock.frames = [];
          const last = this.clock.frames[this.clock.frames.length - 1];
          const prev = last && last.f && this.engine.isLive(last.f) ? last.f : null;
          f = this.engine.ingest(v, prev, this.settings.anime !== false);
        } catch (e) {
          if (e.name === 'SecurityError') this.fail('クロスオリジンの動画のため読み取れません');
          else this.noteError(e.message);
        }
      }
      this.clock.push({ m, f });
      if (!this.video.paused) this.clock.sync(m, md.expectedDisplayTime, this.video.playbackRate);
      this.dirty = true;
    }

    measureDisplay(ts) {
      if (this.lastRaf != null) {
        const d = ts - this.lastRaf;
        if (d > 2 && d < 100) {
          this.rafDiffs.push(d);
          if (this.rafDiffs.length > 90) this.rafDiffs.shift();
          if (this.rafDiffs.length >= 20 && (this.rafDiffs.length % 10 === 0 || this.rafDiffs.length === 20)) {
            const s = [...this.rafDiffs].sort((a, b) => a - b);
            const med = s[s.length >> 1];
            this.displayHz = 1000 / med;
            this.clock.vsync = med / 1000;
          }
        }
      }
      this.lastRaf = ts;
    }

    onRaf(ts) {
      if (!this.active) return;
      this.rafId = requestAnimationFrame(this.onRaf);
      this.measureDisplay(ts);
      this.overlay.poll(ts);
      if (!this.ready) return;
      this.draw(ts);
      // queued inference goes to the GPU after this display frame's render
      if (this.engine && !this.failed) {
        const c = this.clock;
        try {
          this.engine.pump(c.vsync * 1000, (c.frameDur * 1000) / Math.max(0.1, this.video.playbackRate));
        } catch (e) {
          this.noteError(e.message);
        }
        c.extra = this.engine.chunks * c.vsync;
      }
    }

    draw(ts) {
      const s = this.settings;
      const v = this.video;
      const reason = this.bypassReason();
      if (reason !== this.reason) {
        this.reason = reason;
        if (reason) this.flush();
      }
      if (reason) {
        this.audio.set(0);
        this.updateOSD(ts);
        return;
      }
      const size = this.overlay.canvas.width + 'x' + this.overlay.canvas.height;
      if (size !== this.renderedSize) this.dirty = true;
      const playing = !v.paused && !v.seeking && v.readyState >= 2;
      const c = this.clock;
      let sel;
      if (playing) {
        // Output below the display rate (target setting / performance guard): skipped display frames
        // keep showing the canvas's last image.
        let when = ts + (c.latency || 0);
        const rate = this.outRate();
        if (rate < this.displayHz * 0.97) {
          const n = this.displayHz / rate;
          const nr = Math.round(n);
          if (Math.abs(n - nr) < 0.04 * nr) {
            // the display rate is a multiple of the target: draw every nr-th display frame (even pacing)
            if (this.rafCount++ % nr !== 0) return this.updateOSD(ts);
          } else {
            // arbitrary rate: show the frame for the latest instant of an evenly spaced output grid
            const k = Math.floor((when * rate) / 1000);
            if (k === this.lastGrid) return this.updateOSD(ts);
            this.lastGrid = k;
            when = (k * 1000) / rate;
          }
        }
        sel = c.pick(when);
      } else {
        if (!this.dirty) return this.updateOSD(ts);
        const last = c.frames[c.frames.length - 1];
        sel = last && { a: last, b: last, t: 0 };
      }
      if (!sel || !sel.a.f || !this.engine.isLive(sel.a.f)) {
        this.overlay.show(false);
        return this.updateOSD(ts);
      }
      // late = the next source frame has not arrived in time; widen the delay (ignored while the clock settles)
      if (playing && sel.late > 0 && v.readyState >= 3 && c.count >= 12) {
        c.underruns++;
        c.margin = Math.min(c.margin + 0.002, Math.min(0.04, 2 * c.frameDur));
      }
      const b = sel.b.f && this.engine.isLive(sel.b.f) ? sel.b.f : sel.a.f;
      const rect = this.overlay.contentRect(v.videoWidth, v.videoHeight);
      let paired;
      try {
        paired = this.engine.render(this.ctx.getCurrentTexture().createView(), this.format, rect, sel.a.f, b, sel.t, {
          mode: s.showVectors ? 4 : 0,
          split: s.demo ? Math.round(this.overlay.canvas.width / 2) : 0,
          masking: s.masking / 100,
        });
      } catch (e) {
        this.noteError(e.message);
        return;
      }
      // the inference for this pair has not been submitted in time: widen the delay as well
      if (playing && !paired && b !== sel.a.f && c.count >= 12 && v.readyState >= 3) {
        c.underruns++;
        c.margin = Math.min(c.margin + 0.002, Math.min(0.04, 2 * c.frameDur));
      }
      if (S.trace) S.trace.push([ts, sel.target, sel.t, sel.a.m, sel.hold ? 1 : 0]);
      this.renderedSize = size;
      this.overlay.show(true);
      this.dirty = false;
      this.stats.frames++;
      if (paired && sel.t > 0) this.stats.interp++;
      if (s.audioSync) {
        this.audio.enable();
        this.audio.set(c.delay / Math.max(0.1, v.playbackRate) + s.audioOffsetMs / 1000);
      }
      this.updateOSD(ts);
    }

    updateOSD(ts) {
      const st = this.stats;
      if (ts - st.since >= 1000) {
        st.outFps = (st.frames * 1000) / (ts - st.since);
        st.interpPct = st.frames ? (100 * st.interp) / st.frames : 0;
        st.frames = 0;
        st.interp = 0;
        st.since = ts;
        if (!this.reason && !this.video.paused && this.ready) this.checkPerf();
        if (this.osd.on) this.osd.update(this.statusText());
      }
    }

    status() {
      const v = this.video;
      const c = this.clock;
      const r = this.engine && this.engine.res;
      return {
        site: this.site.name,
        state: this.reason ? 'bypass' : this.ready ? 'active' : 'init',
        reason: this.reason,
        src: v.videoWidth + 'x' + v.videoHeight,
        srcFps: c.diffs.length ? 1 / c.frameDur : 0,
        displayHz: this.displayHz,
        outFps: this.stats.outFps,
        interpPct: this.stats.interpPct,
        me: r ? `${r.mw}x${r.mh}` : '-',
        chunks: this.engine ? this.engine.chunks : 0,
        delayMs: c.delay * 1000,
        audio: this.settings.audioSync ? this.audio.state : 'off',
        // audio is delayed by the same video delay (+ user offset) once the WebAudio route is active
        audioDelayMs:
          this.settings.audioSync && this.audio.active && !this.reason
            ? (c.delay / Math.max(0.1, v.playbackRate)) * 1000 + this.settings.audioOffsetMs
            : 0,
        underruns: c.underruns,
        drops: c.drops,
        errors: this.errors,
        lastError: this.lastError || '',
        preset: this.effectivePreset(),
        targetFps: this.outRate(),
        gpuLoad: this.gpuLoad(),
        meMs: this.perf.me,
        renderMs: this.perf.render,
        perfNote: this.perf.note,
        capture: this.capture ? this.capture.state : '',
      };
    }

    statusText() {
      const st = this.status();
      if (st.reason) return `NuruNuru Frames AI: ${st.reason}`;
      const gpu = st.gpuLoad == null ? '-' : `${(st.gpuLoad * 100).toFixed(0)}%`;
      const audio = { off: 'オフ', idle: '待機', waiting: 'クリック待ち', failed: '使用不可' };
      const audioText = st.audio === 'active' ? `+${st.audioDelayMs.toFixed(0)}ms` : audio[st.audio] || st.audio;
      return (
        `NuruNuru Frames AI [${PRESET_NAMES[st.preset]}]\n` +
        `${st.src} ${st.srcFps.toFixed(2)}fps → ${st.outFps.toFixed(0)}fps (目標 ${st.targetFps.toFixed(0)} / 補間 ${st.interpPct.toFixed(0)}%)\n` +
        `RIFE ${st.me} (${st.chunks} 分割) | 映像遅延 ${st.delayMs.toFixed(0)}ms | 音ズレ補正 ${audioText} | GPU ${gpu}\n` +
        `underrun ${st.underruns} | drop ${st.drops} | err ${st.errors}` +
        (st.perfNote ? `\n${st.perfNote}` : '')
      );
    }

    // Debug capture (popup button): source frames + interpolated output of the next few seconds.
    startCapture(seconds) {
      if (this.capture && this.capture.state === 'recording') return;
      const st = this.status();
      this.capture = new S.Capture(this.video, this.overlay.canvas, seconds, {
        page: location.href,
        src: st.src,
        srcFps: st.srcFps,
        displayHz: st.displayHz,
        settings: this.settings,
        preset: st.preset,
        model: this.engine && this.engine.net.model,
        modelSize: st.me,
        userAgent: navigator.userAgent,
      });
    }

    destroy() {
      this.active = false;
      cancelAnimationFrame(this.rafId);
      if (this.capture) this.capture.cancel();
      for (const [ev, fn] of Object.entries(this.handlers)) this.video.removeEventListener(ev, fn);
      document.removeEventListener('visibilitychange', this.onVis);
      this.audio.set(0);
      this.osd.destroy();
      this.overlay.destroy();
      if (this.ctx) this.ctx.unconfigure();
      if (this.engine) this.engine.destroy();
    }
  }

  S.Player = Player;

  // ---- bootstrap -------------------------------------------------------------------------------
  const site = S.detectSite();
  if (!site || !globalThis.chrome || !chrome.runtime || !chrome.runtime.id) return;
  let settings = S.mergeSettings({});
  let player = null;

  const wanted = () => settings.enabled && settings.sites[site.id] !== false;

  // The regular NuruNuru Frames (block matching) puts a canvas.nnf-canvas next to the same video:
  // when it is running, stay out of its way.
  const BLOCKED_REASON = '通常版の NuruNuru Frames が動作中のため停止しています（どちらか一方をオフにしてください）';
  let blocked = false;

  function poll() {
    let v = wanted() ? site.find() : null;
    blocked = !!(v && v.parentNode && v.parentNode.querySelector(':scope > canvas.nnf-canvas'));
    if (blocked) v = null;
    if (player && (player.video !== v || !v.isConnected)) {
      player.destroy();
      player = null;
    }
    if (!player && v && v.isConnected) player = new Player(v, settings, site);
    S.player = player;
  }

  S.onDeviceLost = (info) => {
    console.warn(TAG, 'GPU device lost:', info && info.message);
    if (player) {
      player.destroy();
      player = null;
    }
    setTimeout(poll, 1500);
  };

  S.loadSettings().then((s) => {
    settings = s;
    S.onSettingsChanged((ns) => {
      settings = ns;
      if (player) player.applySettings(ns);
      poll();
    });
    poll();
    setInterval(poll, 1000);
    window.addEventListener('yt-navigate-finish', poll);
  });

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg && msg.type === 'nnf-capture' && player) {
      player.startCapture(msg.seconds || 3);
      sendResponse(player.status());
    } else if (msg && msg.type === 'nnf-status' && player) {
      sendResponse(player.status());
    } else if (msg && msg.type === 'nnf-status' && blocked) {
      sendResponse({ site: site.name, state: 'bypass', reason: BLOCKED_REASON });
    }
    return false;
  });
})();
