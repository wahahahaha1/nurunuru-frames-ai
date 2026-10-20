// Frame timing: keeps the most recent source frames with their media timestamps, estimates the
// source frame duration and a smoothed media clock, and decides which pair (a, b) and phase t to
// show at a given display time. Output runs `delay` media-seconds behind the video so that the
// next source frame is always available.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});
  const KEEP = 5; // must stay below the interpolator's ring size (6)

  class FrameClock {
    constructor() {
      this.reset();
      this.frameDur = 1 / 30;
      this.vsync = 1 / 60;
      this.margin = 0;
      this.extra = 0; // inference pipeline latency (seconds), set by the player
      this.underruns = 0;
      this.drops = 0;
    }

    // Drop frames and clock sync (seek, stall, tab switch). The frame-rate estimate survives.
    reset() {
      this.frames = [];
      this.offset = null; // media = offset + rate * wallSeconds
      this.latency = null; // expectedDisplayTime - callback time (ms)
      this.lastPresented = null;
      this.count = 0; // frames since reset
      if (!this.diffs) this.diffs = [];
    }

    // New source: forget the frame-rate estimate too.
    resetAll() {
      this.diffs = [];
      this.reset();
    }

    // Frame duration as the mean of the middle half of recent timestamp differences
    // (robust to gaps, and unbiased for 1 ms container timestamps that alternate 16/17 ms).
    updateFrameDur(dt) {
      if (!(dt > 0.002 && dt < 0.25)) return;
      this.diffs.push(dt);
      if (this.diffs.length > 24) this.diffs.shift();
      const s = [...this.diffs].sort((x, y) => x - y);
      const lo = s.length >> 2;
      const mid = s.slice(lo, Math.max(lo + 1, s.length - lo));
      this.frameDur = mid.reduce((a, b) => a + b, 0) / mid.length;
    }

    // Returns 'dup' for a repeated timestamp, 'jump' when the timeline is discontinuous
    // (seek, loop, stall recovery: caller should flush), otherwise 'ok'.
    classify(mediaTime) {
      const last = this.frames[this.frames.length - 1];
      if (!last) return 'ok';
      const dt = mediaTime - last.m;
      if (Math.abs(dt) < 1e-6) return 'dup';
      if (dt < 0 || dt > Math.max(0.3, 4 * this.frameDur)) return 'jump';
      return 'ok';
    }

    observe(mediaTime, md, now) {
      const last = this.frames[this.frames.length - 1];
      if (last) this.updateFrameDur(mediaTime - last.m);
      if (this.lastPresented != null && md.presentedFrames - this.lastPresented > 1) {
        this.drops += md.presentedFrames - this.lastPresented - 1;
      }
      this.lastPresented = md.presentedFrames;
      const lat = md.expectedDisplayTime - now;
      if (lat >= 0 && lat < 100) this.latency = this.latency == null ? lat : this.latency + 0.1 * (lat - this.latency);
    }

    push(rec) {
      this.count++;
      this.frames.push(rec);
      while (this.frames.length > KEEP) this.frames.shift();
    }

    // Feed the media clock with (display time, media time) of a presented frame.
    sync(mediaTime, displayMs, rate) {
      const o = mediaTime - (rate * displayMs) / 1000;
      if (this.offset == null || Math.abs(o - this.offset) > 0.1 || rate !== this.rate) {
        this.offset = o;
      } else {
        this.offset += 0.08 * (o - this.offset);
      }
      this.rate = rate;
    }

    // Output delay in media seconds: one source frame + one display frame of jitter allowance
    // + the inference pipeline latency + whatever extra the underrun detector has added.
    get delay() {
      return this.frameDur + this.vsync + 0.001 + this.extra + this.margin;
    }

    // Pick what to show at display time displayMs. Returns { a, b, t } or null.
    pick(displayMs) {
      const fr = this.frames;
      if (!fr.length) return null;
      const last = fr[fr.length - 1];
      if (this.offset == null || fr.length < 2) return { a: last, b: last, t: 0, hold: true };
      const target = this.offset + (this.rate * displayMs) / 1000 - this.delay;
      if (target >= last.m) {
        return { a: last, b: last, t: 0, hold: true, late: target - last.m, target };
      }
      for (let i = fr.length - 2; i >= 0; i--) {
        if (fr[i].m <= target) {
          const a = fr[i];
          const b = fr[i + 1];
          return { a, b, t: Math.min(1, Math.max(0, (target - a.m) / (b.m - a.m))), target };
        }
      }
      return { a: fr[0], b: fr[0], t: 0, hold: true, target };
    }
  }

  S.FrameClock = FrameClock;
})();
