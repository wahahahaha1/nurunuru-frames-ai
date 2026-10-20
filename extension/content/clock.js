// Frame timing: keeps the most recent source frames with their media timestamps, estimates the
// source frame duration and a smoothed media clock, and decides which pair (a, b) and phase t to
// show at a given display time. Output runs `delay` media-seconds behind the video so that the
// next source frame is always available.
// Repeated drawings (anime on twos / threes): frames whose engine record says dupState 'dup' are
// skipped, so pairs span distinct drawings, and the delay grows to the longest recent span.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});
  const KEEP = 7; // must stay below the interpolator's ring size (8)
  const MAX_SPAN = 3.5; // longest span between drawings that is interpolated over, in source frames

  const stateOf = (rec) => (rec.f ? rec.f.dupState || 'distinct' : 'distinct');

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
      this.spans = []; // recent media-time spans between distinct drawings
      this.lastSpanEnd = -Infinity;
      this.offset = null; // media = offset + rate * wallSeconds
      this.latency = null; // expectedDisplayTime - callback time (ms)
      this.lastPresented = null;
      this.count = 0; // frames since reset
      if (!this.diffs) this.diffs = [];
    }

    // New source: forget the frame-rate estimate and the underrun margin too.
    resetAll() {
      this.diffs = [];
      this.margin = 0;
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

    // Longest recent span between distinct drawings (one source frame for ordinary video).
    get spanDur() {
      const cap = MAX_SPAN * this.frameDur;
      let m = this.frameDur;
      for (const s of this.spans) if (s <= cap) m = Math.max(m, s);
      return m;
    }

    // Output delay in media seconds: one drawing span + one display frame of jitter allowance
    // + the inference pipeline latency + whatever extra the underrun detector has added.
    get delay() {
      return this.spanDur + this.vsync + 0.001 + this.extra + this.margin;
    }

    // Distinct drawings among the kept frames (records whose duplicate test is still pending are left
    // out until it resolves; a drawing that started before the oldest kept frame is represented by
    // that frame); records new spans between them.
    distinct() {
      const out = this.frames.filter((r, i) => stateOf(r) === 'distinct' || (i === 0 && stateOf(r) === 'dup'));
      for (let i = 1; i < out.length; i++) {
        if (out[i].m > this.lastSpanEnd) {
          this.spans.push(out[i].m - out[i - 1].m);
          if (this.spans.length > 12) this.spans.shift();
          this.lastSpanEnd = out[i].m;
        }
      }
      return out;
    }

    // Pick what to show at display time displayMs. Returns { a, b, t } or null.
    pick(displayMs) {
      const all = this.frames;
      if (!all.length) return null;
      const fr = this.distinct();
      if (!fr.length) return null;
      const last = fr[fr.length - 1];
      if (this.offset == null || fr.length < 2) return { a: last, b: last, t: 0, hold: true };
      const target = this.offset + (this.rate * displayMs) / 1000 - this.delay;
      if (target >= last.m) {
        // late only when even the newest frame (repeat or not) is behind the target
        const newest = all[all.length - 1].m;
        return { a: last, b: last, t: 0, hold: true, late: target > newest ? target - newest : 0, target };
      }
      for (let i = fr.length - 2; i >= 0; i--) {
        if (fr[i].m <= target) {
          const a = fr[i];
          const b = fr[i + 1];
          // after a long still, hold the drawing and move only over the last usual span
          const start = Math.max(a.m, b.m - this.spanDur);
          return { a, b, t: Math.min(1, Math.max(0, (target - start) / (b.m - start))), target };
        }
      }
      return { a: fr[0], b: fr[0], t: 0, hold: true, target };
    }
  }

  S.FrameClock = FrameClock;
})();
