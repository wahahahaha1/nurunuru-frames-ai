// Delays the video's audio by the interpolation latency (WebAudio DelayNodes), so lips stay in sync.
// Routing a media element through WebAudio is permanent, so it is only done once the AudioContext
// is confirmed running (after a user gesture); otherwise the element is left untouched.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});
  // A gliding delay changes the playback speed while it glides, which is heard as a wavering pitch
  // in music. Instead, two delay lines are kept and a change crossfades from one to the other: no
  // pitch change, only a short overlap. Changes are also kept rare: the delay follows the largest
  // video delay of the last WINDOW_MS plus HEADROOM (audio slightly late is hard to notice, early is
  // not), and only moves when the video delay exceeds it or it has been STEP_DOWN too large.
  const FADE = 0.06;
  const WINDOW_MS = 10000;
  const HEADROOM = 0.02;
  const STEP_DOWN = 0.06;
  const fadeIn = new Float32Array(64).map((_, i) => Math.sin((i / 63) * Math.PI * 0.5));
  const fadeOut = fadeIn.map((_, i) => fadeIn[63 - i]);

  class AudioDelay {
    constructor(video) {
      this.video = video;
      this.ctx = null;
      this.lines = null; // two { delay, gain }, lines[0] audible
      this.state = 'idle'; // idle | waiting | active | failed
      this.target = 0; // delay being applied (or about to be, once the route is active)
      this.fadeEnd = 0;
      this.pending = null; // a change waiting for the running crossfade to finish
      this.peaks = []; // [second, largest wanted delay in that second]
      this.onGesture = () => this.tryConnect();
    }

    enable() {
      if (this.state === 'active' || this.state === 'failed') return;
      if (navigator.userActivation && navigator.userActivation.hasBeenActive) {
        this.tryConnect();
      } else if (this.state !== 'waiting') {
        this.state = 'waiting';
        window.addEventListener('pointerdown', this.onGesture, true);
        window.addEventListener('keydown', this.onGesture, true);
      }
    }

    async tryConnect() {
      if (this.state === 'active' || this.state === 'failed' || this.connecting) return;
      this.connecting = true;
      try {
        // one AudioContext per page, shared by every video element we route
        if (!S.audioCtx) S.audioCtx = new AudioContext({ latencyHint: 'interactive' });
        this.ctx = S.audioCtx;
        if (this.ctx.state !== 'running') await this.ctx.resume();
        if (this.ctx.state !== 'running') return; // still blocked: wait for the next gesture
        const src = this.ctx.createMediaElementSource(this.video);
        this.lines = [0, 1].map((k) => {
          const delay = this.ctx.createDelay(1.0);
          const gain = this.ctx.createGain();
          delay.delayTime.value = this.target;
          gain.gain.value = k === 0 ? 1 : 0;
          src.connect(delay).connect(gain).connect(this.ctx.destination);
          return { delay, gain };
        });
        const ctx = this.ctx;
        ctx.onstatechange = () => {
          if (ctx.state === 'suspended') ctx.resume().catch(() => {});
        };
        this.state = 'active';
        window.removeEventListener('pointerdown', this.onGesture, true);
        window.removeEventListener('keydown', this.onGesture, true);
      } catch (e) {
        // e.g. the page already attached its own MediaElementSource
        this.state = 'failed';
        this.error = e.message;
        window.removeEventListener('pointerdown', this.onGesture, true);
        window.removeEventListener('keydown', this.onGesture, true);
      } finally {
        this.connecting = false;
      }
    }

    // Follow the video delay (seconds, wall clock) with the policy above; nowMs is the display time.
    follow(want, nowMs) {
      const sec = Math.floor(nowMs / 1000);
      const last = this.peaks[this.peaks.length - 1];
      if (last && last[0] === sec) last[1] = Math.max(last[1], want);
      else this.peaks.push([sec, want]);
      while (this.peaks.length && this.peaks[0][0] <= sec - WINDOW_MS / 1000) this.peaks.shift();
      const peak = Math.max(...this.peaks.map((p) => p[1]));
      const cur = (this.pending != null ? this.pending : this.target) + (this.ctx ? this.ctx.baseLatency || 0 : 0);
      if (want > cur || cur - (peak + HEADROOM) > STEP_DOWN) this.set(Math.max(want, peak) + HEADROOM);
    }

    // Set the delay at once (seconds, wall clock), by a crossfade; 0 also clears the history.
    set(sec) {
      if (sec === 0) this.peaks = [];
      const extra = this.ctx ? this.ctx.baseLatency || 0 : 0;
      const v = Math.max(0, Math.min(0.9, sec - extra));
      if (!this.lines) {
        this.target = v;
        return;
      }
      const now = this.ctx.currentTime;
      if (now < this.fadeEnd) {
        this.pending = v;
        clearTimeout(this.pendingTimer);
        this.pendingTimer = setTimeout(() => {
          const p = this.pending;
          this.pending = null;
          if (p != null) this.set(p + extra);
        }, (this.fadeEnd - now) * 1000 + 20);
        return;
      }
      this.pending = null;
      if (Math.abs(v - this.target) < 0.002) return;
      this.target = v;
      const [on, off] = this.lines;
      off.delay.delayTime.setValueAtTime(v, now);
      on.gain.gain.cancelScheduledValues(now);
      off.gain.gain.cancelScheduledValues(now);
      on.gain.gain.setValueCurveAtTime(fadeOut, now + 0.005, FADE);
      off.gain.gain.setValueCurveAtTime(fadeIn, now + 0.005, FADE);
      this.fadeEnd = now + 0.005 + FADE;
      this.lines = [off, on];
    }

    // the delay being applied, in seconds (including the AudioContext's own latency)
    get delay() {
      return this.target + (this.ctx ? this.ctx.baseLatency || 0 : 0);
    }

    get active() {
      return this.state === 'active';
    }
  }

  S.AudioDelay = AudioDelay;
})();
