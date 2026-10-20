// Delays the video's audio by the interpolation latency (WebAudio DelayNode), so lips stay in sync.
// Routing a media element through WebAudio is permanent, so it is only done once the AudioContext
// is confirmed running (after a user gesture); otherwise the element is left untouched.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});

  class AudioDelay {
    constructor(video) {
      this.video = video;
      this.ctx = null;
      this.delayNode = null;
      this.state = 'idle'; // idle | waiting | active | failed
      this.target = 0;
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
        this.delayNode = this.ctx.createDelay(1.0);
        this.delayNode.delayTime.value = this.target;
        src.connect(this.delayNode).connect(this.ctx.destination);
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

    // Delay in seconds (wall clock). The AudioContext's own processing latency is subtracted.
    set(sec) {
      const extra = this.ctx ? this.ctx.baseLatency || 0 : 0;
      const v = Math.max(0, Math.min(0.9, sec - extra));
      if (Math.abs(v - this.target) < 0.002) return;
      this.target = v;
      if (this.delayNode) this.delayNode.delayTime.setTargetAtTime(v, this.ctx.currentTime, 0.05);
    }

    get active() {
      return this.state === 'active';
    }
  }

  S.AudioDelay = AudioDelay;
})();
