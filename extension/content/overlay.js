// Canvas overlay that sits exactly on top of the page's <video> element.
// It is inserted as the video's next sibling with the same z-index, so anything the site
// stacks above the video (comments, controls, storyboards) stays above the canvas too.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});

  class Overlay {
    constructor(video) {
      this.video = video;
      this.canvas = document.createElement('canvas');
      this.canvas.className = 'nnf-ai-canvas';
      this.canvas.style.cssText =
        'position:absolute;pointer-events:none;margin:0;padding:0;border:0;display:none;background:transparent;';
      this.visible = false;
      this.fit = 'contain';
      this.lastCheck = 0;
      this.box = '';
      this.ensureAttached();
      this.ro = new ResizeObserver(() => this.update());
      this.ro.observe(video);
      this.mo = new MutationObserver(() => this.update());
      this.mo.observe(video, { attributes: true, attributeFilter: ['style', 'class'] });
      this.onWin = () => this.update();
      window.addEventListener('resize', this.onWin);
      document.addEventListener('fullscreenchange', this.onWin);
      this.update();
    }

    ensureAttached() {
      const v = this.video;
      if (!v.parentNode) return false;
      if (this.canvas.parentNode !== v.parentNode || this.canvas.previousSibling !== v) {
        v.parentNode.insertBefore(this.canvas, v.nextSibling);
      }
      return true;
    }

    // Copy the video's box (position, size, z-index, transform) onto the canvas.
    update() {
      const v = this.video;
      const c = this.canvas;
      if (!this.ensureAttached()) return;
      const cs = getComputedStyle(v);
      this.fit = cs.objectFit || 'contain';
      const transformed = cs.transform && cs.transform !== 'none';
      const st = c.style;
      st.zIndex = cs.zIndex === 'auto' ? '' : cs.zIndex;
      st.transform = transformed ? cs.transform : '';
      st.transformOrigin = cs.transformOrigin;
      st.borderRadius = cs.borderRadius;
      let left = v.offsetLeft;
      let top = v.offsetTop;
      let width = v.offsetWidth;
      let height = v.offsetHeight;
      if (!transformed) {
        const vr = v.getBoundingClientRect();
        width = vr.width;
        height = vr.height;
        st.left = left + 'px';
        st.top = top + 'px';
        // correct sub-pixel offsets (offsetLeft/Top are rounded)
        const cr = c.getBoundingClientRect();
        if (c.style.display !== 'none') {
          left += vr.left - cr.left;
          top += vr.top - cr.top;
        }
      }
      st.left = left + 'px';
      st.top = top + 'px';
      st.width = width + 'px';
      st.height = height + 'px';
      const dpr = window.devicePixelRatio || 1;
      const pw = Math.max(1, Math.round(width * dpr));
      const ph = Math.max(1, Math.round(height * dpr));
      if (c.width !== pw) c.width = pw;
      if (c.height !== ph) c.height = ph;
      this.box = [left, top, width, height, this.fit].join();
    }

    // Cheap periodic re-check (called from the render loop).
    poll(now) {
      if (now - this.lastCheck < 250) return;
      this.lastCheck = now;
      const r = this.video.getBoundingClientRect();
      const key = [r.left, r.top, r.width, r.height].join();
      if (key !== this.lastRect || this.canvas.previousSibling !== this.video) {
        this.lastRect = key;
        this.update();
      }
    }

    // Content rectangle of the video frame inside the canvas (device pixels), honouring object-fit.
    contentRect(vw, vh) {
      const cw = this.canvas.width;
      const ch = this.canvas.height;
      if (!vw || !vh || this.fit === 'fill') return [0, 0, cw, ch];
      const s = this.fit === 'cover' ? Math.max(cw / vw, ch / vh) : Math.min(cw / vw, ch / vh);
      const w = vw * s;
      const h = vh * s;
      return [(cw - w) / 2, (ch - h) / 2, w, h];
    }

    show(on) {
      if (on === this.visible) return;
      this.visible = on;
      this.canvas.style.display = on ? 'block' : 'none';
      if (on) this.update();
    }

    destroy() {
      this.ro.disconnect();
      this.mo.disconnect();
      window.removeEventListener('resize', this.onWin);
      document.removeEventListener('fullscreenchange', this.onWin);
      this.canvas.remove();
    }
  }

  S.Overlay = Overlay;
})();
