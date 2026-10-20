// Small statistics overlay (top-left of the video), toggled in the popup.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});

  class OSD {
    constructor(overlay) {
      this.overlay = overlay;
      this.el = document.createElement('div');
      this.el.className = 'nnf-ai-osd';
      this.el.style.cssText =
        'position:absolute;pointer-events:none;z-index:2147483646;font:12px/1.35 Consolas,monospace;' +
        'color:#fff;background:rgba(0,0,0,.6);padding:4px 7px;border-radius:4px;white-space:pre;display:none;';
      this.on = false;
    }

    show(on) {
      this.on = on;
      this.el.style.display = on ? 'block' : 'none';
      if (!on) this.el.remove();
    }

    update(text) {
      if (!this.on) return;
      const c = this.overlay.canvas;
      if (this.el.parentNode !== c.parentNode && c.parentNode) c.parentNode.insertBefore(this.el, c.nextSibling);
      this.el.style.left = parseFloat(c.style.left || 0) + 8 + 'px';
      this.el.style.top = parseFloat(c.style.top || 0) + 8 + 'px';
      this.el.textContent = text;
    }

    destroy() {
      this.el.remove();
    }
  }

  S.OSD = OSD;
})();
