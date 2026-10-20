// Debug capture (started from the popup): for a few seconds, saves every new source frame as PNG
// (in a zip, with their media timestamps) and records the interpolated output canvas as WebM, so
// interpolation quality can be checked offline. Everything stays on this computer.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});
  const MAX_WIDTH = 1280; // source frames are scaled down to this width

  // CRC-32 for the zip writer
  const CRC = new Uint32Array(256).map((_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }

  // Minimal zip (stored, no compression): files = [{ name, data: Uint8Array }] -> Blob
  function zip(files) {
    const enc = new TextEncoder();
    const parts = [];
    const central = [];
    let offset = 0;
    for (const f of files) {
      const name = enc.encode(f.name);
      const crc = crc32(f.data);
      const h = new DataView(new ArrayBuffer(30));
      h.setUint32(0, 0x04034b50, true);
      h.setUint16(4, 20, true);
      h.setUint32(14, crc, true);
      h.setUint32(18, f.data.length, true);
      h.setUint32(22, f.data.length, true);
      h.setUint16(26, name.length, true);
      parts.push(h.buffer, name, f.data);
      const c = new DataView(new ArrayBuffer(46));
      c.setUint32(0, 0x02014b50, true);
      c.setUint16(4, 20, true);
      c.setUint16(6, 20, true);
      c.setUint32(16, crc, true);
      c.setUint32(20, f.data.length, true);
      c.setUint32(24, f.data.length, true);
      c.setUint16(28, name.length, true);
      c.setUint32(42, offset, true);
      central.push(c.buffer, name);
      offset += 30 + name.length + f.data.length;
    }
    const size = central.reduce((a, b) => a + b.byteLength, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, files.length, true);
    end.setUint16(10, files.length, true);
    end.setUint32(12, size, true);
    end.setUint32(16, offset, true);
    return new Blob([...parts, ...central, end.buffer], { type: 'application/zip' });
  }

  function download(blob, name) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
  }

  class Capture {
    constructor(video, canvas, seconds, info) {
      this.video = video;
      this.seconds = seconds;
      this.info = info;
      this.frames = [];
      this.state = 'recording';
      this.until = null; // media time at which source capture stops
      this.chunks = [];
      try {
        const stream = canvas.captureStream();
        const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((m) => MediaRecorder.isTypeSupported(m));
        this.recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 40e6 });
        this.recorder.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
        this.recorder.start();
      } catch (e) {
        this.recorder = null;
      }
      this.timer = setTimeout(() => this.finish(), (seconds + 0.5) * 1000 / Math.max(0.1, video.playbackRate));
    }

    // Called for every new source frame (requestVideoFrameCallback).
    onFrame(mediaTime) {
      if (this.state !== 'recording') return;
      if (this.until == null) this.until = mediaTime + this.seconds;
      if (mediaTime > this.until) return;
      const v = this.video;
      const s = Math.min(1, MAX_WIDTH / v.videoWidth);
      const c = new OffscreenCanvas(Math.round(v.videoWidth * s), Math.round(v.videoHeight * s));
      c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
      this.frames.push({ mediaTime, blob: c.convertToBlob({ type: 'image/png' }) });
    }

    async finish() {
      if (this.state !== 'recording') return;
      this.state = 'saving';
      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const webm = this.recorder
        ? new Promise((res) => {
            this.recorder.onstop = () => res(new Blob(this.chunks, { type: 'video/webm' }));
            this.recorder.stop();
          })
        : null;
      try {
        const files = [];
        for (let i = 0; i < this.frames.length; i++) {
          const f = this.frames[i];
          files.push({ name: `frames/${String(i).padStart(4, '0')}.png`, data: new Uint8Array(await (await f.blob).arrayBuffer()) });
        }
        const meta = {
          ...this.info,
          capturedAt: new Date().toISOString(),
          frames: this.frames.map((f, i) => ({ file: files[i].name, mediaTime: f.mediaTime })),
        };
        files.push({ name: 'meta.json', data: new TextEncoder().encode(JSON.stringify(meta, null, 1)) });
        download(zip(files), `nnf-source-${stamp}.zip`);
        if (webm) download(await webm, `nnf-output-${stamp}.webm`);
        this.state = `保存しました（元フレーム ${this.frames.length} 枚${webm ? ' + 出力動画' : ''}）`;
      } catch (e) {
        this.state = '保存に失敗しました: ' + e.message;
      }
      this.frames = [];
    }

    cancel() {
      clearTimeout(this.timer);
      if (this.recorder && this.recorder.state !== 'inactive') this.recorder.stop();
      this.state = 'cancelled';
      this.frames = [];
    }
  }

  S.Capture = Capture;
})();
