// Shared WebGPU device + engine presets.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});

  // Presets = width of the RIFE input (frames are resized to it, sides rounded to multiples of 32).
  // The inference cost grows with the pixel count: about 10 / 23 / 40 G multiply-adds per source frame.
  S.PRESETS = {
    fast: { modelWidth: 512 },
    standard: { modelWidth: 768 },
    quality: { modelWidth: 1024 },
  };

  // Tuning shared by all presets.
  S.TUNING = {
    maxFrameWidth: 2560, // frames wider than this are downscaled on ingest
    sceneFrac: 0.45, // share of pixels whose warped samples disagree that marks a scene change
    masking: 0, // 0..1, artifact masking towards a plain blend
  };

  S.getDevice = function getDevice() {
    if (S._devicePromise) return S._devicePromise;
    S._devicePromise = (async () => {
      if (!navigator.gpu) throw new Error('WebGPU が利用できません (navigator.gpu なし)');
      // powerPreference is ignored on Windows and only produces a console warning there (crbug.com/369219127)
      const platform = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '';
      const opts = /^Win/i.test(platform) ? undefined : { powerPreference: 'high-performance' };
      const adapter = await navigator.gpu.requestAdapter(opts);
      if (!adapter) throw new Error('WebGPU アダプターを取得できません');
      // timestamp queries let the player measure its own GPU cost (performance guard)
      const requiredFeatures = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
      const device = await adapter.requestDevice({ requiredFeatures });
      device.lost.then((info) => {
        S._devicePromise = null;
        if (S.onDeviceLost) S.onDeviceLost(info);
      });
      return device;
    })();
    S._devicePromise.catch(() => {
      S._devicePromise = null;
    });
    return S._devicePromise;
  };

  // RIFE network for a device (weights are fetched once and kept for device re-creation).
  S.getRifeNet = function getRifeNet(device) {
    if (S._net && S._net.device === device) return S._net.promise;
    if (!S._weightsPromise) {
      S._weightsPromise = S.loadRifeWeights(chrome.runtime.getURL('models/rife-v4.6.bin'));
      S._weightsPromise.catch(() => {
        S._weightsPromise = null;
      });
    }
    const promise = S._weightsPromise.then((w) => new S.RifeNet(device, w));
    S._net = { device, promise };
    promise.catch(() => {
      if (S._net && S._net.promise === promise) S._net = null;
    });
    return promise;
  };
})();
