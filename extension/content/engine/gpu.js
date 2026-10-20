// Shared WebGPU device + engine presets.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});

  // Presets = RIFE model and the width of its input (frames are resized to it; the coarse blocks pad
  // the sides to multiples of 64 / 128). 4.25 is the model the RIFE author recommends for most
  // scenes (and much better than earlier ones on anime); lite has a coarser first block (1/32) and
  // a lighter last one.
  S.PRESETS = {
    fast: { model: 'rife-4.25-lite', modelWidth: 768 },
    standard: { model: 'rife-4.25', modelWidth: 960 },
    quality: { model: 'rife-4.25', modelWidth: 1280 },
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

  // RIFE network of a model for a device (weights are fetched once and kept for device re-creation).
  S.getRifeNet = function getRifeNet(device, model) {
    S._weights = S._weights || new Map();
    S._nets = S._nets || new Map();
    const key = model;
    const cached = S._nets.get(key);
    if (cached && cached.device === device) return cached.promise;
    let wp = S._weights.get(model);
    if (!wp) {
      const url = (globalThis.chrome && chrome.runtime && chrome.runtime.getURL ? chrome.runtime.getURL('') : '../') + S.RIFE_MODELS[model].file;
      wp = S.loadRifeWeights(url);
      S._weights.set(model, wp);
      wp.catch(() => S._weights.delete(model));
    }
    const promise = wp.then((w) => new S.RifeNet(device, w));
    S._nets.set(key, { device, promise });
    promise.catch(() => {
      if (S._nets.get(key) && S._nets.get(key).promise === promise) S._nets.delete(key);
    });
    return promise;
  };
})();
