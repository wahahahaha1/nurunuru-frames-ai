// Settings shared by the content script, popup and background (chrome.storage.sync).
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});

  S.DEFAULTS = {
    enabled: true,
    sites: { youtube: true, niconico: true },
    preset: 'standard', // fast | standard | quality
    target: 'display', // output rate: display | x2 | x2.5 | x3 (times the source fps) | fixed
    targetFps: 120, // used when target === 'fixed'
    masking: 0, // 0..100: blend towards plain frame mixing where motion vectors are unreliable
    sceneSensitivity: 60, // 0..100: higher = scene changes detected more readily
    audioSync: false, // delay audio by the interpolation latency (routes the video's audio through WebAudio)
    audioOffsetMs: 0, // extra audio delay (+) or advance (-)
    demo: false, // split screen: left original, right interpolated
    showVectors: false,
    osd: false,
  };

  const hasStorage = () => typeof chrome !== 'undefined' && chrome.storage && chrome.storage.sync;

  S.mergeSettings = (stored) => ({
    ...S.DEFAULTS,
    ...stored,
    sites: { ...S.DEFAULTS.sites, ...(stored && stored.sites) },
  });

  S.loadSettings = async () => {
    if (!hasStorage()) return S.mergeSettings({});
    try {
      return S.mergeSettings(await chrome.storage.sync.get(null));
    } catch (e) {
      return S.mergeSettings({});
    }
  };

  S.saveSettings = async (patch) => {
    if (!hasStorage()) return;
    await chrome.storage.sync.set(patch);
  };

  S.onSettingsChanged = (cb) => {
    if (!hasStorage()) return;
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === 'sync') S.loadSettings().then(cb);
    });
  };

  // Settings -> engine parameters
  S.engineParams = (s) => ({
    ...(S.PRESETS[s.preset] || S.PRESETS.standard),
    masking: s.masking / 100,
    sceneFrac: 0.9 - (0.75 * s.sceneSensitivity) / 100,
  });
})();
