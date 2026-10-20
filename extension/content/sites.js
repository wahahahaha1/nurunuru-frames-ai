// Site adapters: which <video> is the main player on each site.
(() => {
  const S = (globalThis.NNF = globalThis.NNF || {});

  // Largest visible MSE-backed (blob:) video, for players whose DOM we do not know.
  function genericVideo(minW = 240) {
    let best = null;
    let bestArea = 0;
    for (const v of document.querySelectorAll('video')) {
      if (!(v.currentSrc || '').startsWith('blob:')) continue;
      const r = v.getBoundingClientRect();
      const area = r.width * r.height;
      if (r.width >= minW && area > bestArea) {
        best = v;
        bestArea = area;
      }
    }
    return best;
  }

  S.SITES = [
    {
      id: 'youtube',
      name: 'YouTube',
      match: (host) => /(^|\.)youtube(-nocookie)?\.com$/.test(host),
      find: () =>
        document.querySelector('#movie_player video.html5-main-video') ||
        document.querySelector('.html5-video-player video.html5-main-video') ||
        genericVideo(),
    },
    {
      id: 'niconico',
      name: 'ニコニコ動画',
      match: (host) => /(^|\.)nicovideo\.jp$/.test(host),
      // The main player is video[data-name="video-content"]; ads play in separate cross-origin videos.
      find: () => document.querySelector('video[data-name="video-content"]') || genericVideo(),
    },
  ];

  S.detectSite = (host = location.hostname) => S.SITES.find((s) => s.match(host)) || null;
})();
