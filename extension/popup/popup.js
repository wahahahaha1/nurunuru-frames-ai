(() => {
  const S = globalThis.NNF;
  const $ = (id) => document.getElementById(id);
  let settings = S.mergeSettings({});

  const fmt = {
    masking: (v) => `${v}`,
    sceneSensitivity: (v) => `${v}`,
    audioOffsetMs: (v) => `${v > 0 ? '+' : ''}${v} ms`,
  };

  let lastStatus = null;

  // Smoothest targets divide the display refresh rate evenly (180 Hz: 180 / 90 / 60 / 45).
  function renderTargetHint() {
    const el = $('targetHint');
    const hz = lastStatus && lastStatus.displayHz;
    if (!hz) {
      el.textContent = 'モニターの Hz を割り切れる値にすると最もなめらかです';
      return;
    }
    const H = Math.round(hz);
    const divs = [1, 2, 3, 4].map((n) => Math.round(H / n)).filter((v) => v >= 24);
    let text = `${H} Hz の画面では ${divs.join(' / ')} fps が最もなめらかです`;
    const tgt = lastStatus.targetFps;
    if (tgt && !lastStatus.reason && tgt < hz * 0.97) {
      const n = hz / tgt;
      const nr = Math.round(n);
      if (Math.abs(n - nr) >= 0.04 * nr) {
        text += `（今の目標 ${tgt.toFixed(0)} fps は割り切れないため、表示間隔が少し不揃いになります）`;
      }
    }
    el.textContent = text;
  }

  function render() {
    $('enabled').checked = settings.enabled;
    $('target').value = settings.target;
    $('targetFps').value = settings.targetFps;
    $('targetFps').disabled = settings.target !== 'fixed';
    renderTargetHint();
    for (const el of document.querySelectorAll('[data-site]')) el.checked = settings.sites[el.dataset.site] !== false;
    for (const b of $('preset').children) b.classList.toggle('sel', b.dataset.v === settings.preset);
    for (const id of ['masking', 'sceneSensitivity', 'audioOffsetMs']) {
      $(id).value = settings[id];
      $(id + 'Out').textContent = fmt[id](settings[id]);
    }
    for (const id of ['hybrid', 'anchors', 'anime', 'audioSync', 'demo', 'showVectors', 'osd']) $(id).checked = settings[id];
  }

  function save(patch) {
    settings = S.mergeSettings({ ...settings, ...patch });
    render();
    S.saveSettings(patch);
  }

  $('enabled').onchange = (e) => save({ enabled: e.target.checked });
  for (const el of document.querySelectorAll('[data-site]')) {
    el.onchange = () => save({ sites: { ...settings.sites, [el.dataset.site]: el.checked } });
  }
  for (const b of $('preset').children) b.onclick = () => save({ preset: b.dataset.v });
  $('target').onchange = (e) => save({ target: e.target.value });
  $('targetFps').onchange = (e) => save({ targetFps: Math.max(24, Math.min(500, Math.round(+e.target.value || 60))) });
  for (const id of ['masking', 'sceneSensitivity', 'audioOffsetMs']) {
    $(id).oninput = (e) => ($(id + 'Out').textContent = fmt[id](+e.target.value));
    $(id).onchange = (e) => save({ [id]: +e.target.value });
  }
  for (const id of ['hybrid', 'anchors', 'anime', 'audioSync', 'demo', 'showVectors', 'osd']) $(id).onchange = (e) => save({ [id]: e.target.checked });

  $('capture').onclick = async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab) await chrome.tabs.sendMessage(tab.id, { type: 'nnf-capture', seconds: 3 });
    } catch (e) {
      $('captureHint').textContent = 'このタブでは記録できません（YouTube / ニコニコ動画の再生中に押してください）';
    }
    refreshStatus();
  };

  $('bench').onclick = (e) => {
    e.preventDefault();
    chrome.tabs.create({ url: chrome.runtime.getURL('bench/bench.html') });
  };

  const audioText = { waiting: 'ページをクリックすると有効', idle: '待機', failed: '使用不可', off: 'OFF' };

  async function refreshStatus() {
    const el = $('status');
    let st = null;
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab) st = await chrome.tabs.sendMessage(tab.id, { type: 'nnf-status' });
    } catch (e) {
      st = null;
    }
    lastStatus = st;
    if (st && st.capture) {
      $('captureHint').textContent = st.capture === 'recording' ? '記録中…（動画を再生したままにしてください）' : st.capture === 'saving' ? '保存中…' : st.capture;
    }
    $('capture').disabled = !!(st && (st.capture === 'recording' || st.capture === 'saving'));
    renderTargetHint();
    if (!st) {
      el.className = 'status';
      el.textContent = settings.enabled
        ? 'このタブでは対象の動画が見つかりません。\n（YouTube / ニコニコ動画の再生ページで使えます）'
        : '補間は OFF です。';
      return;
    }
    el.className = 'status on';
    if (st.reason) {
      el.textContent = `${st.site}: ${st.reason}`;
      return;
    }
    const gpu =
      st.gpuLoad == null
        ? '計測中'
        : `${(st.gpuLoad * 100).toFixed(0)}%（AI 推論 ${st.meMs.toFixed(1)} ms` +
          (st.renderMs != null ? ` / 描画 ${st.renderMs.toFixed(2)} ms）` : '）');
    el.textContent =
      `${st.site}  ${st.src}\n` +
      `${st.srcFps.toFixed(2)} fps → ${st.outFps.toFixed(0)} fps（目標 ${st.targetFps.toFixed(0)}・表示 ${st.displayHz.toFixed(0)} Hz）\n` +
      `補間 ${st.interpPct.toFixed(0)}%　映像遅延 ${st.delayMs.toFixed(0)} ms\n` +
      `音ズレ補正 ${st.audio === 'active' ? `音声を +${st.audioDelayMs.toFixed(0)} ms 遅延中` : audioText[st.audio] || st.audio}\n` +
      `GPU 負荷 ${gpu}\n` +
      `RIFE ${st.me}（${st.chunks} 分割）　underrun ${st.underruns}　drop ${st.drops}` +
      (st.perfNote ? `\n${st.perfNote}` : '') +
      (st.errors ? `\nエラー ${st.errors}: ${st.lastError}` : '');
  }

  S.loadSettings().then((s) => {
    settings = s;
    render();
    refreshStatus();
    setInterval(refreshStatus, 1000);
  });
})();
