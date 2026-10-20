// Keyboard shortcuts: flip settings in storage; content scripts react to the storage change.
chrome.commands.onCommand.addListener(async (command) => {
  const s = await chrome.storage.sync.get(['enabled', 'demo']);
  if (command === 'toggle-enabled') await chrome.storage.sync.set({ enabled: s.enabled === false });
  if (command === 'toggle-demo') await chrome.storage.sync.set({ demo: !s.demo });
});
