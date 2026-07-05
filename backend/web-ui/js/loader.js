const pages = ['dashboard', 'hypotheses', 'graph', 'export', 'docs'];

async function loadAllPages() {
  try {
    await Promise.all(pages.map(async name => {
      const response = await fetch(`/pages/${name}.html`);
      const html = await response.text();
      const container = document.getElementById(`page-${name}`);
      if (container) {
        container.innerHTML = html;
      }
    }));

    console.log('[HF] All pages loaded');

    if (window.onHFReady) {
      await window.onHFReady();
    }
  } catch (error) {
    console.error('[HF] Failed to load pages:', error);
  }
}

loadAllPages();
