async function init() {
  console.log('[HF] Initializing application...');

  let currentUser = null;
  let uploadedFiles = [];
  let currentProblem = null;

  const API_CONFIG = window.__HF_CONFIG__ || { API_BASE: '/api/v1' };
  const API = API_CONFIG.API_BASE;

  function showSuccess(message) {
    const toast = document.createElement('div');
    toast.className = 'toast toast-success';
    toast.innerHTML = `
      <i data-lucide="check-circle-2"></i>
      <span>${message}</span>
    `;
    document.body.appendChild(toast);
    lucide.createIcons();
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  function showError(message) {
    const toast = document.createElement('div');
    toast.className = 'toast toast-error';
    toast.innerHTML = `
      <i data-lucide="x-circle"></i>
      <span>${message}</span>
    `;
    document.body.appendChild(toast);
    lucide.createIcons();
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 300);
    }, 5000);
  }

  function showLoading(message = 'Загрузка...') {
    const overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
      <div class="loading-spinner"></div>
      <div class="loading-text">${message}</div>
    `;
    document.body.appendChild(overlay);
  }

  function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.remove();
  }

  async function apiRequest(endpoint, options = {}) {
    const url = endpoint.startsWith('http') ? endpoint : `${API}${endpoint}`;
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(`API ${endpoint} → ${response.status}`);
    }
    return response.json();
  }

  async function checkHealth() {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2000);

      const response = await fetch('/api/health', { signal: controller.signal });
      clearTimeout(timeout);

      if (!response.ok) return false;
      const result = await response.json();
      if (result.status === 'ok') {
        console.log('[HF] Backend connected');
        const statusEl = document.getElementById('apiStatus');
        if (statusEl) statusEl.textContent = 'подключён';
        return true;
      }
      return false;
    } catch (error) {
      console.error('[HF] Backend not available:', error.message);
      const statusEl = document.getElementById('apiStatus');
      if (statusEl) statusEl.textContent = 'недоступен';
      return false;
    }
  }

  async function switchPage(targetPage, updateHash = true) {
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');
    const headers = document.querySelectorAll('.header');

    navItems.forEach(n => n.classList.remove('active'));
    pages.forEach(p => p.classList.remove('active'));
    headers.forEach(h => h.classList.remove('active'));

    const activeNav = document.querySelector(`.nav-item[data-page="${targetPage}"]`);
    if (activeNav) activeNav.classList.add('active');

    const activePage = document.getElementById(`page-${targetPage}`);
    if (activePage) activePage.classList.add('active');

    const activeHeader = document.querySelector(`.header[data-header-for="${targetPage}"]`);
    if (activeHeader) activeHeader.classList.add('active');

    if (updateHash) {
      history.replaceState(null, '', `#${targetPage}`);
    }

    if (targetPage === 'hypotheses') {
      await loadAllProblems();
    }

    if (targetPage === 'docs') {
      loadDocuments();
    }

    if (targetPage === 'graph') {
      if (!currentProblem) {
        await loadAllProblems();
      }
      loadGraph();
    }
  }

  function pageFromHash() {
    const hash = (window.location.hash || '').replace(/^#/, '');
    const valid = ['dashboard', 'docs', 'hypotheses', 'graph', 'export'];
    return valid.includes(hash) ? hash : 'dashboard';
  }

  function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');

    navItems.forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        switchPage(item.dataset.page, true);
      });
    });

    document.querySelectorAll('[data-jump]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        switchPage(btn.dataset.jump, true);
      });
    });

    window.addEventListener('hashchange', () => {
      switchPage(pageFromHash(), false);
    });
  }

  function navigateFromInitialHash() {
    const page = pageFromHash();
    if (page !== 'dashboard') {
      switchPage(page, false);
    }
  }

  function setupDropzone() {
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');

    if (!dropzone || !fileInput) {
      console.warn('[HF] Dropzone elements not found');
      return;
    }

    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
      handleFiles(e.target.files);
    });

    function handleFiles(files) {
      fileList.style.display = 'block';
      fileList.innerHTML = '';
      uploadedFiles = [];

      Array.from(files).forEach(file => {
        uploadedFiles.push(file);
        const fileElement = document.createElement('div');
        fileElement.className = 'file-item';
        fileElement.innerHTML = `
          <span class="file-name">${file.name}</span>
          <span class="file-size">${(file.size / 1024 / 1024).toFixed(2)} MB</span>
        `;
        fileList.appendChild(fileElement);
      });

      const uploadBtn = document.getElementById('upload-btn');
      if (uploadBtn) {
        uploadBtn.disabled = false;
      }
    }
  }

  function setupUpload() {
    const uploadBtn = document.getElementById('upload-btn');
    const docType = document.getElementById('doc-type');
    const ocrProvider = document.getElementById('ocr-provider');

    if (!uploadBtn) {
      console.warn('[HF] Upload button not found');
      return;
    }

    uploadBtn.addEventListener('click', async () => {
      if (uploadedFiles.length === 0) {
        showError('Сначала добавьте файлы');
        return;
      }

      uploadBtn.disabled = true;
      uploadBtn.innerHTML = '<span class="loading"></span>Загрузка...';

      for (const file of uploadedFiles) {
        const formData = new FormData();
        formData.append('file', file);

        const params = new URLSearchParams();
        if (docType) params.append('doc_type', docType.value);
        if (ocrProvider) params.append('ocr_provider', ocrProvider.value);

        try {
          
          const uploadResponse = await fetch(`${API}/documents/upload?${params.toString()}`, {
            method: 'POST',
            body: formData,
          });
          if (!uploadResponse.ok) throw new Error('Upload failed');
          const { upload_id } = await uploadResponse.json();

          
          const progressDiv = document.createElement('div');
          progressDiv.className = 'upload-progress';
          progressDiv.innerHTML = `
            <div class="progress-header">
              <span class="progress-filename">${file.name}</span>
              <span class="progress-percent">0%</span>
            </div>
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" style="width: 0%"></div>
            </div>
            <div class="progress-stage">Подготовка...</div>
          `;
          uploadBtn.parentElement.appendChild(progressDiv);

          
          let pollInterval = setInterval(async () => {
            try {
              const progressResponse = await fetch(`${API}/documents/upload-progress/${upload_id}`);
              if (!progressResponse.ok) {
                clearInterval(pollInterval);
                return;
              }
              const progress = await progressResponse.json();

              if (progress.status === 'done') {
                clearInterval(pollInterval);
                progressDiv.querySelector('.progress-percent').textContent = '100%';
                progressDiv.querySelector('.progress-bar-fill').style.width = '100%';
                progressDiv.querySelector('.progress-stage').textContent = 'Завершено';
                progressDiv.classList.add('success');
                await loadDashboardStats();
              } else if (progress.status === 'error') {
                clearInterval(pollInterval);
                progressDiv.classList.add('error');
                progressDiv.querySelector('.progress-stage').textContent = progress.detail || 'Ошибка';
              } else {
                progressDiv.querySelector('.progress-percent').textContent = `${progress.percent}%`;
                progressDiv.querySelector('.progress-bar-fill').style.width = `${progress.percent}%`;
                progressDiv.querySelector('.progress-stage').textContent = progress.detail || progress.stage;
              }
            } catch (err) {
              console.error('Progress poll error:', err);
            }
          }, 500);

          
          setTimeout(() => clearInterval(pollInterval), 60000);

        } catch (error) {
          console.error(`[HF] Failed to upload ${file.name}:`, error);
          showError(`Ошибка загрузки ${file.name}: ${error.message}`);
        }
      }

      uploadBtn.disabled = false;
      uploadBtn.innerHTML = 'Загрузить документы';
      uploadedFiles = [];

      const fileList = document.getElementById('file-list');
      if (fileList) {
        fileList.innerHTML = '';
        fileList.style.display = 'none';
      }
      
      showSuccess('Документы успешно загружены');
    });
  }

  function setupGeneration() {
    const generateBtn = document.getElementById('generate-btn');
    const problemInput = document.getElementById('problem-text');
    const hypothesisCount = document.getElementById('hypothesis-count');
    const strategy = document.getElementById('strategy');
    const generationMode = document.getElementById('generation-mode');
    const modelToggle = document.getElementById('model-toggle');
    const genWarning = document.getElementById('gen-warning');

    if (modelToggle) {
      modelToggle.addEventListener('click', (e) => {
        const option = e.target.closest('.model-option');
        if (!option) return;
        modelToggle.querySelectorAll('.model-option').forEach(o => o.classList.remove('active'));
        option.classList.add('active');
        option.querySelector('input').checked = true;
      });
    }

    if (!generateBtn) {
      console.warn('[HF] Generate button not found');
      return;
    }

    let generating = false;

    function setGenerating(active) {
      generating = active;
      if (genWarning) genWarning.style.display = active ? 'flex' : 'none';
    }

    window.addEventListener('beforeunload', (e) => {
      if (generating) {
        e.preventDefault();
        e.returnValue = '';
      }
    });

    generateBtn.addEventListener('click', async () => {
      if (!problemInput || !problemInput.value.trim()) {
        showError('Введите формулировку проблемы');
        return;
      }

      const selectedModel = modelToggle
        ? (modelToggle.querySelector('input:checked') || {}).value || 'deepseek-v4-flash'
        : 'deepseek-v4-flash';

      generateBtn.disabled = true;
      generateBtn.innerHTML = '<span class="loading"></span>Генерация...';
      setGenerating(true);
      showLoading('Генерация гипотез... Это может занять 1-2 минуты');

      try {
        const payload = {
          statement: problemInput.value.trim(),
          num_hypotheses: hypothesisCount ? parseInt(hypothesisCount.value) : 5,
          strategy: strategy ? strategy.value : 'auto',
          model: selectedModel,
          mode: generationMode ? generationMode.value : 'standard',
        };

        const problemResponse = await apiRequest('/problems', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        currentProblem = await apiRequest('/problems/' + problemResponse.problem_id);
        try { localStorage.setItem('hf-last-problem-id', problemResponse.problem_id); } catch (e) {}

        try { localStorage.setItem('lastProblemId', currentProblem.problem_id); } catch (e) {}

        console.log('[HF] Generated problem:', currentProblem);
        hideLoading();
        setGenerating(false);
        showSuccess(`Сгенерировано ${currentProblem.hypotheses?.length || 0} гипотез`);
        loadDashboardStats();
        switchPage('hypotheses');

      } catch (error) {
        console.error('[HF] Generation failed:', error);
        hideLoading();
        setGenerating(false);
        showError('Ошибка генерации: ' + error.message);
      }

      generateBtn.disabled = false;
      generateBtn.innerHTML = 'Сгенерировать гипотезы';
    });
  }

  async function loadAllProblems() {
    try {
      const res = await apiRequest('/problems');
      const list = Array.isArray(res) ? res : (res.problems || []);
      if (list.length === 0) {
        renderAllProblems([]);
        return;
      }
      const problems = await Promise.all(
        list.map(p => apiRequest(`/problems/${p.problem_id}`))
      );
      if (problems.length > 0) {
        currentProblem = problems[0];
      }
      renderAllProblems(problems);
    } catch (error) {
      console.warn('[HF] Could not load problems:', error.message);
    }
  }

  function renderAllProblems(problems) {
    const tbody = document.getElementById('hypotheses-tbody');
    if (!tbody) return;

    if (!problems || problems.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7">Нет сгенерированных гипотез. Используйте форму на главной странице.</td></tr>';
      return;
    }

    tbody.innerHTML = '';

    problems.forEach((problem, pi) => {
      const hyps = problem.hypotheses || [];
      if (hyps.length === 0) return;

      const headerRow = document.createElement('tr');
      const date = problem.created_at ? new Date(problem.created_at).toLocaleString('ru-RU') : '';
      headerRow.innerHTML = `<td colspan="7" class="hyp-group-header">Запуск #${pi + 1} &mdash; ${date} &mdash; ${hyps.length} гипотез<br><span class="hyp-group-problem">${problem.statement || ''}</span></td>`;
      tbody.appendChild(headerRow);

      hyps.forEach((hyp, idx) => {
        const novelty = hyp.novelty || 0;
        const feasibility = hyp.feasibility || 0;
        const impact = hyp.impact || 0;
        const risk = hyp.risk || 0;
        const composite = hyp.composite_score || 0;

        const row = document.createElement('tr');
        row.innerHTML = `
          <td class="hyp-ranking">${idx + 1}</td>
          <td>
            <details class="hyp-details">
              <summary class="hyp-statement">${hyp.statement}</summary>
              <div class="hyp-body">
                <p class="hyp-mechanism">${hyp.mechanism || 'Механизм не указан'}</p>
                <p class="hyp-cot"><strong>Ход рассуждений:</strong> ${hyp.reasoning_trace || 'Недоступно'}</p>
                ${hyp.citations && hyp.citations.length > 0 ?
                  `<p class="hyp-citations"><strong>Источники:</strong> ${hyp.citations.join(', ')}</p>` : ''}
                ${hyp.verification_plan ?
                  `<p class="hyp-verification"><strong>План верификации:</strong> ${hyp.verification_plan}</p>` : ''}
              </div>
            </details>
          </td>
          <td><span class="score-bar" style="--score:${novelty}%">${novelty.toFixed(2)}</span></td>
          <td><span class="score-bar" style="--score:${feasibility}%">${feasibility.toFixed(2)}</span></td>
          <td><span class="score-bar" style="--score:${impact}%">${impact.toFixed(2)}</span></td>
          <td><span class="score-bar" style="--score:${risk}%">${risk.toFixed(2)}</span></td>
          <td><span class="composite-score">${composite.toFixed(3)}</span></td>
        `;
        tbody.appendChild(row);
      });
    });
    lucide.createIcons();
  }

  function renderHypotheses(hypotheses) {
    const tbody = document.getElementById('hypotheses-tbody');
    if (!tbody) return;

    if (!hypotheses || hypotheses.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7">Нет сгенерированных гипотез</td></tr>';
      return;
    }

    tbody.innerHTML = '';

    hypotheses.forEach((hyp, idx) => {
      const novelty = hyp.novelty || 0;
      const feasibility = hyp.feasibility || 0;
      const impact = hyp.impact || 0;
      const risk = hyp.risk || 0;
      const composite = hyp.composite_score || 0;

      const row = document.createElement('tr');
      row.innerHTML = `
        <td class="hyp-ranking">${idx + 1}</td>
        <td>
          <details class="hyp-details">
            <summary class="hyp-statement">${hyp.statement}</summary>
            <div class="hyp-body">
              <p class="hyp-mechanism">${hyp.mechanism || 'Механизм не указан'}</p>
              <p class="hyp-cot"><strong>Ход рассуждений:</strong> ${hyp.reasoning_trace || 'Недоступно'}</p>
              ${hyp.citations && hyp.citations.length > 0 ?
                `<p class="hyp-citations"><strong>Источники:</strong> ${hyp.citations.join(', ')}</p>` : ''}
              ${hyp.verification_plan ?
                `<p class="hyp-verification"><strong>План верификации:</strong> ${hyp.verification_plan}</p>` : ''}
            </div>
          </details>
        </td>
        <td><span class="score-bar" style="--score:${novelty}%">${novelty.toFixed(2)}</span></td>
        <td><span class="score-bar" style="--score:${feasibility}%">${feasibility.toFixed(2)}</span></td>
        <td><span class="score-bar" style="--score:${impact}%">${impact.toFixed(2)}</span></td>
        <td><span class="score-bar" style="--score:${risk}%">${risk.toFixed(2)}</span></td>
        <td><span class="composite-score">${composite.toFixed(3)}</span></td>
      `;
      tbody.appendChild(row);
    });
    lucide.createIcons();
  }

  async function loadDashboardStats() {
    try {
      const [docsRes, problemsRes] = await Promise.all([
        apiRequest('/documents').catch(() => ({ documents: [], total: 0 })),
        apiRequest('/problems').catch(() => [])
      ]);

      const docs = docsRes.documents || [];
      const problemsList = Array.isArray(problemsRes) ? problemsRes : (problemsRes.problems || problemsRes.data || []);

      const docsEl = document.getElementById('docsCountStat');
      if (docsEl) docsEl.textContent = docsRes.total || docs.length;

      const runsEl = document.getElementById('runsCountStat');
      if (runsEl) runsEl.textContent = problemsList.length;

      if (problemsList.length > 0) {
        const problemsDetail = await Promise.all(
          problemsList.map(p => apiRequest(`/problems/${p.problem_id}`).catch(() => ({ hypotheses: [] })))
        );
        const totalHyps = problemsDetail.reduce((sum, p) => sum + ((p.hypotheses || []).length), 0);
        const hypsEl = document.getElementById('hypsCountStat');
        if (hypsEl) hypsEl.textContent = totalHyps;

        if (!currentProblem && problemsDetail[0] && problemsDetail[0].hypotheses && problemsDetail[0].hypotheses.length > 0) {
          currentProblem = problemsDetail[0];
        }
      }

      const ocrEl = document.getElementById('ocrCountStat');
      if (ocrEl) ocrEl.textContent = '4';

    } catch (error) {
      console.error('[HF] Failed to load dashboard stats:', error);
    }
  }

  async function loadDocuments() {
    try {
      const response = await apiRequest('/documents');
      const docs = response.documents || [];

      const totalEl = document.getElementById('docsSumTotal');
      if (totalEl) totalEl.textContent = response.total || docs.length;

      const recognizedEl = document.getElementById('docsSumRecognized');
      if (recognizedEl) recognizedEl.textContent = docs.filter(d => d.status === 'indexed' || d.status === 'recognized').length;

      const chunksEl = document.getElementById('docsSumPages');
      if (chunksEl) chunksEl.textContent = docs.reduce((sum, d) => sum + (d.chunks_count || 0), 0);

      const hypsEl = document.getElementById('docsSumHyps');
      if (hypsEl) hypsEl.textContent = docs.reduce((sum, d) => sum + (d.hypotheses_count || 0), 0);

      renderDocsTable(docs);
      console.log(`[HF] Loaded ${response.total || docs.length} documents`);
    } catch (error) {
      console.error('[HF] Failed to load documents:', error);
      const totalEl = document.getElementById('docsSumTotal');
      if (totalEl) totalEl.textContent = '0';
    }
  }

  function renderDocsTable(docs) {
    const tbody = document.getElementById('docsTableBody');
    if (!tbody) return;

    if (!docs || docs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8">Нет документов</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    docs.forEach((doc, idx) => {
      const row = document.createElement('tr');
      const statusClass = doc.status === 'indexed' || doc.status === 'recognized' ? 'recognized' :
                         doc.status === 'failed' ? 'failed' : 'pending';
      const statusText = doc.status === 'indexed' || doc.status === 'recognized' ? 'Распознан' :
                        doc.status === 'failed' ? 'Ошибка' : 'В очереди';
      row.innerHTML = `
        <td>
          <div class="doc-name-cell">
            <div class="doc-icon">${(doc.name || doc.filename || '').slice(0, 3).toUpperCase()}</div>
            <div>
              <div class="doc-name" title="${doc.title || doc.name || doc.filename || 'Документ ' + (idx + 1)}">${doc.title || doc.name || doc.filename || 'Документ ' + (idx + 1)}</div>
              <div class="doc-sub">${doc.format || '—'}</div>
            </div>
          </div>
        </td>
        <td>${doc.type || '—'}</td>
        <td>${doc.ocr_provider || '—'}</td>
        <td>${doc.chunks_count || 0}</td>
        <td><span class="doc-status ${statusClass}"><span class="dot"></span>${statusText}</span></td>
        <td>${doc.hypotheses_count || 0}</td>
        <td>${doc.created_at ? new Date(doc.created_at).toLocaleDateString('ru-RU') : '—'}</td>
        <td>
          <div style="display:flex;gap:6px;">
            <button class="doc-action" data-action="download" data-doc-id="${doc.id || doc.doc_id || idx}" data-doc-name="${doc.title || doc.name || doc.filename || 'document'}">
              <i data-lucide="download"></i>
              Скачать
            </button>
            <button class="doc-action" data-action="download-ocr" data-doc-id="${doc.id || doc.doc_id || idx}" data-doc-name="${doc.title || doc.name || doc.filename || 'document'}">
              <i data-lucide="file-text"></i>
              OCR текст
            </button>
            <button class="doc-action doc-action-delete" data-action="delete" data-doc-id="${doc.id || doc.doc_id || idx}">
              <i data-lucide="trash-2"></i>
            </button>
          </div>
        </td>
      `;
      tbody.appendChild(row);
    });
    lucide.createIcons();
  }

  function setupDocsFilter() {
    const chips = document.querySelectorAll('.docs-chip');
    const searchInput = document.getElementById('docsSearch');

    chips.forEach(chip => {
      chip.addEventListener('click', () => {
        chips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        filterDocsTable(chip.dataset.filter, searchInput ? searchInput.value.toLowerCase() : '');
      });
    });

    if (searchInput) {
      searchInput.addEventListener('input', () => {
        const activeChip = document.querySelector('.docs-chip.active');
        filterDocsTable(activeChip ? activeChip.dataset.filter : 'all', searchInput.value.toLowerCase());
      });
    }

    const refreshBtn = document.getElementById('docsRefreshBtn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => loadDocuments());
    }
  }

  function filterDocsTable(filter, query) {
    const tbody = document.getElementById('docsTableBody');
    if (!tbody) return;
    const rows = tbody.querySelectorAll('tr');
    rows.forEach(row => {
      if (row.children.length < 8) { row.style.display = 'none'; return; }
      const text = row.textContent.toLowerCase();
      const matchesQuery = !query || text.includes(query);
      let matchesFilter = filter === 'all';
      if (!matchesFilter && row.children[4]) {
        const statusCell = row.children[4].textContent.toLowerCase();
        if (filter === 'indexed') matchesFilter = statusCell.includes('распознан');
        if (filter === 'pending') matchesFilter = statusCell.includes('очеред');
        if (filter === 'failed') matchesFilter = statusCell.includes('ошибк');
      }
      row.style.display = (matchesFilter && matchesQuery) ? '' : 'none';
    });
  }

  function setupExportList() {
    const items = document.querySelectorAll('.export-item');
    items.forEach(item => {
      item.addEventListener('click', async () => {
        const format = item.dataset.format;
        if (!format) return;
        
        if (!currentProblem || !currentProblem.problem_id) {
          showError('Сначала сгенерируйте гипотезы на главной странице');
          return;
        }
        
        try {
          showLoading(`Экспорт в ${format.toUpperCase()}...`);
          const response = await fetch(`${API}/export/${currentProblem.problem_id}?format=${format}`);
          if (!response.ok) throw new Error('Export failed');
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `hypothesis-factory-export-${Date.now()}.${format}`;
          a.click();
          window.URL.revokeObjectURL(url);
          hideLoading();
          showSuccess(`Экспорт в ${format.toUpperCase()} выполнен`);
        } catch (error) {
          hideLoading();
          showError('Ошибка экспорта: ' + error.message);
        }
      });
    });
  }

  function setupDocActions() {
    const tbody = document.getElementById('docsTableBody');
    if (!tbody) return;

    tbody.addEventListener('click', async (e) => {
      const btn = e.target.closest('button[data-action]');
      if (!btn) return;
      const action = btn.dataset.action;
      const docId = btn.dataset.docId;

      if (action === 'delete') {
        if (!confirm('Удалить документ? Это действие нельзя отменить.')) return;
        btn.disabled = true;
        btn.innerHTML = '<span class="loading"></span>';
        try {
          await apiRequest(`/documents/${docId}`, { method: 'DELETE' });
          await loadDocuments();
          showSuccess('Документ удален');
        } catch (error) {
          showError('Ошибка при удалении: ' + error.message);
          btn.disabled = false;
          btn.innerHTML = '<i data-lucide="trash-2"></i>';
          lucide.createIcons();
        }
      } else if (action === 'download') {
        try {
          showLoading('Скачивание документа...');
          const response = await fetch(`${API}/documents/${docId}/download`);
          if (!response.ok) throw new Error('Download failed');
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = btn.dataset.docName || `document-${docId}`;
          a.click();
          window.URL.revokeObjectURL(url);
          hideLoading();
          showSuccess('Документ скачан');
        } catch (error) {
          hideLoading();
          showError('Ошибка при скачивании: ' + error.message);
        }
      } else if (action === 'download-ocr') {
        try {
          showLoading('Скачивание OCR текста...');
          const response = await fetch(`${API}/documents/${docId}/ocr-pdf`);
          if (!response.ok) throw new Error('OCR PDF download failed');
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          const baseName = (btn.dataset.docName || 'document').replace(/\.[^/.]+$/, '');
          a.download = `${baseName}_ocr.pdf`;
          a.click();
          window.URL.revokeObjectURL(url);
          hideLoading();
          showSuccess('OCR текст скачан');
        } catch (error) {
          hideLoading();
          showError('Ошибка при скачивании OCR текста: ' + error.message);
        }
      }
    });
  }

  function setupGraph() {
    const refreshBtn = document.getElementById('graph-refresh-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => loadGraph());
    }

    const exportBtn = document.getElementById('graph-export-btn');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => {
        if (!window.cyInstance) {
          showError('Сначала постройте граф');
          return;
        }
        const elements = window.cyInstance.json().elements;
        const blob = new Blob([JSON.stringify(elements, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `graph-${Date.now()}.json`;
        a.click();
        window.URL.revokeObjectURL(url);
        showSuccess('Граф экспортирован');
      });
    }
  }

  async function loadGraph() {
    const cyContainer = document.getElementById('cy');
    const graphEmpty = document.getElementById('graph-empty');
    const graphLegend = document.getElementById('graph-legend');
    const cyWrapper = document.querySelector('.cy-wrapper');
    
    if (!cyContainer || !currentProblem || !currentProblem.problem_id) {
      if (graphEmpty) {
        graphEmpty.style.display = 'flex';
        graphEmpty.querySelector('.empty-text').textContent = 'Сначала сгенерируйте гипотезы на главной странице.';
      }
      if (cyWrapper) cyWrapper.style.display = 'none';
      if (graphLegend) graphLegend.style.display = 'none';
      return;
    }
    
    try {
      showLoading('Построение графа знаний...');
      
      const response = await apiRequest(`/graph?problem_id=${currentProblem.problem_id}`);
      const graphData = response.graph || response;
      
      const nodes = graphData.nodes || [];
      const edges = graphData.edges || [];
      
      if (nodes.length === 0) {
        hideLoading();
        if (graphEmpty) {
          graphEmpty.style.display = 'flex';
          graphEmpty.querySelector('.empty-text').textContent = 'Не удалось извлечь сущности. Попробуйте сгенерировать гипотезы с другими документами.';
        }
        if (cyWrapper) cyWrapper.style.display = 'none';
        if (graphLegend) graphLegend.style.display = 'none';
        return;
      }
      
      if (graphEmpty) graphEmpty.style.display = 'none';
      if (cyWrapper) cyWrapper.style.display = 'block';
      if (graphLegend) graphLegend.style.display = 'block';
      
      const nodesCountEl = document.getElementById('graph-nodes-count');
      const edgesCountEl = document.getElementById('graph-edges-count');
      const hypsCountEl = document.getElementById('graph-hyps-count');
      
      if (nodesCountEl) nodesCountEl.textContent = nodes.length;
      if (edgesCountEl) edgesCountEl.textContent = edges.length;
      if (hypsCountEl) {
        const hypsCount = nodes.filter(n => n.data && n.data.type === 'hypothesis').length;
        hypsCountEl.textContent = hypsCount;
      }
      
      renderCytoscapeGraph(nodes, edges);
      
      hideLoading();
      showSuccess('Граф знаний построен');
    } catch (error) {
      hideLoading();
      showError('Ошибка построения графа: ' + error.message);
      console.error('[HF] Failed to load graph:', error);
    }
  }

  function renderCytoscapeGraph(nodes, edges) {
    const cyContainer = document.getElementById('cy');
    if (!cyContainer) return;

    const cyElements = [
      ...nodes.map(node => ({
        group: 'nodes',
        data: {
          id: node.data.id,
          label: node.data.label || node.data.id,
          type: node.data.type || 'property',
          color: node.data.color || '#888888',
        }
      })),
      ...edges.map(edge => ({
        group: 'edges',
        data: {
          id: edge.data.id || `${edge.data.source}_${edge.data.target}`,
          source: edge.data.source,
          target: edge.data.target,
          type: edge.data.type || 'related',
        }
      }))
    ];

    if (window.cyInstance) {
      window.cyInstance.destroy();
    }

    const cy = cytoscape({
      container: cyContainer,
      elements: cyElements,
      style: [
        {
          selector: 'node',
          style: {
            'label': '',
            'background-color': 'data(color)',
            'color': '#fff',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '11px',
            'font-weight': '600',
            'text-wrap': 'wrap',
            'text-max-width': node => Math.min(node.data('label').length * 6.5, 160) + 'px',
            'width': node => {
              const len = (node.data('label') || '').length;
              return Math.max(36, len * 7.5) + 'px';
            },
            'height': node => {
              const lines = Math.ceil((node.data('label') || '').length / 22);
              return Math.max(36, 14 + lines * 30) + 'px';
            },
            'padding': '6px',
            'border-width': 2,
            'border-color': '#fff',
            'border-opacity': 0.25,
            'text-outline-color': '#111',
            'text-outline-width': 2,
            'overlay-padding': '6px',
            'transition-property': 'background-color, border-width, text-outline-width',
            'transition-duration': '0.15s',
          }
        },
        {
          selector: 'node:hover',
          style: {
            'label': 'data(label)',
            'border-width': 3,
            'border-opacity': 0.6,
          }
        },
        {
          selector: 'node:selected',
          style: {
            'label': 'data(label)',
            'border-width': 3,
            'border-color': '#fff',
            'border-opacity': 1,
          }
        },
        {
          selector: 'node[type="hypothesis"]',
          style: {
            'shape': 'round-rectangle',
            'font-size': '10px',
            'background-color': '#e03e3e',
            'border-color': '#ff6b6b',
          }
        },
        {
          selector: 'node[type="material"]',
          style: {
            'background-color': '#0077bc',
            'shape': 'ellipse',
          }
        },
        {
          selector: 'node[type="element"]',
          style: {
            'background-color': '#10b981',
            'shape': 'ellipse',
          }
        },
        {
          selector: 'node[type="process"]',
          style: {
            'background-color': '#f59e0b',
            'shape': 'round-triangle',
          }
        },
        {
          selector: 'node[type="property"]',
          style: {
            'background-color': '#8b5cf6',
            'shape': 'diamond',
          }
        },
        {
          selector: 'node[type="equipment"]',
          style: {
            'background-color': '#ec4899',
            'shape': 'rectangle',
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 1.2,
            'line-color': '#666',
            'target-arrow-color': '#666',
            'target-arrow-shape': 'triangle',
            'target-arrow-scale': 0.6,
            'curve-style': 'bezier',
            'arrow-scale': 0.7,
            'opacity': 0.06,
            'transition-property': 'opacity, line-color, width',
            'transition-duration': '0.2s',
          }
        },
        {
          selector: 'edge[type="proposes"]',
          style: {
            'line-color': '#e03e3e',
            'target-arrow-color': '#e03e3e',
            'line-style': 'dashed',
          }
        },
        {
          selector: 'edge:hover',
          style: {
            'opacity': 0.5,
            'width': 2,
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 3,
            'border-color': '#fff',
            'border-opacity': 1,
          }
        },
        {
          selector: 'node:selected edge',
          style: {
            'opacity': 0.35,
            'width': 1.5,
          }
        },
        {
          selector: 'edge:selected',
          style: {
            'opacity': 0.6,
            'width': 2.5,
            'line-style': 'solid',
          }
        }
      ],
      layout: {
        name: 'cose',
        animate: true,
        animationDuration: 800,
        animationEasing: 'ease-out-cubic',
        randomize: true,
        fit: true,
        padding: 60,
        nodeRepulsion: node => 65000,
        idealEdgeLength: node => 160,
        edgeElasticity: node => 100,
        nestingFactor: 5,
        gravity: 80,
        numIter: 2500,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0,
        nodeOverlap: 25,
      },
      userZoomingEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: true,
      minZoom: 0.2,
      maxZoom: 3,
      wheelSensitivity: 0.3,
    });

    cy.on('tap', 'node', function(evt) {
      const node = evt.target;
      const connectedEdges = node.connectedEdges();
      const connectedNeighbors = node.neighborhood('node');

      cy.elements().forEach(el => {
        if (el === node) return;
        el.style('opacity', 0.18);
      });

      connectedNeighbors.forEach(sibling => {
        sibling.style('opacity', 1);
        sibling.style('label', sibling.data('label'));
      });
      connectedEdges.forEach(edge => {
        edge.style('opacity', 0.55);
        edge.style('width', 1.5);
      });

      node.style('label', node.data('label'));
      node.style('opacity', 1);

      const label = node.data('label');
      const type = node.data('type');
      console.log(`[HF] Node tapped: ${label} (${type})`);
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        cy.elements().forEach(el => el.removeStyle());
      }
    });

    window.cyInstance = cy;
  }

  function setupEmbeddingSelector() {
    const select = document.getElementById('embedding-mode');
    if (!select) return;

    apiRequest('/settings/embedding').then(res => {
      select.value = res.provider || 'auto';
    }).catch(() => {});

    select.addEventListener('change', async () => {
      try {
        await apiRequest('/settings/embedding', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ provider: select.value }),
        });
        showSuccess(`Embeddings: ${select.options[select.selectedIndex].text}`);
      } catch (e) {
        showError('Не удалось сменить embeddings провайдер');
      }
    });
  }

  const isHealthy = await checkHealth();
  if (isHealthy) {
    setupNavigation();
    setupDropzone();
    setupUpload();
    setupGeneration();
    setupEmbeddingSelector();
    setupDocsFilter();
    setupExportList();
    setupDocActions();
    setupGraph();
    navigateFromInitialHash();
    loadDashboardStats();
    console.log('[HF] Application initialized successfully');
  } else {
    console.error('[HF] Cannot initialize - backend not available');
  }
}

init();
