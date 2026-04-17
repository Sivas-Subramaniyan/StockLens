/* ─────────────────────────────────────────────────────────
   StockLens — AI Research Terminal  |  Frontend JS
   ───────────────────────────────────────────────────────── */

const BASE = window.location.origin === 'file://' ? 'http://localhost:8000' : window.location.origin;

let allCompanies   = [];
let cardView       = false;
let currentJobId   = null;
let pollTimer      = null;
let currentCompany = null;
let tokenPollTimer = null;
let popoverOpen    = false;

// ── Session key helpers ───────────────────────────────────
// sessionStorage is scoped to the browser tab — cleared automatically
// when the tab is closed, ensuring every new visitor starts fresh.
const SESSION_KEY_NAME = 'sl_gemini_key';

function getSessionKey()         { return sessionStorage.getItem(SESSION_KEY_NAME) || ''; }
function setSessionKey(key)      { sessionStorage.setItem(SESSION_KEY_NAME, key); }
function clearSessionKey()       { sessionStorage.removeItem(SESSION_KEY_NAME); }
function sessionKeyHeaders()     {
  const k = getSessionKey();
  return k ? { 'X-Gemini-Key': k } : {};
}

// ── Init ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initTabs();
  loadCompanies();
  loadAlgorithm();
  loadCompaniesForSelect();
  loadReports();
  fetchSettings();
  fetchTokenStats();

  // Start polling token stats every 15 s
  tokenPollTimer = setInterval(fetchTokenStats, 15_000);

  // Companies controls
  document.getElementById('topNSelect').addEventListener('change', loadCompanies);
  document.getElementById('refreshBtn').addEventListener('click', loadCompanies);
  document.getElementById('toggleViewBtn').addEventListener('click', toggleView);

  // Research
  document.getElementById('startBtn').addEventListener('click', startResearch);
  document.getElementById('viewReportBtn').addEventListener('click', viewReport);
  document.getElementById('downloadReportBtn').addEventListener('click', downloadReport);
  document.getElementById('closeReportBtn').addEventListener('click', () => {
    document.getElementById('reportViewer').style.display = 'none';
  });

  // Share
  document.getElementById('shareWABtn').addEventListener('click', shareOnWhatsApp);
  document.getElementById('copyLinkBtn').addEventListener('click', copyPdfLink);

  // Reports tab
  document.getElementById('refreshReportsBtn').addEventListener('click', loadReports);

  // Modal
  document.getElementById('modalClose').addEventListener('click', closeModal);
  document.getElementById('reportModal').addEventListener('click', e => {
    if (e.target === document.getElementById('reportModal')) closeModal();
  });

  // Theme
  document.getElementById('themeToggle').addEventListener('click', toggleTheme);

  // Company select sync
  document.getElementById('companySelect').addEventListener('change', () => {
    document.getElementById('companyInput').value = document.getElementById('companySelect').value;
  });
  document.getElementById('companyInput').addEventListener('input', () => {
    document.getElementById('companySelect').value = '';
  });

  // Settings drawer
  document.getElementById('settingsBtn').addEventListener('click', openDrawer);
  document.getElementById('keyStatusPill').addEventListener('click', openDrawer);
  document.getElementById('drawerClose').addEventListener('click', closeDrawer);
  document.getElementById('drawerOverlay').addEventListener('click', closeDrawer);
  document.getElementById('saveKeyBtn').addEventListener('click', saveApiKey);
  document.getElementById('keyToggleVis').addEventListener('click', toggleKeyVisibility);
  document.getElementById('resetTokensBtn').addEventListener('click', resetTokens);
  document.getElementById('saveRiskBtn').addEventListener('click', saveRiskProfile);

  // Inline API key banner (Research tab)
  document.getElementById('bannerSaveBtn').addEventListener('click', saveBannerKey);
  document.getElementById('bannerKeyInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') saveBannerKey();
  });

  // Token widget popover
  document.getElementById('tokenWidget').addEventListener('click', toggleTokenPopover);
  document.addEventListener('click', e => {
    const widget = document.getElementById('tokenWidget');
    const popover = document.getElementById('tokenPopover');
    if (popoverOpen && !widget.contains(e.target) && !popover.contains(e.target)) {
      hideTokenPopover();
    }
  });

  // Enter key in API key input
  document.getElementById('keyInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') saveApiKey();
  });
});

// ── Theme ─────────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  document.getElementById('themeToggle').textContent = saved === 'dark' ? '☀' : '🌙';
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  document.getElementById('themeToggle').textContent = next === 'dark' ? '☀' : '🌙';
}

// ── Tabs ──────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    });
  });
}
function switchToTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
}

// ── Settings — fetch current ──────────────────────────────
async function fetchSettings() {
  try {
    const res  = await fetch(`${BASE}/settings`);
    const data = await res.json();
    applySettings(data);
  } catch {
    // Server not up yet — still apply session state from sessionStorage
    applySettings({});
  }
}

function applySettings(data) {
  const dot   = document.getElementById('ksDot');
  const label = document.getElementById('ksLabel');
  const kcVal = document.getElementById('kcValue');
  const kcSt  = document.getElementById('kcStatus');
  const badge = document.getElementById('badgeModel');

  if (data.model) badge.textContent = data.model.replace('models/', '');

  // Key status is always derived from this browser session,
  // not from the server — so different visitors are fully independent.
  const sessionKey = getSessionKey();
  const isActive   = !!sessionKey;

  kcVal.textContent = isActive ? (data.key_label || maskKey(sessionKey)) : '—';

  if (isActive) {
    dot.className     = 'ks-dot active';
    label.textContent = 'Active';
    kcSt.textContent  = '✓ Active';
    kcSt.className    = 'kc-status active';
  } else if (data.key_status === 'invalid') {
    dot.className     = 'ks-dot invalid';
    label.textContent = 'Invalid key';
    kcSt.textContent  = '✗ Invalid';
    kcSt.className    = 'kc-status invalid';
    clearSessionKey();
  } else {
    dot.className     = 'ks-dot checking';
    label.textContent = 'Enter key';
    kcSt.textContent  = '';
    kcSt.className    = 'kc-status';
  }

  // Show / hide the inline API key banner in the Research tab
  const banner = document.getElementById('apiKeyBanner');
  if (banner) banner.style.display = isActive ? 'none' : 'flex';
}

function maskKey(key) {
  if (!key || key.length < 8) return '***';
  return key.slice(0, 4) + '…' + key.slice(-4);
}

// ── Settings drawer open/close ────────────────────────────
function openDrawer() {
  document.getElementById('settingsDrawer').classList.add('open');
  document.getElementById('drawerOverlay').classList.add('open');
  fetchSettings();
  fetchTokenStats();
  loadRiskProfile();
}
function closeDrawer() {
  document.getElementById('settingsDrawer').classList.remove('open');
  document.getElementById('drawerOverlay').classList.remove('open');
}

// ── Key visibility toggle ─────────────────────────────────
function toggleKeyVisibility() {
  const input = document.getElementById('keyInput');
  const btn   = document.getElementById('keyToggleVis');
  if (input.type === 'password') {
    input.type    = 'text';
    btn.textContent = '🙈';
  } else {
    input.type    = 'password';
    btn.textContent = '👁';
  }
}

// ── Save / verify API key ─────────────────────────────────
async function saveApiKey() {
  const newKey = document.getElementById('keyInput').value.trim();
  if (!newKey) {
    showKeyFeedback('Please enter an API key.', 'error');
    return;
  }

  const btn     = document.getElementById('saveKeyBtn');
  const btnText = document.getElementById('saveKeyBtnText');
  const ri      = document.getElementById('restartIndicator');
  const riMsg   = document.getElementById('riMsg');

  btn.disabled  = true;
  btnText.textContent = '⏳ Verifying…';
  hideKeyFeedback();
  ri.style.display = 'none';

  try {
    const res  = await fetch(`${BASE}/settings/api-key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: newKey }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'Verification failed');
    }

    // Show restart animation
    ri.style.display = 'flex';
    riMsg.textContent = 'Reinitialising system…';

    await delay(1200);
    riMsg.textContent = 'Reloading companies…';

    // Refresh everything with the new key
    await loadCompanies();
    await loadCompaniesForSelect();
    await fetchSettings();
    fetchTokenStats();

    await delay(600);
    ri.style.display = 'none';
    riMsg.textContent = 'System ready.';

    // Store in this browser session only (cleared on tab close)
    setSessionKey(newKey);

    // Clear the input
    document.getElementById('keyInput').value = '';
    document.getElementById('keyInput').type  = 'password';
    document.getElementById('keyToggleVis').textContent = '👁';

    showKeyFeedback(`✓ ${data.message}`, 'success');

    // Update header key status
    applySettings(data);

  } catch (err) {
    ri.style.display = 'none';
    const msg = err.message || 'Unknown error';
    showKeyFeedback(`✗ ${msg}`, 'error');
    // Mark key as invalid in header
    document.getElementById('ksDot').className   = 'ks-dot invalid';
    document.getElementById('ksLabel').textContent = 'Invalid';
  } finally {
    btn.disabled  = false;
    btnText.textContent = '✓ Verify & Apply';
  }
}

function showKeyFeedback(msg, type) {
  const el = document.getElementById('keyFeedback');
  el.textContent = msg;
  el.className   = `key-feedback ${type}`;
  el.style.display = 'block';
}
function hideKeyFeedback() {
  document.getElementById('keyFeedback').style.display = 'none';
}

// ── Token counter ─────────────────────────────────────────
async function fetchTokenStats() {
  try {
    const res  = await fetch(`${BASE}/stats/tokens`);
    const data = await res.json();
    renderTokenStats(data);
    // Also sync key status
    applySettings({ key_label: data.key_label, key_status: data.key_status });
  } catch { /* silent */ }
}

function renderTokenStats(data) {
  const total    = data.total_tokens    || 0;
  const input    = data.input_tokens    || 0;
  const output   = data.output_tokens   || 0;
  const calls    = data.api_calls       || 0;
  const sessions = data.sessions        || 0;
  const lastUpd  = data.last_updated;

  const fmt = n => n.toLocaleString();

  // Header widget
  const countEl = document.getElementById('twCount');
  if (parseInt(countEl.textContent.replace(/,/g,'')) !== total) {
    countEl.textContent = fmt(total);
    countEl.classList.add('updated');
    setTimeout(() => countEl.classList.remove('updated'), 1500);
  }

  // Popover
  document.getElementById('tpInput').textContent    = fmt(input);
  document.getElementById('tpOutput').textContent   = fmt(output);
  document.getElementById('tpTotal').textContent    = fmt(total);
  document.getElementById('tpCalls').textContent    = calls;
  document.getElementById('tpSessions').textContent = sessions;
  document.getElementById('tpLast').textContent     = lastUpd
    ? new Date(lastUpd).toLocaleTimeString() : '—';

  // Drawer breakdown
  document.getElementById('tbInput').textContent    = fmt(input);
  document.getElementById('tbOutput').textContent   = fmt(output);
  document.getElementById('tbTotal').textContent    = fmt(total);
  document.getElementById('tbCalls').textContent    = calls;
  document.getElementById('tbSessions').textContent = sessions;
  document.getElementById('tokenBarUsed').textContent = fmt(total);

  // Token bar (soft limit ~250K/day)
  const SOFT_LIMIT = 250_000;
  const pct   = Math.min((total / SOFT_LIMIT) * 100, 100).toFixed(1);
  const fill  = document.getElementById('tokenBarFill');
  fill.style.width = `${pct}%`;
  fill.className   = `token-bar-fill${pct > 80 ? ' warn' : ''}`;
}

function toggleTokenPopover() {
  popoverOpen ? hideTokenPopover() : showTokenPopover();
}
function showTokenPopover() {
  document.getElementById('tokenPopover').style.display = 'block';
  popoverOpen = true;
}
function hideTokenPopover() {
  document.getElementById('tokenPopover').style.display = 'none';
  popoverOpen = false;
}

async function resetTokens() {
  // Local reset only (server counter keeps running across server restarts)
  renderTokenStats({ total_tokens: 0, input_tokens: 0, output_tokens: 0,
                     api_calls: 0, sessions: 0, last_updated: null });
}

// ── Companies ─────────────────────────────────────────────
async function loadCompanies() {
  const topN = document.getElementById('topNSelect').value;
  const url  = topN ? `${BASE}/companies?top_n=${topN}` : `${BASE}/companies`;
  showSkeletons(true);
  document.getElementById('companiesContainer').innerHTML = '';
  try {
    const res  = await fetch(url);
    const data = await res.json();
    allCompanies = data;
    renderCompanies(data);
  } catch (err) {
    document.getElementById('companiesContainer').innerHTML =
      `<div class="error-msg">Failed to load companies: ${err.message}</div>`;
  } finally {
    showSkeletons(false);
  }
}
function showSkeletons(show) {
  document.getElementById('companiesLoading').style.display = show ? 'flex' : 'none';
}
function renderCompanies(data) {
  const container = document.getElementById('companiesContainer');
  if (!data.length) {
    container.innerHTML = `<div class="empty-state"><div class="es-icon">🏢</div><h3>No companies found</h3></div>`;
    return;
  }
  container.innerHTML = cardView ? buildCards(data) : buildTable(data);
}

function buildTable(data) {
  const maxScore = Math.max(...data.map(c => parseFloat(c.investment_score) || 0));
  let html = `<div class="table-wrap"><table>
    <thead><tr>
      <th>Rank</th><th>Company</th><th>Mkt Cap (Cr)</th>
      <th>P/E</th><th>ROCE %</th><th>Score</th><th></th>
    </tr></thead><tbody>`;
  data.forEach(c => {
    const score = parseFloat(c.investment_score) || 0;
    const pct   = maxScore > 0 ? (score / maxScore * 100).toFixed(1) : 0;
    const rClass = c.rank === 1 ? 'rank-1' : c.rank === 2 ? 'rank-2' : c.rank === 3 ? 'rank-3' : 'rank-other';
    html += `<tr>
      <td><span class="rank-badge ${rClass}">${c.rank}</span></td>
      <td><span class="company-name">${esc(c.name)}</span></td>
      <td><span class="metric-val">${esc(c.market_cap)}</span></td>
      <td><span class="metric-val">${esc(c.pe_ratio)}</span></td>
      <td><span class="metric-val">${esc(c.roce)}</span></td>
      <td class="score-cell">
        <div class="score-bar-wrap">
          <div class="score-bar"><div class="score-bar-fill" style="width:${pct}%"></div></div>
          <span class="score-num">${score.toFixed(3)}</span>
        </div>
      </td>
      <td><button class="btn btn-neutral btn-sm" onclick="selectForResearch('${esc(c.name)}',${c.rank})">Analyse</button></td>
    </tr>`;
  });
  html += `</tbody></table></div>`;
  return html;
}

function buildCards(data) {
  const maxScore = Math.max(...data.map(c => parseFloat(c.investment_score) || 0));
  let html = '<div class="cards-grid">';
  data.forEach(c => {
    const score = parseFloat(c.investment_score) || 0;
    const pct   = maxScore > 0 ? (score / maxScore * 100).toFixed(1) : 0;
    const rClass = c.rank === 1 ? 'rank-1' : c.rank === 2 ? 'rank-2' : c.rank === 3 ? 'rank-3' : 'rank-other';
    html += `<div class="company-card">
      <div class="cc-header">
        <span class="cc-name">${esc(c.name)}</span>
        <span class="rank-badge ${rClass}">${c.rank}</span>
      </div>
      <div class="cc-metrics">
        <div class="cc-metric"><label>Mkt Cap</label><div class="val">${esc(c.market_cap)}</div></div>
        <div class="cc-metric"><label>P/E</label><div class="val">${esc(c.pe_ratio)}</div></div>
        <div class="cc-metric"><label>ROCE %</label><div class="val">${esc(c.roce)}</div></div>
      </div>
      <div class="cc-score">
        <label>Score <span>${score.toFixed(4)}</span></label>
        <div class="cc-score-bar"><div class="cc-score-fill" style="width:${pct}%"></div></div>
      </div>
      <button class="btn btn-neutral btn-sm" style="width:100%;justify-content:center"
        onclick="selectForResearch('${esc(c.name)}',${c.rank})">⚗ Analyse</button>
    </div>`;
  });
  html += '</div>';
  return html;
}

function toggleView() {
  cardView = !cardView;
  document.getElementById('toggleViewBtn').textContent = cardView ? '☰ Table' : '⊞ Cards';
  renderCompanies(allCompanies);
}
function selectForResearch(name) {
  switchToTab('research');
  document.getElementById('companySelect').value = name;
  document.getElementById('companyInput').value  = name;
  document.getElementById('progressPanel').style.display = 'none';
  document.getElementById('resultsPanel').style.display  = 'none';
}

// ── Algorithm ─────────────────────────────────────────────
async function loadAlgorithm() {
  try {
    const res  = await fetch(`${BASE}/scoring-algorithm`);
    const data = await res.json();
    renderAlgorithm(data);
  } catch (err) {
    document.getElementById('algorithmContent').innerHTML =
      `<div class="error-msg">Failed to load: ${err.message}</div>`;
  }
}
function renderAlgorithm(data) {
  const maxW = Math.max(...Object.values(data.weights));
  const weightsHtml = Object.entries(data.weights).map(([key, val]) => {
    const pct  = (val * 100).toFixed(0);
    const barW = (val / maxW * 100).toFixed(1);
    return `<div class="weight-card">
      <div class="wc-top"><span class="wc-name">${key.replace(/_/g,' ')}</span><span class="wc-pct">${pct}%</span></div>
      <div class="wc-bar"><div class="wc-bar-fill" style="width:${barW}%"></div></div>
      <div class="wc-desc">${data.metrics[key] || key}</div>
    </div>`;
  }).join('');
  const stepsHtml = data.process.map((step, i) => `
    <div class="process-step">
      <div class="ps-num">${i + 1}</div>
      <div class="ps-text">${step.replace(/^\d+\.\s*/,'')}</div>
    </div>`).join('');
  document.getElementById('algorithmContent').innerHTML = `
    <div class="algo-content">
      <div class="algo-section">
        <h3>Overview</h3>
        <p style="color:var(--text2);font-size:14px;line-height:1.7">${data.description}</p>
      </div>
      <div class="algo-section"><h3>Metric Weights</h3><div class="weights-grid">${weightsHtml}</div></div>
      <div class="algo-section"><h3>Scoring Process</h3><div class="process-steps">${stepsHtml}</div></div>
    </div>`;
}

async function loadCompaniesForSelect() {
  try {
    const res  = await fetch(`${BASE}/companies?top_n=100`);
    const data = await res.json();
    const sel  = document.getElementById('companySelect');
    // Clear existing options except the placeholder
    while (sel.options.length > 1) sel.remove(1);
    data.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.name;
      opt.textContent = `${c.rank}. ${c.name}`;
      sel.appendChild(opt);
    });
  } catch { /* silent */ }
}

// ── Research ──────────────────────────────────────────────
async function startResearch() {
  const fromSelect = document.getElementById('companySelect').value;
  const fromInput  = document.getElementById('companyInput').value.trim();
  const name = fromInput || fromSelect;
  if (!name) { alert('Please select or enter a company name'); return; }

  const company = allCompanies.find(c => c.name === name);
  const rank    = company ? company.rank : null;
  const btn     = document.getElementById('startBtn');
  btn.disabled  = true;
  btn.textContent = '⏳ Starting…';

  try {
    const res = await fetch(`${BASE}/research/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...sessionKeyHeaders() },
      body: JSON.stringify({ company_name: name, company_rank: rank }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start');
    }
    const data = await res.json();
    currentJobId   = data.research_id;
    currentCompany = data.company_name;

    document.getElementById('progressPanel').style.display = 'block';
    document.getElementById('resultsPanel').style.display  = 'none';
    const sp = document.getElementById('sharePanel');
    if (sp) sp.style.display = 'none';
    document.getElementById('progressCompany').textContent = data.company_name;
    document.getElementById('jobId').textContent = `ID: ${data.research_id.slice(0,8)}…`;
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('progressPct').textContent   = '0%';
    resetTimeline();
    startPolling(data.research_id);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    btn.disabled    = false;
    btn.textContent = '▶ Start Research';
  }
}

function startPolling(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => pollStatus(jobId), 3000);
}
async function pollStatus(jobId) {
  try {
    const res    = await fetch(`${BASE}/research/status/${jobId}`);
    const status = await res.json();
    updateProgress(status);
    if (status.status === 'completed') {
      clearInterval(pollTimer);
      loadResults(jobId);
      fetchTokenStats();   // refresh token counter after completion
    } else if (status.status === 'error') {
      clearInterval(pollTimer);
    }
  } catch { /* ignore network hiccup */ }
}
function updateProgress(status) {
  const prog = status.progress;
  const pct  = Math.round((prog.current / prog.total) * 100);
  document.getElementById('progressFill').style.width = `${pct}%`;
  document.getElementById('progressPct').textContent  = `${pct}%`;
  document.getElementById('liveLog').textContent      = `› ${prog.message || ''}`;
  updateTimeline(status);
}

const STEP_ORDER = ['company_selected','research_agent','report_generation','validation','completed'];
function resetTimeline() {
  document.querySelectorAll('.tl-step').forEach(el => el.classList.remove('done','active','error'));
}
function updateTimeline(status) {
  const cur    = status.current_step;
  const curIdx = STEP_ORDER.indexOf(cur);
  STEP_ORDER.forEach((step, idx) => {
    const el = document.querySelector(`.tl-step[data-step="${step}"]`);
    if (!el) return;
    el.classList.remove('done','active','error');
    if (status.status === 'error') {
      if (idx < curIdx)  el.classList.add('done');
      else if (step === cur) el.classList.add('error');
    } else if (cur === 'completed') {
      el.classList.add('done');
    } else {
      if (idx < curIdx)  el.classList.add('done');
      else if (step === cur) el.classList.add('active');
    }
    if (step === cur && status.status !== 'error') {
      const msg = status.progress.message || '';
      if (msg) el.querySelector('.tl-msg').textContent = msg;
    }
    if (status.status === 'error' && step === cur) {
      el.querySelector('.tl-msg').textContent = status.error || 'An error occurred';
    }
  });
}

async function loadResults(jobId) {
  try {
    const res  = await fetch(`${BASE}/research/results/${jobId}`);
    const data = await res.json();
    renderResults(data);
    document.getElementById('resultsPanel').style.display = 'block';
    document.getElementById('resultsPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    console.error('Could not load results:', err);
  }
}
function renderResults(data) {
  const rec  = (data.recommendation || '').toUpperCase();
  const val  = data.validation || {};
  const card = document.getElementById('verdictCard');
  card.className = 'verdict-card' + (rec === 'BUY' ? ' buy' : rec === 'AVOID' ? ' avoid' : '');
  document.getElementById('verdictRec').textContent    = rec || '—';
  document.getElementById('verdictConf').textContent   = val.confidence || data.confidence || '—';
  document.getElementById('verdictReturn').textContent = val.expected_return_3y || '—';
  document.getElementById('verdictProb').textContent   = val.probability_40pct_return || '—';

  const tokensUsed = data.tokens_used;
  if (tokensUsed) {
    document.getElementById('verdictTokens').textContent =
      (tokensUsed.total_tokens || 0).toLocaleString();
  }

  // TLDR card
  const tldrCard = document.getElementById('tldrCard');
  const tldrText = document.getElementById('tldrText');
  if (data.tldr && tldrCard && tldrText) {
    tldrText.textContent = data.tldr;
    tldrCard.style.display = 'block';
    tldrCard.className = 'tldr-card' + (rec === 'BUY' ? ' buy' : rec === 'AVOID' ? ' avoid' : '');
  } else if (tldrCard) {
    tldrCard.style.display = 'none';
  }

  // Wire up PDF download button
  const pdfUrl = `${BASE}/reports/pdf/${encodeURIComponent(currentCompany)}`;
  const pdfBtn = document.getElementById('downloadPdfBtn');
  if (pdfBtn) pdfBtn.href = pdfUrl;

  // Show share panel
  const sharePanel = document.getElementById('sharePanel');
  if (sharePanel) sharePanel.style.display = 'block';

  // Clear previous share hint
  const hint = document.getElementById('shareHint');
  if (hint) { hint.textContent = ''; hint.className = 'share-hint'; }
}

async function viewReport() {
  if (!currentJobId) return;
  try {
    const statusRes = await fetch(`${BASE}/research/status/${currentJobId}`);
    const status    = await statusRes.json();
    const reportRes = await fetch(`${BASE}/reports/${encodeURIComponent(status.company_name)}`);
    const text      = await reportRes.text();
    const viewer = document.getElementById('reportViewer');
    document.getElementById('reportTitle').textContent = `${status.company_name} — Analyst Report`;
    document.getElementById('reportBody').innerHTML    = marked.parse(text);
    viewer.style.display = 'block';
    viewer.scrollIntoView({ behavior: 'smooth' });
  } catch (err) { alert(`Could not load report: ${err.message}`); }
}
async function downloadReport() {
  if (!currentJobId) return;
  try {
    const statusRes = await fetch(`${BASE}/research/status/${currentJobId}`);
    const status    = await statusRes.json();
    const res       = await fetch(`${BASE}/reports/${encodeURIComponent(status.company_name)}`);
    const blob      = await res.blob();
    const url       = URL.createObjectURL(blob);
    const a         = document.createElement('a');
    a.href = url; a.download = `${status.company_name}_Analyst_Report.md`;
    document.body.appendChild(a); a.click();
    URL.revokeObjectURL(url); a.remove();
  } catch (err) { alert(`Download failed: ${err.message}`); }
}

// ── Reports tab ────────────────────────────────────────────
async function loadReports() {
  const container = document.getElementById('reportsContainer');
  container.innerHTML = '<div class="loading-text">Loading reports…</div>';
  try {
    const res  = await fetch(`${BASE}/reports/list`);
    const data = await res.json();
    if (!data.reports || !data.reports.length) {
      container.innerHTML = `<div class="empty-state">
        <div class="es-icon">📄</div><h3>No reports yet</h3>
        <p>Run a research analysis to generate your first report.</p></div>`;
      return;
    }
    let html = '<div class="reports-list">';
    data.reports.forEach(r => {
      html += `<div class="report-item">
        <div>
          <div class="ri-name">${esc(r.company_name)}</div>
          <div class="ri-meta">${r.date} · ${(r.size/1024).toFixed(1)} KB</div>
        </div>
        <div class="ri-actions">
          <button class="btn btn-neutral btn-sm" onclick="openReportModal('${esc(r.company_name)}')">📖 View</button>
          <a class="btn btn-neutral btn-sm" href="${BASE}/reports/${encodeURIComponent(r.company_name)}"
             download="${esc(r.filename)}">⬇ DL</a>
        </div>
      </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
  } catch (err) {
    container.innerHTML = `<div class="error-msg">Failed to load reports: ${err.message}</div>`;
  }
}
async function openReportModal(companyName) {
  try {
    const res  = await fetch(`${BASE}/reports/${encodeURIComponent(companyName)}`);
    const text = await res.text();
    document.getElementById('modalTitle').textContent = `${companyName} — Analyst Report`;
    document.getElementById('modalBody').innerHTML    = marked.parse(text);
    document.getElementById('reportModal').style.display = 'flex';
  } catch (err) { alert(`Could not load: ${err.message}`); }
}
function closeModal() {
  document.getElementById('reportModal').style.display = 'none';
}

// ── Inline API key banner ─────────────────────────────────
async function saveBannerKey() {
  const input    = document.getElementById('bannerKeyInput');
  const btn      = document.getElementById('bannerSaveBtn');
  const feedback = document.getElementById('bannerFeedback');
  const newKey   = input.value.trim();
  if (!newKey) return;

  btn.disabled      = true;
  btn.textContent   = '⏳';
  feedback.style.display = 'none';

  try {
    const res  = await fetch(`${BASE}/settings/api-key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: newKey }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Verification failed');

    // Store in this browser session only (cleared on tab close)
    setSessionKey(newKey);

    // Success — key is now active
    input.value          = '';
    feedback.textContent = '✓ Key activated! You can now start research.';
    feedback.className   = 'akb-feedback success';
    feedback.style.display = 'block';

    // Refresh settings everywhere and hide banner
    applySettings(data);
    await loadCompanies();
    await loadCompaniesForSelect();
    fetchTokenStats();

  } catch (err) {
    feedback.textContent = `✗ ${err.message}`;
    feedback.className   = 'akb-feedback error';
    feedback.style.display = 'block';
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Activate';
  }
}

// ── Share + PDF ───────────────────────────────────────────

function _buildShareMessage() {
  const company = currentCompany || 'this company';
  const rec     = document.getElementById('verdictRec')?.textContent    || 'N/A';
  const conf    = document.getElementById('verdictConf')?.textContent   || 'N/A';
  const ret     = document.getElementById('verdictReturn')?.textContent || 'N/A';
  const pdfUrl  = `${BASE}/reports/pdf/${encodeURIComponent(company)}`;
  const appUrl  = BASE;
  const msg =
    `📊 *${company}* — StockLens AI Report\n` +
    `Verdict: *${rec}* (${conf} confidence)\n` +
    `Expected 3Y Return: ${ret}\n\n` +
    `📄 Full PDF Report:\n${pdfUrl}\n\n` +
    `🔍 Analyse more stocks free:\n${appUrl}`;
  return { msg, pdfUrl };
}

function shareOnWhatsApp() {
  const { msg } = _buildShareMessage();
  // wa.me works on mobile (opens WhatsApp app) and desktop (opens WhatsApp Web)
  window.open(`https://wa.me/?text=${encodeURIComponent(msg)}`, '_blank');
  _setShareHint('Opening WhatsApp…', 'success');
}

async function copyPdfLink() {
  const { pdfUrl } = _buildShareMessage();
  const hint = document.getElementById('shareHint');
  try {
    await navigator.clipboard.writeText(pdfUrl);
    _setShareHint('✓ PDF link copied to clipboard!', 'success');
  } catch {
    // Fallback for browsers without clipboard API
    try {
      const ta = document.createElement('textarea');
      ta.value = pdfUrl;
      ta.style.position = 'fixed';
      ta.style.opacity  = '0';
      document.body.appendChild(ta);
      ta.focus(); ta.select();
      document.execCommand('copy');
      ta.remove();
      _setShareHint('✓ PDF link copied!', 'success');
    } catch {
      _setShareHint(`PDF link: ${pdfUrl}`, '');
    }
  }
}

function _setShareHint(msg, type) {
  const hint = document.getElementById('shareHint');
  if (!hint) return;
  hint.textContent = msg;
  hint.className   = `share-hint${type ? ' ' + type : ''}`;
}

// ── Risk Appetite Barometer ───────────────────────────────

const RR_LABELS = { 1: '60%+ in 3Y', 2: '50%+ in 3Y', 3: '40%+ in 3Y', 4: '32%+ in 3Y', 5: '25%+ in 3Y' };
const GT_LABELS = { 1: 'Zero tolerance', 2: 'Very strict', 3: 'Moderate', 4: 'Flexible', 5: 'Lenient' };
const BM_LABELS = { 1: 'Proven only', 2: 'Conservative', 3: 'Balanced', 4: 'Growth-ready', 5: 'Early-stage OK' };

const APPETITE_CLASS = {
  'Ultra-Conservative': 'ultra-conservative',
  'Conservative':       'conservative',
  'Balanced':           'balanced',
  'Aggressive':         'aggressive',
  'Speculative':        'speculative',
};

function appetiteLabel(rr, gt, bm) {
  const s = (rr + gt + bm) / 3;
  if (s <= 1.5) return 'Ultra-Conservative';
  if (s <= 2.4) return 'Conservative';
  if (s <= 3.5) return 'Balanced';
  if (s <= 4.2) return 'Aggressive';
  return 'Speculative';
}

function onSliderChange() {
  const rr = parseInt(document.getElementById('sliderRR').value);
  const gt = parseInt(document.getElementById('sliderGT').value);
  const bm = parseInt(document.getElementById('sliderBM').value);

  document.getElementById('rrValue').textContent = RR_LABELS[rr] || rr;
  document.getElementById('gtValue').textContent = GT_LABELS[gt] || gt;
  document.getElementById('bmValue').textContent = BM_LABELS[bm] || bm;

  const label = appetiteLabel(rr, gt, bm);
  const pill  = document.getElementById('appetitePill');
  if (pill) {
    pill.textContent  = label;
    pill.className    = 'ag-pill ' + (APPETITE_CLASS[label] || 'balanced');
  }
}

async function loadRiskProfile() {
  try {
    const res  = await fetch(`${BASE}/settings/risk-profile`);
    const data = await res.json();
    document.getElementById('sliderRR').value = data.return_hurdle        || 3;
    document.getElementById('sliderGT').value = data.governance_tolerance || 3;
    document.getElementById('sliderBM').value = data.business_maturity    || 3;
    onSliderChange();  // update labels + gauge
  } catch { /* silent — server may not be up */ }
}

async function saveRiskProfile() {
  const rr  = parseInt(document.getElementById('sliderRR').value);
  const gt  = parseInt(document.getElementById('sliderGT').value);
  const bm  = parseInt(document.getElementById('sliderBM').value);
  const btn = document.getElementById('saveRiskBtn');
  const txt = document.getElementById('saveRiskBtnText');
  const fb  = document.getElementById('riskFeedback');

  btn.disabled   = true;
  txt.textContent = '⏳ Saving…';
  fb.style.display = 'none';

  try {
    const res  = await fetch(`${BASE}/settings/risk-profile`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ return_hurdle: rr, governance_tolerance: gt, business_maturity: bm }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Save failed');

    fb.textContent = `✓ Profile saved — ${data.appetite_label} (${data.return_threshold_pct}%+ return hurdle)`;
    fb.className   = 'key-feedback success';
    fb.style.display = 'block';
  } catch (err) {
    fb.textContent = `✗ ${err.message}`;
    fb.className   = 'key-feedback error';
    fb.style.display = 'block';
  } finally {
    btn.disabled    = false;
    txt.textContent = 'Save Profile';
  }
}

// ── Helpers ───────────────────────────────────────────────
function esc(str) {
  return String(str ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }
