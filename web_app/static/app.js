const state = { datasets: [], runs: [], selectedDataset: 'sample', selectedTrainingDataset: null, selectedRun: null, selectedPredictionRun: null, activeView: 'overview', poller: null };
const API_BASE = (window.TRANSFORMER_API_URL || '').replace(/\/$/, '');
let API_KEY = sessionStorage.getItem('transformer-access-key') || '';
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const percent = (value) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (API_KEY) headers.set('Authorization', `Bearer ${API_KEY}`);
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  let body;
  try { body = await response.json(); } catch { body = {}; }
  if (!response.ok) {
    const detail = body.detail;
    const error = new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return body;
}

let toastTimer;
function toast(message, error = false) {
  const node = $('#toast');
  node.textContent = message;
  node.className = `toast show${error ? ' error' : ''}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.className = 'toast', 4200);
}

function go(view) {
  state.activeView = view;
  $$('.view').forEach((node) => node.classList.toggle('active', node.id === `view-${view}`));
  $$('.nav-item').forEach((node) => node.classList.toggle('active', node.dataset.view === view));
  $('#breadcrumb-current').textContent = view.charAt(0).toUpperCase() + view.slice(1);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function badge(status) {
  return `<span class="status-badge ${esc(status || 'neutral')}">${esc(status || 'Idle')}</span>`;
}

function renderOverview() {
  const completed = state.runs.filter((run) => run.status === 'complete');
  const best = completed.map((run) => run.metrics?.test?.macro?.f1).filter((score) => typeof score === 'number');
  $('#stat-datasets').textContent = state.datasets.length;
  $('#stat-runs').textContent = state.runs.length;
  $('#stat-complete').textContent = completed.length;
  $('#stat-best').textContent = best.length ? percent(Math.max(...best)) : '—';
  const hasRealRun = completed.some((run) => run.model_name !== 'hf-internal-testing/tiny-random-bert' && run.dataset_id !== 'sample');
  $('#stat-best').nextElementSibling.textContent = best.length && !hasRealRun ? 'Demo workflow check' : 'Macro averaged';
  $('#overview-datasets').innerHTML = state.datasets.slice(0, 3).map((dataset) => `
    <div class="compact-item"><span class="compact-icon">▤</span><div class="compact-item-main"><strong>${esc(dataset.name)}</strong><small>${Object.keys(dataset.labels).length} labels · ${dataset.sample ? 'Practice data' : 'Uploaded CSV'}</small></div><span class="compact-meta">${dataset.count} rows</span></div>`).join('') || '<div class="empty-list">Upload a CSV to add your first dataset.</div>';
  $('#overview-runs').innerHTML = state.runs.slice(0, 3).map((run) => `
    <div class="compact-item"><span class="compact-icon run">◉</span><div class="compact-item-main"><strong>${esc(run.dataset_name)}</strong><small>${esc(run.model_name.split('/').pop())} · ${esc(new Date(run.created_at).toLocaleDateString())}</small></div>${badge(run.status)}</div>`).join('') || '<div class="empty-list">Your training runs will appear here. Start with the sample dataset or upload your own CSV.</div>';
}

function renderDatasets() {
  $('#dataset-count').textContent = state.datasets.length;
  $('#dataset-library').innerHTML = state.datasets.map((dataset) => `
    <button class="dataset-row ${dataset.id === state.selectedDataset ? 'selected' : ''}" data-dataset="${esc(dataset.id)}"><span class="compact-icon">▤</span><span><strong>${esc(dataset.name)}</strong><small>${dataset.count} examples · ${Object.keys(dataset.labels).length} labels ${dataset.sample ? '· Sample' : ''}</small></span><span class="arrow">↗</span></button>`).join('');
  const dataset = state.datasets.find((item) => item.id === state.selectedDataset) || state.datasets[0];
  if (!dataset) return;
  state.selectedDataset = dataset.id;
  $('#dataset-preview-title').textContent = dataset.name;
  $('#dataset-preview-count').textContent = `${dataset.count} ROWS`;
  const max = Math.max(...Object.values(dataset.labels));
  $('#label-chart').innerHTML = Object.entries(dataset.labels).map(([label, count], index) => `<div class="label-line"><div class="label-line-top"><span>${esc(label)}</span><strong>${count} · ${Math.round(count / dataset.count * 100)}%</strong></div><div class="bar-track"><div class="bar-fill ${index % 2 ? 'alt' : ''}" style="width:${count / max * 100}%"></div></div></div>`).join('');
  $('#example-rows').innerHTML = dataset.preview.map((row) => `<div class="example-row"><span title="${esc(row.text)}">${esc(row.text)}</span><span class="label-chip">${esc(row.label)}</span></div>`).join('');
  $('#train-dataset').innerHTML = state.datasets.map((item) => `<option value="${esc(item.id)}">${esc(item.name)} (${item.count} rows)</option>`).join('');
  const wanted = state.selectedTrainingDataset || dataset.id;
  $('#train-dataset').value = state.datasets.some((item) => item.id === wanted) ? wanted : dataset.id;
}

function renderRunOptions() {
  const previousEval = state.selectedRun || $('#evaluation-run').value;
  const previousPredict = $('#prediction-run').value;
  $('#evaluation-run').innerHTML = state.runs.length ? state.runs.map((run) => `<option value="${esc(run.id)}">${esc(run.dataset_name)} · ${esc(new Date(run.created_at).toLocaleString())} · ${esc(run.status)}</option>`).join('') : '<option value="">No experiments yet</option>';
  state.selectedRun = state.runs.some((run) => run.id === previousEval) ? previousEval : state.runs[0]?.id || null;
  $('#evaluation-run').value = state.selectedRun || '';
  const complete = state.runs.filter((run) => run.status === 'complete');
  $('#prediction-run').innerHTML = complete.length ? complete.map((run) => `<option value="${esc(run.id)}">${esc(run.dataset_name)} · ${esc(run.model_name.split('/').pop())}</option>`).join('') : '<option value="">Train a model first</option>';
  const wantedPrediction = state.selectedPredictionRun || previousPredict;
  if (complete.some((run) => run.id === wantedPrediction)) $('#prediction-run').value = wantedPrediction;
  renderEvaluation();
}

function renderLive() {
  const active = state.runs.find((run) => ['running', 'queued'].includes(run.status));
  const latest = active || state.runs[0];
  $('#live-status').className = `status-badge ${latest?.status || 'neutral'}`;
  $('#live-status').textContent = latest?.status || 'Idle';
  if (!latest) { $('#live-content').innerHTML = 'Start a run to see progress here.'; return; }
  const progress = Math.round((latest.epoch || 0) / latest.epochs * 100);
  $('#live-content').innerHTML = `<div class="progress-details"><strong>${esc(latest.dataset_name)}</strong><p>${esc(latest.model_name.split('/').pop())}${latest.status === 'failed' ? ` · ${esc(latest.error || 'Training failed')}` : ''}</p><div class="progress-track"><div class="progress-fill" style="width:${progress}%"></div></div><small>Epoch ${latest.epoch || 0} of ${latest.epochs}${latest.latest_validation ? ` · Validation F1 ${percent(latest.latest_validation.macro.f1)}` : ''}</small></div>`;
  $('#train-submit').disabled = !!active;
  $('#train-submit').textContent = active ? 'Training in progress…' : 'Start training →';
}

function renderEvaluation() {
  const run = state.runs.find((item) => item.id === state.selectedRun);
  $('#evaluation-status').className = `status-badge ${run?.status || 'neutral'}`;
  $('#evaluation-status').textContent = run?.status || 'No run selected';
  const metrics = run?.metrics?.test;
  $('#evaluation-empty').classList.toggle('hidden', !!metrics);
  $('#evaluation-results').classList.toggle('hidden', !metrics);
  $('#download-metrics').classList.toggle('hidden', !metrics);
  if (!metrics) {
    $('#evaluation-empty h3').textContent = run?.status === 'failed' ? 'Training failed' : run ? 'Experiment in progress' : 'No completed experiments yet';
    $('#evaluation-empty p').textContent = run?.error || (run ? 'Results appear here once training and held-out testing finish.' : 'Train your first model to see precision, recall, F1, and the confusion matrix here.');
    return;
  }
  $('#download-metrics').href = `${API_BASE}/api/runs/${run.id}/metrics`;
  $('#metric-accuracy').textContent = percent(metrics.accuracy);
  $('#metric-precision').textContent = percent(metrics.macro.precision);
  $('#metric-recall').textContent = percent(metrics.macro.recall);
  $('#metric-f1').textContent = percent(metrics.macro.f1);
  $('#class-table').innerHTML = Object.entries(metrics.per_class).map(([label, row]) => `<tr><td>${esc(label)}</td><td>${percent(row.precision)}</td><td>${percent(row.recall)}</td><td>${percent(row.f1)}</td><td>${row.support}</td></tr>`).join('');
  const count = metrics.labels.length;
  let cells = '<div></div>' + metrics.labels.map((name) => `<div class="matrix-cell heading">${esc(name)}</div>`).join('');
  metrics.confusion_matrix.forEach((row, index) => { cells += `<div class="matrix-cell heading">${esc(metrics.labels[index])}</div>` + row.map((value) => `<div class="matrix-cell value" style="opacity:${0.5 + value / Math.max(1, ...row) * 0.5}">${value}</div>`).join(''); });
  $('#confusion-matrix').style.gridTemplateColumns = `repeat(${count + 1}, minmax(0,1fr))`;
  $('#confusion-matrix').innerHTML = cells;
  $('#best-epoch').textContent = `BEST EPOCH ${run.metrics.best_epoch}`;
  const history = run.history || [];
  $('#history-chart').innerHTML = history.map((entry) => `<div class="history-column"><strong>${percent(entry.validation.macro.f1)}</strong><div class="history-bar-wrap"><div class="history-bar" style="height:${Math.max(4, entry.validation.macro.f1 * 100)}%"></div></div><small>Epoch ${entry.epoch}</small></div>`).join('');
}

async function refresh() {
  try {
    const overview = await api('/api/overview');
    state.datasets = overview.datasets;
    state.runs = overview.runs;
    renderOverview(); renderDatasets(); renderRunOptions(); renderLive();
    $('#access-overlay').classList.add('hidden');
    $('#system-status').textContent = 'System ready';
    $('#system-pulse').style.background = '';
    return true;
  } catch (error) {
    $('#system-status').textContent = error.status === 401 ? 'Access key required' : 'Server unavailable';
    $('#system-pulse').style.background = '#ed847d';
    if (error.status === 401) $('#access-overlay').classList.remove('hidden');
    else toast(error.message, true);
    return false;
  }
}

async function downloadMetrics(event) {
  event.preventDefault();
  const run = state.runs.find((item) => item.id === state.selectedRun);
  if (!run) return;
  try {
    const headers = API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {};
    const response = await fetch(`${API_BASE}/api/runs/${run.id}/metrics`, { headers });
    if (!response.ok) throw new Error('Could not download metrics');
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url; link.download = `${run.id}-metrics.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { toast(error.message, true); }
}

async function upload(file) {
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  try {
    const result = await api('/api/datasets', { method: 'POST', body: form });
    state.selectedDataset = result.id;
    state.selectedTrainingDataset = result.id;
    await refresh();
    go('datasets');
    toast(`${result.name} is ready to train`);
  } catch (error) { toast(error.message, true); }
}

async function startTraining(event) {
  event.preventDefault();
  const payload = {
    dataset_id: $('#train-dataset').value, model_name: $('#train-model').value,
    epochs: Number($('#train-epochs').value), batch_size: Number($('#train-batch').value),
    max_length: Number($('#train-length').value), learning_rate: Number($('#train-rate').value)
  };
  try {
    const run = await api('/api/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    state.selectedRun = run.id;
    state.selectedPredictionRun = run.id;
    await refresh();
    toast('Training started. Progress will update automatically.');
  } catch (error) { toast(error.message, true); }
}

async function predict() {
  const runId = $('#prediction-run').value;
  const text = $('#prediction-text').value.trim();
  if (!runId) return toast('Train a model first', true);
  if (!text) return toast('Enter some text to classify', true);
  const button = $('#predict-submit');
  button.disabled = true; button.textContent = 'Analyzing…';
  try {
    const result = await api(`/api/runs/${runId}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
    $('#prediction-output').className = 'prediction-result';
    $('#prediction-output').innerHTML = `<small>PREDICTED LABEL</small><strong>${esc(result.label)}</strong><p>${percent(result.confidence)} model probability</p>` + Object.entries(result.probabilities).map(([label, value]) => `<div class="probability-row"><div><span>${esc(label)}</span><strong>${percent(value)}</strong></div><div class="bar-track"><div class="bar-fill" style="width:${value * 100}%"></div></div></div>`).join('');
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; button.innerHTML = 'Analyze text <span>→</span>'; }
}

document.addEventListener('click', (event) => {
  const destination = event.target.closest('[data-goto]');
  if (destination) go(destination.dataset.goto);
  const nav = event.target.closest('[data-view]');
  if (nav) go(nav.dataset.view);
  const dataset = event.target.closest('[data-dataset]');
  if (dataset) { state.selectedDataset = dataset.dataset.dataset; state.selectedTrainingDataset = state.selectedDataset; renderDatasets(); }
  const example = event.target.closest('[data-example]');
  if (example) { $('#prediction-text').value = example.dataset.example; $('#char-count').textContent = `${$('#prediction-text').value.length} characters`; }
});
$('#dataset-file').addEventListener('change', (event) => upload(event.target.files[0]));
const dropZone = $('#drop-zone');
['dragenter', 'dragover'].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', (event) => upload(event.dataTransfer.files[0]));
$('#train-form').addEventListener('submit', startTraining);
$('#train-dataset').addEventListener('change', (event) => { state.selectedTrainingDataset = event.target.value; });
$('#evaluation-run').addEventListener('change', (event) => { state.selectedRun = event.target.value; renderEvaluation(); });
$('#prediction-run').addEventListener('change', (event) => { state.selectedPredictionRun = event.target.value; });
$('#prediction-text').addEventListener('input', (event) => $('#char-count').textContent = `${event.target.value.length} characters`);
$('#predict-submit').addEventListener('click', predict);
$('#download-metrics').addEventListener('click', downloadMetrics);
$('#access-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  API_KEY = $('#access-key').value;
  sessionStorage.setItem('transformer-access-key', API_KEY);
  const unlocked = await refresh();
  $('#access-error').textContent = unlocked ? '' : 'That key did not unlock the workspace.';
});
refresh();
state.poller = setInterval(() => { if ($('#access-overlay').classList.contains('hidden')) refresh(); }, 3000);
