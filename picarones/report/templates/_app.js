'use strict';

// ── Palette couleurs par moteur ──────────────────────────────────
const PALETTE = [
  '#2563eb','#dc2626','#16a34a','#ca8a04','#7c3aed',
  '#0891b2','#c2410c','#0f766e','#9333ea','#b45309',
];
function engineColor(idx) { return PALETTE[idx % PALETTE.length]; }

// ── Navigation ──────────────────────────────────────────────────
let currentView = 'ranking';
function _switchView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + name).classList.add('active');
  // Activer le bon onglet nav
  const tabMap = {ranking:'classement',gallery:'galerie',document:'document',characters:'caract',analyses:'analyses'};
  const prefix = tabMap[name] || name;
  document.querySelectorAll('.tab-btn').forEach(b => {
    if (b.textContent.toLowerCase().startsWith(prefix.toLowerCase())) b.classList.add('active');
  });
  currentView = name;
  if (name === 'analyses' && !chartsBuilt) buildCharts();
  if (name === 'characters' && !charViewBuilt) initCharView();
}
function showView(name) {
  _switchView(name);
  updateURL(name);
  // Sprint A6 — re-attache les boutons d'a11y aux nouveaux charts
  // qui ont été instanciés paresseusement au switch de vue.
  if (typeof attachChartA11y === 'function') {
    setTimeout(attachChartA11y, 50);
  }
}

// ── Formatage ───────────────────────────────────────────────────
function pct(v, d=2) {
  if (v === null || v === undefined) return '—';
  return (v * 100).toFixed(d) + ' %';
}
function cerColor(v) {
  if (v < 0.05) return '#16a34a';
  if (v < 0.15) return '#ca8a04';
  if (v < 0.30) return '#ea580c';
  return '#dc2626';
}
function cerBg(v) {
  if (v < 0.05) return '#dcfce7';
  if (v < 0.15) return '#fef9c3';
  if (v < 0.30) return '#ffedd5';
  return '#fee2e2';
}
// Sprint 30 — accessibilité WCAG : un tier non-couleur permet aux
// daltoniens et aux lecteurs d'écran de distinguer les paliers.
// Mappé en CSS sur des patterns visuels (icône + bordure) en plus
// de la couleur. ``aria-label`` complète pour les lecteurs d'écran.
function cerTier(v) {
  if (v < 0.05) return 'excellent';
  if (v < 0.15) return 'acceptable';
  if (v < 0.30) return 'mediocre';
  return 'critical';
}
function cerTierIcon(tier) {
  // Caractères unicode lisibles indépendamment de la couleur.
  return {
    excellent:  '●',  // disque plein
    acceptable: '◐',  // demi-disque
    mediocre:   '◑',  // demi-disque inverse
    critical:   '○',  // cercle vide
  }[tier] || '';
}
function cerTierLabel(tier) {
  return {
    excellent:  'CER excellent',
    acceptable: 'CER acceptable',
    mediocre:   'CER médiocre',
    critical:   'CER critique',
  }[tier] || 'CER';
}
function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Diff renderer ──────────────────────────────────────────────
function renderDiff(ops) {
  if (!ops || !ops.length) return '<em style="color:var(--text-muted)">— aucune sortie —</em>';
  return ops.map(op => {
    if (op.op === 'equal')
      return '<span class="d-eq">' + esc(op.text) + '</span>';
    if (op.op === 'insert')
      return '<span class="d-ins" title="Insertion OCR">' + esc(op.text) + '</span>';
    if (op.op === 'delete')
      return '<span class="d-del" title="Suppression (présent GT)">' + esc(op.text) + '</span>';
    if (op.op === 'replace')
      return '<span class="d-rep-old" title="Remplacement">' + esc(op.old) + '</span>'
           + '<span class="d-rep-new">' + esc(op.new) + '</span>';
    return '';
  }).join(' ');
}

// ── Rendu côte à côte (char-level) ──────────────────────────────────
function renderSideBySide(docId) {
  const doc = DATA.documents.find(d => d.doc_id === docId);
  if (!doc) return;

  const sel = document.getElementById('sbs-engine-dropdown');
  const engineIdx = sel && sel.value !== '' ? parseInt(sel.value, 10) : 0;
  const er = doc.engine_results[engineIdx];
  if (!er) return;

  const ops = er.diff || [];

  // Construire le HTML GT (gauche) et OCR (droite) depuis les mêmes ops
  let gtHtml = '', ocrHtml = '';
  ops.forEach(op => {
    if (op.op === 'equal') {
      const t = esc(op.text);
      gtHtml  += t;
      ocrHtml += t;
    } else if (op.op === 'delete') {
      // Présent dans GT, absent de l'OCR → orange dans GT
      gtHtml += '<span class="d-miss" title="Manquant dans OCR">' + esc(op.text) + '</span>';
    } else if (op.op === 'insert') {
      // Présent dans OCR, absent du GT → vert dans OCR
      ocrHtml += '<span class="d-ins-ocr" title="Insertion OCR">' + esc(op.text) + '</span>';
    } else if (op.op === 'replace') {
      // Substitution : orange dans GT, rouge dans OCR
      gtHtml  += '<span class="d-miss" title="Substitution GT">' + esc(op.old) + '</span>';
      ocrHtml += '<span class="d-err"  title="Différent du GT">'       + esc(op.new) + '</span>';
    }
  });

  document.getElementById('sbs-gt-body').innerHTML  = gtHtml  || '<em style="color:var(--text-muted)">—</em>';
  document.getElementById('sbs-ocr-body').innerHTML = ocrHtml || '<em style="color:var(--text-muted)">Aucune sortie</em>';

  // En-tête OCR : nom moteur + CER
  const c = cerColor(er.cer); const bg = cerBg(er.cer);
  const tier = cerTier(er.cer);
  document.getElementById('sbs-ocr-engine-name').textContent = er.engine;
  const cerBadgeEl = document.getElementById('sbs-ocr-cer');
  // Sprint 30 — ajout du tier non-couleur + aria-label (a11y WCAG).
  // L'icône préfixée distingue les paliers indépendamment de la couleur,
  // et ``aria-label`` est lu par les lecteurs d'écran.
  cerBadgeEl.textContent = `${cerTierIcon(tier)} ${pct(er.cer)}`;
  cerBadgeEl.setAttribute('data-cer-tier', tier);
  cerBadgeEl.setAttribute('aria-label', `${cerTierLabel(tier)} ${pct(er.cer)}`);
  cerBadgeEl.style.cssText = `color:${c};background:${bg};display:inline-block`;

  // Pipeline triple-diff (si applicable)
  const tripleEl = document.getElementById('sbs-triple-diff');
  if (er.ocr_intermediate) {
    const ocrDiffHtml = renderDiff(er.ocr_diff);
    const llmDiffHtml = renderDiff(er.llm_correction_diff);
    const isPipeline = er.ocr_intermediate !== undefined;
    const modeLabel = {text_only:'texte seul', text_and_image:'image+texte', zero_shot:'zero-shot'}[er.pipeline_mode] || '';
    const pipeTag = `<span class="pipeline-tag">⛓ ${modeLabel || 'pipeline'}</span>`;
    let onBadge = '';
    if (er.over_normalization) {
      const on = er.over_normalization;
      const onPct = (on.score * 100).toFixed(2);
      const cls = on.score > 0.05 ? 'over-norm-badge high' : 'over-norm-badge';
      onBadge = `<span class="${cls}" title="Classe 10 — sur-normalisation LLM">Sur-norm. ${onPct}%</span>`;
    }
    let diplomaBadge = '';
    if (er.cer_diplomatic !== null && er.cer_diplomatic !== undefined) {
      const dipC = cerColor(er.cer_diplomatic); const dipB = cerBg(er.cer_diplomatic);
      const delta = er.cer - er.cer_diplomatic;
      const deltaHint = delta > 0.001 ? ` (−${(delta*100).toFixed(1)}% avec normalisation)` : '';
      diplomaBadge = `<span class="cer-badge" style="color:${dipC};background:${dipB};opacity:.85"
        title="CER diplomatique${deltaHint}">diplo. ${pct(er.cer_diplomatic)}</span>`;
    }
    tripleEl.style.display = '';
    tripleEl.innerHTML = `
      <div style="margin-top:.75rem;padding-top:.75rem;border-top:1px solid var(--border)">
        <div style="display:flex;align-items:center;gap:.4rem;margin-bottom:.5rem;font-size:.83rem;font-weight:600">
          ${pipeTag} ${diplomaBadge} ${onBadge}
          <span class="badge" style="background:#f1f5f9">WER ${pct(er.wer)}</span>
        </div>
        <div class="triple-diff-wrap">
          <div class="triple-diff-section">
            <h5>GT → OCR brut</h5>
            ${ocrDiffHtml || '<em style="color:var(--text-muted)">—</em>'}
          </div>
          <div class="triple-diff-section">
            <h5>OCR brut → Correction LLM</h5>
            ${llmDiffHtml || '<em style="color:var(--text-muted)">—</em>'}
          </div>
        </div>
      </div>`;
  } else {
    // Afficher WER / CER diplomatique même hors pipeline
    let diplomaBadge = '';
    if (er.cer_diplomatic !== null && er.cer_diplomatic !== undefined) {
      const dipC = cerColor(er.cer_diplomatic); const dipB = cerBg(er.cer_diplomatic);
      const delta = er.cer - er.cer_diplomatic;
      const deltaHint = delta > 0.001 ? ` (−${(delta*100).toFixed(1)}% avec normalisation)` : '';
      diplomaBadge = `<span class="cer-badge" style="color:${dipC};background:${dipB};opacity:.85"
        title="CER diplomatique${deltaHint}">diplo. ${pct(er.cer_diplomatic)}</span>`;
    }
    const errBadge = er.error ? `<span class="badge" style="background:#fee2e2;color:#dc2626">Erreur</span>` : '';
    if (diplomaBadge || errBadge) {
      tripleEl.style.display = '';
      tripleEl.innerHTML = `<div style="margin-top:.5rem;display:flex;gap:.4rem;flex-wrap:wrap;font-size:.82rem">
        <span class="badge" style="background:#f1f5f9">WER ${pct(er.wer)}</span>
        ${diplomaBadge} ${errBadge}
      </div>`;
    } else {
      tripleEl.style.display = 'none';
      tripleEl.innerHTML = '';
    }
  }
}

// ── Score badge (ligatures / diacritiques) ───────────────────────
function _scoreBadge(v, label) {
  if (v === null || v === undefined) return '<span style="color:var(--text-muted)">—</span>';
  const pctVal = (v * 100).toFixed(1);
  const color = v >= 0.9 ? '#16a34a' : v >= 0.7 ? '#ca8a04' : '#dc2626';
  const bg = v >= 0.9 ? '#f0fdf4' : v >= 0.7 ? '#fefce8' : '#fef2f2';
  return `<span class="cer-badge" style="color:${color};background:${bg}" title="${label} : ${pctVal}%">${pctVal}%</span>`;
}

// ── Vue Classement ──────────────────────────────────────────────
let rankingSort = { col: 'cer', dir: 'asc' };

function renderRanking() {
  const engines = [...DATA.engines];
  // Trier
  engines.sort((a, b) => {
    let va = a[rankingSort.col], vb = b[rankingSort.col];
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    if (va === null) va = Infinity;
    if (vb === null) vb = Infinity;
    return rankingSort.dir === 'asc' ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  });

  const tbody = document.getElementById('ranking-tbody');
  tbody.innerHTML = engines.map((e, i) => {
    const rank = i + 1;
    const badgeClass = rank === 1 ? 'rank-badge rank-1' : 'rank-badge';
    const cerC = cerColor(e.cer); const cerB = cerBg(e.cer);
    const barW = Math.min(100, e.cer * 100 * 3);

    // Badge pipeline
    let pipelineBadge = '';
    let pipelineStepsHtml = '';
    if (e.is_pipeline && e.pipeline_info) {
      const pi = e.pipeline_info;
      const modeLabel = {text_only:'texte', text_and_image:'image+texte', zero_shot:'zero-shot'}[pi.pipeline_mode] || pi.pipeline_mode || '';
      pipelineBadge = `<span class="pipeline-tag" title="Pipeline OCR+LLM — mode ${modeLabel}">
        ⛓ pipeline<span class="pipe-arrow">·${modeLabel}</span></span>`;
      if (pi.pipeline_steps) {
        pipelineStepsHtml = `<div class="pipeline-steps">` +
          pi.pipeline_steps.map(s => s.type === 'ocr'
            ? `<span class="step-chip ocr">OCR: ${esc(s.engine)}</span>`
            : `<span class="step-chip llm">LLM: ${esc(s.model)}</span>`
          ).join(`<span class="step-arrow">→</span>`) +
          `</div>`;
      }
    }

    // Sur-normalisation (classe 10)
    let overNormCell = '<td style="color:var(--text-muted)">—</td>';
    if (e.is_pipeline && e.pipeline_info && e.pipeline_info.over_normalization) {
      const on = e.pipeline_info.over_normalization;
      const onPct = (on.score * 100).toFixed(2);
      const cls = on.score > 0.05 ? 'over-norm-badge high' : 'over-norm-badge';
      overNormCell = `<td><span class="${cls}" title="Classe 10 — ${on.over_normalized_count} mots corrects dégradés sur ${on.total_correct_ocr_words}">${onPct} %</span></td>`;
    }

    // CER diplomatique
    let diploCerCell = '<td style="color:var(--text-muted)">—</td>';
    if (e.cer_diplomatic !== null && e.cer_diplomatic !== undefined) {
      const dipC = cerColor(e.cer_diplomatic); const dipB = cerBg(e.cer_diplomatic);
      const delta = e.cer - e.cer_diplomatic;
      const deltaStr = delta > 0.001 ? ` <span style="font-size:.65rem;color:#059669">-${(delta*100).toFixed(1)}%</span>` : '';
      const profileHint = e.cer_diplomatic_profile ? ` title="Profil : ${esc(e.cer_diplomatic_profile)}"` : '';
      diploCerCell = `<td${profileHint}>
        <span class="cer-badge" style="color:${dipC};background:${dipB}">${pct(e.cer_diplomatic)}</span>${deltaStr}
      </td>`;
    }

    // ── Sprint 10 : Gini + Ancrage ─────────────────────────────────────
    let giniCell = '<td style="color:var(--text-muted)">—</td>';
    if (e.gini !== null && e.gini !== undefined) {
      const gv = e.gini;
      const gColor = gv < 0.3 ? '#16a34a' : gv < 0.5 ? '#ca8a04' : '#dc2626';
      const gBg = gv < 0.3 ? '#f0fdf4' : gv < 0.5 ? '#fefce8' : '#fef2f2';
      giniCell = `<td><span class="cer-badge" style="color:${gColor};background:${gBg}"
        title="Gini=${gv.toFixed(3)} — 0=uniforme, 1=concentré">${gv.toFixed(3)}</span></td>`;
    }
    let anchorCell = '<td style="color:var(--text-muted)">—</td>';
    if (e.anchor_score !== null && e.anchor_score !== undefined) {
      const av = e.anchor_score;
      const hallBadge = (e.hallucinating_doc_rate && e.hallucinating_doc_rate > 0.2)
        ? ' <span title="Hallucinations détectées">⚠️</span>' : '';
      anchorCell = `<td>${_scoreBadge(av, 'Ancrage trigrammes')}${hallBadge}</td>`;
    }

    return `<tr data-engine="${esc(e.name)}">
      <td><span class="${badgeClass}">${rank}</span></td>
      <td>
        <span class="engine-name">${esc(e.name)}</span>
        ${pipelineBadge}
        ${e.is_vlm ? '<span class="pipeline-tag" style="background:#fce7f3;color:#9d174d">👁 VLM</span>' : ''}
        <span class="engine-version">v${esc(e.version)}</span>
        ${pipelineStepsHtml}
      </td>
      <td data-col="cer">
        <span class="bar" style="width:${barW}px;background:${cerC}"></span>
        <span class="cer-badge" style="color:${cerC};background:${cerB}">${pct(e.cer)}</span>
      </td>
      ${diploCerCell.replace('<td', '<td data-col="cer_diplomatic"')}
      <td data-col="wer">${pct(e.wer)}</td>
      <td data-col="mer">${pct(e.mer)}</td>
      <td data-col="wil">${pct(e.wil)}</td>
      <td data-col="ligature_score">${_scoreBadge(e.ligature_score, 'Ligatures')}</td>
      <td data-col="diacritic_score">${_scoreBadge(e.diacritic_score, 'Diacritiques')}</td>
      ${giniCell.replace('<td', '<td data-col="gini"')}
      ${anchorCell.replace('<td', '<td data-col="anchor_score"')}
      <td style="color:var(--text-muted)">${pct(e.cer_median)}</td>
      <td style="color:var(--text-muted)">${pct(e.cer_min)}</td>
      <td style="color:var(--text-muted)">${pct(e.cer_max)}</td>
      ${overNormCell}
      <td><span class="pill">${e.doc_count}</span></td>
    </tr>`;
  }).join('');

  // Stats globales
  const pipelineCount = DATA.engines.filter(e => e.is_pipeline).length;
  const totalDocs = DATA.meta.document_count;
  const exclCount = EXCLUDED_DOCS.size;
  const activeDocs = totalDocs - exclCount;
  const stats = document.getElementById('ranking-stats');
  stats.innerHTML = `
    <div class="stat">Corpus <b>${esc(DATA.meta.corpus_name)}</b></div>
    <div class="stat">Documents <b>${activeDocs}</b>${exclCount > 0 ? ` <span style="font-size:.75rem;color:#dc2626">(−${exclCount} exclu${exclCount>1?'s':''})</span>` : ''}</div>
    <div class="stat">Concurrents <b>${DATA.engines.length}</b>
      ${pipelineCount ? `<span class="pipeline-tag" style="margin-left:.3rem">${pipelineCount} pipeline${pipelineCount>1?'s':''}</span>` : ''}
    </div>
  `;
}

// Tri au clic sur en-tête
document.querySelectorAll('#ranking-table th.sortable').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (rankingSort.col === col) {
      rankingSort.dir = rankingSort.dir === 'asc' ? 'desc' : 'asc';
    } else {
      rankingSort.col = col;
      rankingSort.dir = 'asc';
    }
    document.querySelectorAll('#ranking-table th').forEach(t => {
      t.classList.remove('sorted');
      const icon = t.querySelector('.sort-icon');
      if (icon) icon.textContent = '↕';
    });
    th.classList.add('sorted');
    const icon = th.querySelector('.sort-icon');
    if (icon) icon.textContent = rankingSort.dir === 'asc' ? '↑' : '↓';
    renderRanking();
  });
});

// ── Système d'exclusion globale ─────────────────────────────────
// Union de toutes les sources d'exclusion (manuelle + hallucination toggles)
const EXCLUDED_DOCS = new Set();
const _manualExclusions = new Set();
const _hallucinationExclusions = new Set();

// Données originales sauvegardées pour recalcul
const _originalEngines = JSON.parse(JSON.stringify(DATA.engines));

function _updateExcludedDocs() {
  EXCLUDED_DOCS.clear();
  _manualExclusions.forEach(id => EXCLUDED_DOCS.add(id));
  _hallucinationExclusions.forEach(id => EXCLUDED_DOCS.add(id));
  _updateExclusionBanner();
}

function _updateExclusionBanner() {
  const banner = document.getElementById('global-exclusion-banner');
  const text = document.getElementById('global-exclusion-text');
  if (EXCLUDED_DOCS.size > 0) {
    banner.style.display = '';
    text.textContent = EXCLUDED_DOCS.size + ' document' + (EXCLUDED_DOCS.size > 1 ? 's' : '') +
      ' exclu' + (EXCLUDED_DOCS.size > 1 ? 's' : '') + ' de l\'analyse' +
      (_manualExclusions.size > 0 ? ' (' + _manualExclusions.size + ' manuel' + (_manualExclusions.size > 1 ? 's' : '') + ')' : '') +
      (_hallucinationExclusions.size > 0 ? ' (' + _hallucinationExclusions.size + ' hallucination' + (_hallucinationExclusions.size > 1 ? 's' : '') + ')' : '');
  } else {
    banner.style.display = 'none';
  }
}

function resetAllExclusions() {
  _manualExclusions.clear();
  _hallucinationExclusions.clear();
  EXCLUDED_DOCS.clear();
  _updateExclusionBanner();
  // Reset hallucination toggles
  ['robust-cer-toggle','robust-anchor-toggle','robust-ratio-toggle'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) { btn.dataset.active = 'true'; btn.textContent = '✓'; btn.closest('label').classList.remove('criterion-off'); }
  });
  document.getElementById('robust-cer').value = 100;
  document.getElementById('robust-cer-val').textContent = '100%';
  document.getElementById('robust-anchor').value = 0.5;
  document.getElementById('robust-anchor-val').textContent = '0.50';
  document.getElementById('robust-ratio').value = 1.5;
  document.getElementById('robust-ratio-val').textContent = '1.5';
  recalculateAll();
  renderGallery();
}

function _recalcEngineMetrics() {
  // Recalcule les métriques agrégées de chaque moteur en excluant EXCLUDED_DOCS
  DATA.engines.forEach((eng, idx) => {
    const orig = _originalEngines[idx];
    if (EXCLUDED_DOCS.size === 0) {
      // Restaurer les valeurs originales
      eng.cer = orig.cer;
      eng.wer = orig.wer;
      eng.mer = orig.mer;
      eng.wil = orig.wil;
      eng.cer_median = orig.cer_median;
      eng.cer_min = orig.cer_min;
      eng.cer_max = orig.cer_max;
      eng.cer_values = orig.cer_values.slice();
      eng.doc_count = orig.doc_count;
      eng.gini = orig.gini;
      eng.anchor_score = orig.anchor_score;
      eng.length_ratio = orig.length_ratio;
      eng.hallucinating_doc_rate = orig.hallucinating_doc_rate;
      return;
    }
    // Recalculer depuis les documents non exclus
    const cerVals = [], werVals = [], merVals = [], wilVals = [];
    const giniVals = [], anchorVals = [];
    DATA.documents.forEach(doc => {
      if (EXCLUDED_DOCS.has(doc.doc_id)) return;
      const er = doc.engine_results.find(r => r.engine === eng.name);
      if (!er || er.error) return;
      if (er.cer !== null) cerVals.push(er.cer);
      if (er.wer !== null) werVals.push(er.wer);
      if (er.mer !== null) merVals.push(er.mer);
      if (er.wil !== null) wilVals.push(er.wil);
      const lm = er.line_metrics;
      if (lm && lm.gini !== null) giniVals.push(lm.gini);
      const hm = er.hallucination_metrics;
      if (hm && hm.anchor_score !== null) anchorVals.push(hm.anchor_score);
    });
    const mean = arr => arr.length ? arr.reduce((a,b) => a+b, 0) / arr.length : 0;
    const sorted = arr => [...arr].sort((a,b) => a - b);
    const median = arr => {
      if (!arr.length) return 0;
      const s = sorted(arr); const n = s.length;
      return n % 2 === 0 ? (s[n/2-1] + s[n/2]) / 2 : s[Math.floor(n/2)];
    };
    eng.cer = cerVals.length ? mean(cerVals) : orig.cer;
    eng.wer = werVals.length ? mean(werVals) : orig.wer;
    eng.mer = merVals.length ? mean(merVals) : orig.mer;
    eng.wil = wilVals.length ? mean(wilVals) : orig.wil;
    eng.cer_median = cerVals.length ? median(cerVals) : orig.cer_median;
    eng.cer_min = cerVals.length ? Math.min(...cerVals) : orig.cer_min;
    eng.cer_max = cerVals.length ? Math.max(...cerVals) : orig.cer_max;
    eng.cer_values = cerVals;
    eng.doc_count = cerVals.length;
    eng.gini = giniVals.length ? mean(giniVals) : orig.gini;
    eng.anchor_score = anchorVals.length ? mean(anchorVals) : orig.anchor_score;
  });
}

function recalculateAll() {
  console.log('[Picarones] recalculateAll — EXCLUDED_DOCS:', [...EXCLUDED_DOCS]);
  _recalcEngineMetrics();
  renderRanking();
  renderRobustMetrics();
  // Rebuild charts if they were already built
  if (chartsBuilt) {
    chartsBuilt = false;
    Object.keys(chartInstances).forEach(id => destroyChart(id));
    buildCharts();
  }
}

// ── Métriques robustes ──────────────────────────────────────────

function _computeHallucinationExclusions() {
  // Recalcule _hallucinationExclusions à partir des toggles/sliders
  _hallucinationExclusions.clear();
  const cerOn     = document.getElementById('robust-cer-toggle').dataset.active === 'true';
  const anchorOn  = document.getElementById('robust-anchor-toggle').dataset.active === 'true';
  const ratioOn   = document.getElementById('robust-ratio-toggle').dataset.active === 'true';
  const cerThreshold   = parseInt(document.getElementById('robust-cer').value) / 100;
  const anchorThreshold = parseFloat(document.getElementById('robust-anchor').value);
  const ratioThreshold  = parseFloat(document.getElementById('robust-ratio').value);

  DATA.documents.forEach(doc => {
    // Un doc est exclu par hallucination si AU MOINS un moteur le détecte comme problématique
    const dominated = doc.engine_results.some(er => {
      if (!er || er.error) return false;
      const hm = er.hallucination_metrics;
      if (cerOn && cerThreshold < 1.0 && er.cer !== null && er.cer > cerThreshold) return true;
      if (anchorOn && hm && hm.anchor_score < anchorThreshold) return true;
      if (ratioOn && hm && hm.length_ratio > ratioThreshold) return true;
      return false;
    });
    if (dominated) _hallucinationExclusions.add(doc.doc_id);
  });
  console.log('[Picarones] _hallucinationExclusions:', [..._hallucinationExclusions]);
  _updateExcludedDocs();
}

function _robustStat(arr) {
  // Retourne {mean, median, p90, p95} ou null si tableau vide
  if (!arr.length) return null;
  const sorted = [...arr].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((a, b) => a + b, 0) / n;
  const median = n % 2 === 0 ? (sorted[n/2-1] + sorted[n/2]) / 2 : sorted[Math.floor(n/2)];
  const p90 = sorted[Math.min(Math.ceil(n * 0.9) - 1, n - 1)];
  const p95 = sorted[Math.min(Math.ceil(n * 0.95) - 1, n - 1)];
  return { mean, median, p90, p95 };
}

function _deltaCell(globalVal, robustVal) {
  if (robustVal === null || globalVal === null) return '—';
  const delta = robustVal - globalVal;
  const cls = delta < -0.001 ? 'color:#16a34a' : delta > 0.001 ? 'color:#dc2626' : 'color:var(--text-muted)';
  const sign = delta >= 0 ? '+' : '';
  return `<span style="${cls}">${sign}${(delta*100).toFixed(2)}%</span>`;
}

function toggleRobustCriterion(id, btn) {
  const active = btn.dataset.active !== 'true';
  btn.dataset.active = active ? 'true' : 'false';
  btn.textContent = active ? '✓' : '✕';
  btn.closest('label').classList.toggle('criterion-off', !active);
  _computeHallucinationExclusions();
  recalculateAll();
}

function renderRobustMetrics() {
  const cerOn     = document.getElementById('robust-cer-toggle').dataset.active === 'true';
  const anchorOn  = document.getElementById('robust-anchor-toggle').dataset.active === 'true';
  const ratioOn   = document.getElementById('robust-ratio-toggle').dataset.active === 'true';
  const cerThreshold   = parseInt(document.getElementById('robust-cer').value) / 100;
  const anchorThreshold = parseFloat(document.getElementById('robust-anchor').value);
  const ratioThreshold  = parseFloat(document.getElementById('robust-ratio').value);
  const totalDocs = DATA.documents.length;

  // Pour chaque engine : recalculer métriques en excluant les docs problématiques
  const results = DATA.engines.map(eng => {
    const excluded = [];
    const cerVals = [], werVals = [], merVals = [], wilVals = [], giniVals = [], anchorVals = [];

    DATA.documents.forEach(doc => {
      const er = doc.engine_results.find(r => r.engine === eng.name);
      if (!er || er.error) return;
      const hm = er.hallucination_metrics;
      const lm = er.line_metrics;

      // Raisons d'exclusion
      const reasons = [];
      if (cerOn && cerThreshold < 1.0 && er.cer !== null && er.cer > cerThreshold)
        reasons.push(`CER ${(er.cer*100).toFixed(1)}% > ${(cerThreshold*100).toFixed(0)}%`);
      if (anchorOn && hm && hm.anchor_score < anchorThreshold)
        reasons.push(`ancrage ${hm.anchor_score.toFixed(3)} < ${anchorThreshold.toFixed(2)}`);
      if (ratioOn && hm && hm.length_ratio > ratioThreshold)
        reasons.push(`ratio ${hm.length_ratio.toFixed(2)} > ${ratioThreshold.toFixed(1)}`);
      if (_manualExclusions.has(doc.doc_id))
        reasons.push('exclusion manuelle');

      if (reasons.length > 0) {
        excluded.push({
          doc_id: doc.doc_id,
          cer: er.cer,
          anchor: hm ? hm.anchor_score : undefined,
          ratio: hm ? hm.length_ratio : undefined,
          reasons,
        });
      } else {
        if (er.cer !== null) cerVals.push(er.cer);
        if (er.wer !== null) werVals.push(er.wer);
        if (er.mer !== null) merVals.push(er.mer);
        if (er.wil !== null) wilVals.push(er.wil);
        if (lm && lm.gini !== null) giniVals.push(lm.gini);
        if (hm && hm.anchor_score !== null) anchorVals.push(hm.anchor_score);
      }
    });

    const meanOf = arr => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : null;
    return {
      name: eng.name,
      global_cer: eng.cer,
      global_wer: eng.wer,
      global_mer: eng.mer,
      global_wil: eng.wil,
      robust_cer: _robustStat(cerVals),
      robust_wer: meanOf(werVals),
      robust_mer: meanOf(merVals),
      robust_wil: meanOf(wilVals),
      robust_gini: meanOf(giniVals),
      robust_anchor: meanOf(anchorVals),
      robust_docs: cerVals.length,
      excluded_count: excluded.length,
      excluded_docs: excluded,
    };
  });

  // Résumé — nombre unique de docs exclus (au moins par un moteur)
  const allExcludedIds = new Set(results.flatMap(r => r.excluded_docs.map(d => d.doc_id)));
  const countExcl = allExcludedIds.size;
  const countIncl = totalDocs - countExcl;
  const summaryEl = document.getElementById('robust-summary');
  summaryEl.textContent = countExcl === 0
    ? `Aucun document exclu — métriques calculées sur ${totalDocs} documents.`
    : `${countExcl} doc${countExcl>1?'s':''} exclu${countExcl>1?'s':''} sur ${totalDocs} — métriques robustes calculées sur ${countIncl} document${countIncl>1?'s':''}.`;

  if (!results.some(r => r.robust_cer !== null)) {
    document.getElementById('robust-table-wrap').innerHTML =
      '<p style="color:var(--text-muted);font-size:.82rem">Aucune donnée disponible pour ce corpus.</p>';
    document.getElementById('robust-excluded-docs').innerHTML = '';
    return;
  }

  // Tableau comparatif étendu
  const fmt = v => v !== null ? pct(v) : '—';
  const rows = results.map(r => {
    const rs = r.robust_cer;
    const robCerMean = rs ? rs.mean : null;
    return `<tr>
      <td style="font-weight:600;white-space:nowrap">${esc(r.name)}</td>
      <td style="text-align:center">${fmt(r.global_cer)}</td>
      <td style="text-align:center">${rs ? pct(rs.mean) : '—'}</td>
      <td style="text-align:center">${_deltaCell(r.global_cer, robCerMean)}</td>
      <td style="text-align:center;color:var(--text-muted)">${rs ? pct(rs.median) : '—'}</td>
      <td style="text-align:center;color:var(--text-muted)">${rs ? pct(rs.p90) : '—'}</td>
      <td style="text-align:center;color:var(--text-muted)">${rs ? pct(rs.p95) : '—'}</td>
      <td style="text-align:center">${fmt(r.global_wer)}</td>
      <td style="text-align:center">${fmt(r.robust_wer)}</td>
      <td style="text-align:center">${_deltaCell(r.global_wer, r.robust_wer)}</td>
      <td style="text-align:center">${fmt(r.global_mer)}</td>
      <td style="text-align:center">${fmt(r.robust_mer)}</td>
      <td style="text-align:center">${fmt(r.global_wil)}</td>
      <td style="text-align:center">${fmt(r.robust_wil)}</td>
      <td style="text-align:center;color:var(--text-muted)">${r.robust_gini !== null ? r.robust_gini.toFixed(3) : '—'}</td>
      <td style="text-align:center;color:var(--text-muted)">${r.robust_anchor !== null ? r.robust_anchor.toFixed(3) : '—'}</td>
      <td style="text-align:center;color:var(--text-muted)">${r.excluded_count} / ${r.robust_docs}</td>
    </tr>`;
  }).join('');

  const thStyle = 'padding:.35rem .5rem;font-size:.75rem;white-space:nowrap;text-align:center;border-bottom:1px solid var(--border)';
  const thStyleL = thStyle + ';text-align:left';
  document.getElementById('robust-table-wrap').innerHTML = `
    <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;font-size:.82rem">
      <thead>
        <tr style="background:var(--bg)">
          <th style="${thStyleL}">Moteur</th>
          <th colspan="3" style="${thStyle};border-left:2px solid var(--border)">— CER —</th>
          <th colspan="3" style="${thStyle}">— CER robuste détail —</th>
          <th colspan="3" style="${thStyle};border-left:2px solid var(--border)">— WER —</th>
          <th colspan="2" style="${thStyle};border-left:2px solid var(--border)">— MER —</th>
          <th colspan="2" style="${thStyle};border-left:2px solid var(--border)">— WIL —</th>
          <th style="${thStyle};border-left:2px solid var(--border)">Gini rob.</th>
          <th style="${thStyle}">Ancrage rob.</th>
          <th style="${thStyle}">Excl./Incl.</th>
        </tr>
        <tr style="background:var(--bg)">
          <th style="${thStyleL}"></th>
          <th style="${thStyle};border-left:2px solid var(--border)">Global</th>
          <th style="${thStyle}">Robuste</th>
          <th style="${thStyle}">Δ</th>
          <th style="${thStyle}">Médiane</th>
          <th style="${thStyle}">P90</th>
          <th style="${thStyle}">P95</th>
          <th style="${thStyle};border-left:2px solid var(--border)">Global</th>
          <th style="${thStyle}">Robuste</th>
          <th style="${thStyle}">Δ</th>
          <th style="${thStyle};border-left:2px solid var(--border)">Global</th>
          <th style="${thStyle}">Robuste</th>
          <th style="${thStyle};border-left:2px solid var(--border)">Global</th>
          <th style="${thStyle}">Robuste</th>
          <th style="${thStyle};border-left:2px solid var(--border)"></th>
          <th style="${thStyle}"></th>
          <th style="${thStyle}"></th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    </div>`;

  // Documents exclus — liste déroulante unifiée
  if (allExcludedIds.size > 0) {
    // Collecter infos par doc_id (union des raisons de tous les moteurs)
    const docInfoMap = new Map();
    results.forEach(r => {
      r.excluded_docs.forEach(d => {
        if (!docInfoMap.has(d.doc_id)) {
          docInfoMap.set(d.doc_id, { doc_id: d.doc_id, cer: d.cer, anchor: d.anchor, ratio: d.ratio, reasons: new Set() });
        }
        d.reasons.forEach(reason => docInfoMap.get(d.doc_id).reasons.add(reason));
      });
    });
    const uniqDocs = [...docInfoMap.values()].sort((a,b) => a.doc_id.localeCompare(b.doc_id));
    document.getElementById('robust-excluded-docs').innerHTML =
      `<details><summary style="cursor:pointer;font-size:.82rem;color:var(--text-muted)">` +
      `▶ Documents exclus (${uniqDocs.length})</summary>` +
      `<ul style="margin:.4rem 0 0 1rem;font-size:.8rem;color:var(--text-muted);max-height:220px;overflow-y:auto">` +
      uniqDocs.map(d => {
        const cerStr = d.cer !== null && d.cer !== undefined ? ` CER ${(d.cer*100).toFixed(1)}%` : '';
        return `<li><a href="#" onclick="openDocument('${esc(d.doc_id)}');return false">${esc(d.doc_id)}</a>${cerStr} — ${[...d.reasons].join(', ')}</li>`;
      }).join('') +
      `</ul></details>`;
  } else {
    document.getElementById('robust-excluded-docs').innerHTML = '';
  }
}

// ── Vue Galerie ─────────────────────────────────────────────────
function toggleGalleryExclusion(docId, checked) {
  if (checked) {
    _manualExclusions.delete(docId);
  } else {
    _manualExclusions.add(docId);
  }
  _updateExcludedDocs();
  _updateGalleryExclusionUI();
}

function resetGalleryExclusions() {
  _manualExclusions.clear();
  _updateExcludedDocs();
  renderGallery();
  recalculateAll();
}

function _updateGalleryExclusionUI() {
  const count = _manualExclusions.size;
  const btn = document.getElementById('gallery-reset-btn');
  const info = document.getElementById('gallery-exclusion-info');
  if (count > 0) {
    btn.style.display = '';
    info.style.display = '';
    info.textContent = `${count} document${count>1?'s':''} exclu${count>1?'s':''} manuellement de l'analyse.`;
  } else {
    btn.style.display = 'none';
    info.style.display = 'none';
  }
  recalculateAll();
}

function renderGallery() {
  const sortKey  = document.getElementById('gallery-sort').value;
  const filterCer = parseFloat(document.getElementById('gallery-filter-cer').value) / 100 || 0;
  const filterEngine = document.getElementById('gallery-engine-select').value;

  let docs = [...DATA.documents];

  // Filtre CER
  if (filterCer > 0) {
    docs = docs.filter(d => {
      if (filterEngine) {
        const er = d.engine_results.find(r => r.engine === filterEngine);
        return er && er.cer >= filterCer;
      }
      return d.mean_cer >= filterCer;
    });
  }

  // Tri
  docs.sort((a, b) => {
    if (sortKey === 'mean_cer') return a.mean_cer - b.mean_cer;
    if (sortKey === 'difficulty_score') return (b.difficulty_score||0) - (a.difficulty_score||0);
    if (sortKey === 'best_engine') return a.best_engine.localeCompare(b.best_engine);
    return a.doc_id.localeCompare(b.doc_id);
  });

  const grid = document.getElementById('gallery-grid');
  const empty = document.getElementById('gallery-empty');

  if (!docs.length) {
    grid.innerHTML = '';
    empty.style.display = '';
    return;
  }
  empty.style.display = 'none';

  // Mise à jour bouton reset
  const btn = document.getElementById('gallery-reset-btn');
  const info = document.getElementById('gallery-exclusion-info');
  if (_manualExclusions.size > 0) {
    btn.style.display = '';
    info.style.display = '';
    info.textContent = `${_manualExclusions.size} document${_manualExclusions.size>1?'s':''} exclu${_manualExclusions.size>1?'s':''} manuellement de l'analyse.`;
  } else {
    btn.style.display = 'none';
    info.style.display = 'none';
  }

  grid.innerHTML = docs.map(doc => {
    const imgTag = doc.image_b64
      ? `<img src="${doc.image_b64}" alt="${esc(doc.doc_id)}" loading="lazy">`
      : `<div class="img-placeholder">🖹</div>`;

    const badges = doc.engine_results.map(er => {
      const c = cerColor(er.cer); const bg = cerBg(er.cer);
      const isPipe = er.ocr_intermediate !== undefined;
      const label = isPipe ? '⛓' + er.engine.slice(0,8) : er.engine.slice(0,8);
      return `<span class="engine-cer-badge" style="color:${c};background:${bg}"
        title="${esc(er.engine)}${isPipe?' (pipeline)':''}">${esc(label)} ${pct(er.cer,1)}</span>`;
    }).join('');

    // Difficulty badge
    let diffBadge = '';
    if (doc.difficulty_score !== undefined) {
      const dScore = doc.difficulty_score;
      const dColor = dScore < 0.25 ? '#16a34a' : dScore < 0.5 ? '#ca8a04' : dScore < 0.75 ? '#ea580c' : '#dc2626';
      const dBg    = dScore < 0.25 ? '#f0fdf4' : dScore < 0.5 ? '#fefce8' : dScore < 0.75 ? '#fff7ed' : '#fef2f2';
      diffBadge = `<span class="diff-badge" style="color:${dColor};background:${dBg};margin-left:.3rem"
        title="Difficulté intrinsèque : ${doc.difficulty_label}">⚡ ${doc.difficulty_label}</span>`;
    }

    const isExcluded = _manualExclusions.has(doc.doc_id);
    const checkboxId = `gal-chk-${doc.doc_id.replace(/[^a-z0-9]/gi,'_')}`;
    const cardStyle = isExcluded ? 'opacity:.5;border:2px dashed #dc2626' : '';
    return `<div class="gallery-card" style="${cardStyle}">
      <label class="gallery-exclude-label" title="${isExcluded ? 'Inclure dans l\'analyse' : 'Exclure de l\'analyse'}"
        style="position:absolute;top:.35rem;right:.35rem;z-index:2;cursor:pointer;background:rgba(255,255,255,.85);border-radius:.25rem;padding:.1rem .25rem;font-size:.7rem;display:flex;align-items:center;gap:.25rem">
        <input type="checkbox" id="${checkboxId}" ${isExcluded ? '' : 'checked'}
          onchange="toggleGalleryExclusion('${esc(doc.doc_id)}',this.checked)"
          onclick="event.stopPropagation()">
        <span>${isExcluded ? 'Exclu' : 'Inclus'}</span>
      </label>
      <div onclick="openDocument('${esc(doc.doc_id)}')">
        ${imgTag}
        <div class="gallery-card-body">
          <div class="gallery-card-title">${esc(doc.doc_id)}${diffBadge}</div>
          <div class="gallery-card-badges">${badges}</div>
        </div>
      </div>
    </div>`;
  }).join('');
}

// ── Vue Document ────────────────────────────────────────────────
let currentDocId = null;
let zoomLevel = 1;
let dragStart = null;
let imgOffset = { x: 0, y: 0 };

function openDocument(docId) {
  _switchView('document');
  updateURL('document', { doc: docId });
  loadDocument(docId);
}

function loadDocument(docId) {
  const doc = DATA.documents.find(d => d.doc_id === docId);
  if (!doc) return;
  currentDocId = docId;

  // Sidebar : highlight
  document.querySelectorAll('.doc-list-item').forEach(el => {
    el.classList.toggle('active', el.dataset.docId === docId);
  });

  // Titre
  document.getElementById('doc-detail-title').textContent = doc.doc_id;

  // Métriques
  const metricsDiv = document.getElementById('doc-detail-metrics');
  const cer = doc.mean_cer;
  const dScore = doc.difficulty_score;
  const dColor = dScore < 0.25 ? '#16a34a' : dScore < 0.5 ? '#ca8a04' : dScore < 0.75 ? '#ea580c' : '#dc2626';
  const dLabel = doc.difficulty_label || '';
  metricsDiv.innerHTML = `<div class="stat">CER moyen <b style="color:${cerColor(cer)}">${pct(cer)}</b></div>
    <div class="stat">Meilleur moteur <b>${esc(doc.best_engine)}</b></div>
    ${dScore !== undefined ? `<div class="stat">Difficulté <b style="color:${dColor}">${dLabel} (${(dScore*100).toFixed(0)}%)</b></div>` : ''}`;

  // Image
  resetZoom();
  const img = document.getElementById('doc-image');
  const placeholder = document.getElementById('doc-image-placeholder');
  if (doc.image_b64) {
    img.src = doc.image_b64;
    img.style.display = '';
    placeholder.style.display = 'none';
  } else {
    img.style.display = 'none';
    placeholder.style.display = '';
    placeholder.innerHTML = `<span style="font-size:2rem">🖹</span><span>${esc(doc.image_path)}</span>`;
  }

  // Side-by-side diff — sélecteur de concurrent
  const selWrap = document.getElementById('sbs-engine-select');
  const sel = document.getElementById('sbs-engine-dropdown');
  if (doc.engine_results.length > 1) {
    sel.innerHTML = doc.engine_results.map((er, i) =>
      `<option value="${i}">${esc(er.engine)}</option>`
    ).join('');
    selWrap.style.display = '';
  } else {
    sel.innerHTML = '';
    selWrap.style.display = 'none';
  }
  renderSideBySide(docId);

  // ── Sprint 10 : distribution CER par ligne ──────────────────────────
  const lineCard = document.getElementById('doc-line-metrics-card');
  const lineContent = document.getElementById('doc-line-metrics-content');
  // Prendre le premier moteur ayant des line_metrics
  const erWithLine = doc.engine_results.find(er => er.line_metrics);
  if (erWithLine && erWithLine.line_metrics) {
    lineCard.style.display = '';
    lineContent.innerHTML = renderLineMetrics(doc.engine_results);
  } else {
    lineCard.style.display = 'none';
  }

  // ── Sprint 10 : hallucinations ──────────────────────────────────────
  const hallCard = document.getElementById('doc-hallucination-card');
  const hallContent = document.getElementById('doc-hallucination-content');
  const erWithHall = doc.engine_results.find(er => er.hallucination_metrics && er.hallucination_metrics.is_hallucinating);
  if (erWithHall || doc.engine_results.some(er => er.hallucination_metrics)) {
    hallCard.style.display = '';
    hallContent.innerHTML = renderHallucinationPanel(doc.engine_results);
  } else {
    hallCard.style.display = 'none';
  }
}

// ── Sprint 10 : rendu distribution CER par ligne ────────────────
function renderLineMetrics(engineResults) {
  const heatmapColors = (v) => {
    if (v < 0.05) return '#86efac';
    if (v < 0.15) return '#fde68a';
    if (v < 0.30) return '#fb923c';
    return '#f87171';
  };

  return engineResults.filter(er => er.line_metrics).map(er => {
    const lm = er.line_metrics;
    const c = cerColor(er.cer); const bg = cerBg(er.cer);

    // Heatmap de position
    const heatmap = lm.heatmap || [];
    const maxHeat = Math.max(...heatmap, 0.01);
    const heatmapHtml = heatmap.length > 0
      ? `<div class="heatmap-wrap">` +
        heatmap.map((v, i) => {
          const h = Math.max(4, Math.round(60 * v / maxHeat));
          return `<div class="heatmap-bar" style="height:${h}px;background:${heatmapColors(v)}"
            title="Tranche ${i+1}/${heatmap.length} — CER=${(v*100).toFixed(1)}%"></div>`;
        }).join('') +
        `</div><div class="heatmap-labels"><span>${I18N.heatmap_start||'Début'}</span><span>${I18N.heatmap_mid||'Milieu'}</span><span>${I18N.heatmap_end||'Fin'}</span></div>`
      : '<em style="color:var(--text-muted)">—</em>';

    // Percentiles
    const p = lm.percentiles || {};
    const pctBars = ['p50','p75','p90','p95','p99'].map(k => {
      const v = p[k] || 0;
      const w = Math.min(100, v * 100 * 2);
      const fillColor = v < 0.15 ? '#86efac' : v < 0.30 ? '#fde68a' : '#f87171';
      return `<div class="pct-bar-row">
        <span class="pct-bar-label">${k}</span>
        <div class="pct-bar-track"><div class="pct-bar-fill" style="width:${w}%;background:${fillColor}"></div></div>
        <span class="pct-bar-val">${(v*100).toFixed(1)}%</span>
      </div>`;
    }).join('');

    // Taux catastrophiques
    const cr = lm.catastrophic_rate || {};
    const crRows = Object.entries(cr).map(([t, rate]) => {
      const tPct = (parseFloat(t)*100).toFixed(0);
      const ratePct = (rate*100).toFixed(1);
      const color = rate < 0.05 ? '#16a34a' : rate < 0.15 ? '#ca8a04' : '#dc2626';
      return `<span class="stat"><b style="color:${color}">${ratePct}%</b> lignes CER&gt;${tPct}%</span>`;
    }).join('');

    // Gini
    const gini = lm.gini !== undefined ? lm.gini.toFixed(3) : '—';
    const giniColor = lm.gini < 0.3 ? '#16a34a' : lm.gini < 0.5 ? '#ca8a04' : '#dc2626';

    return `<div style="margin-bottom:1.25rem;padding-bottom:1rem;border-bottom:1px solid var(--border)">
      <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem">
        <strong>${esc(er.engine)}</strong>
        <span class="cer-badge" style="color:${c};background:${bg}">${pct(er.cer)}</span>
        <span class="stat">Gini <b style="color:${giniColor}">${gini}</b></span>
        <span class="stat">${lm.line_count} ${I18N.lines||'lignes'}</span>
        ${crRows}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
        <div>
          <div style="font-size:.75rem;font-weight:600;color:var(--text-muted);margin-bottom:.3rem">${I18N.heatmap_title||'CARTE THERMIQUE (position)'}</div>
          ${heatmapHtml}
        </div>
        <div>
          <div style="font-size:.75rem;font-weight:600;color:var(--text-muted);margin-bottom:.3rem">${I18N.percentile_title||'PERCENTILES CER'}</div>
          <div class="pct-bars">${pctBars}</div>
        </div>
      </div>
    </div>`;
  }).join('') || `<em style="color:var(--text-muted)">${I18N.no_line_metrics||'Aucune métrique de ligne disponible.'}</em>`;
}

// ── Sprint 10 : rendu panneau hallucinations ─────────────────────
function renderHallucinationPanel(engineResults) {
  const withHall = engineResults.filter(er => er.hallucination_metrics);
  if (!withHall.length) return `<em style="color:var(--text-muted)">${I18N.no_hall_metrics||"Aucune métrique d'hallucination disponible."}</em>`;

  return withHall.map(er => {
    const hm = er.hallucination_metrics;
    const isHall = hm.is_hallucinating;
    const badgeClass = isHall ? 'hallucination-badge' : 'hallucination-badge ok';
    const badgeLabel = isHall ? (I18N.hall_detected||'⚠️ Hallucinations détectées') : (I18N.hall_ok||'✓ Ancrage satisfaisant');

    const blocksHtml = hm.hallucinated_blocks && hm.hallucinated_blocks.length > 0
      ? hm.hallucinated_blocks.slice(0, 5).map(b =>
          `<div class="halluc-block">
            <div class="halluc-block-meta">${I18N.hall_block_label||'Bloc halluciné'} — ${b.length} mots (tokens ${b.start_token}–${b.end_token})</div>
            ${esc(b.text)}
          </div>`
        ).join('') +
        (hm.hallucinated_blocks.length > 5 ? `<div style="font-size:.72rem;color:var(--text-muted);margin-top:.25rem">… ${hm.hallucinated_blocks.length - 5} ${I18N.hall_more_blocks||'bloc(s) supplémentaire(s)'}</div>` : '')
      : `<em style="color:var(--text-muted);font-size:.8rem">${I18N.no_hall_blocks||'Aucun bloc halluciné détecté.'}</em>`;

    return `<div style="margin-bottom:1.25rem;padding-bottom:1rem;border-bottom:1px solid var(--border)">
      <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.6rem;flex-wrap:wrap">
        <strong>${esc(er.engine)}</strong>
        <span class="${badgeClass}">${badgeLabel}</span>
        <span class="stat">Ancrage <b>${(hm.anchor_score*100).toFixed(1)}%</b></span>
        <span class="stat">Ratio longueur <b>${hm.length_ratio.toFixed(2)}</b></span>
        <span class="stat">Insertion nette <b>${(hm.net_insertion_rate*100).toFixed(1)}%</b></span>
        <span class="stat">${hm.gt_word_count} mots GT / ${hm.hyp_word_count} mots sortie</span>
      </div>
      ${isHall ? `<div style="margin-bottom:.5rem;font-size:.82rem;font-weight:600;color:#9d174d">${I18N.hall_blocks_title||'Blocs sans ancrage dans le GT :'}</div>` : ''}
      ${isHall ? blocksHtml : ''}
    </div>`;
  }).join('');
}

// ── Sprint 10 — Scatter Gini vs CER moyen ──────────────────────
function buildGiniCerScatter() {
  const canvas = document.getElementById('chart-gini-cer');
  if (!canvas) return;
  const pts = DATA.gini_vs_cer || [];
  if (!pts.length) {
    canvas.parentElement.innerHTML = `<p style="color:var(--text-muted);padding:1rem">${I18N.no_gini||'Données Gini non disponibles.'}</p>`;
    return;
  }
  const datasets = pts.map((p, i) => ({
    label: p.engine,
    data: [{ x: p.cer * 100, y: p.gini }],
    backgroundColor: engineColor(DATA.engines.findIndex(e => e.name === p.engine)) + 'cc',
    borderColor: engineColor(DATA.engines.findIndex(e => e.name === p.engine)),
    borderWidth: p.is_pipeline ? 2 : 1,
    pointRadius: p.is_pipeline ? 9 : 7,
    pointStyle: p.is_pipeline ? 'triangle' : 'circle',
  }));

  chartInstances['gini-cer'] = new Chart(canvas.getContext('2d'), {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { size: 11 } } },
        tooltip: { callbacks: {
          label: ctx => `${ctx.dataset.label}: CER=${ctx.parsed.x.toFixed(2)}%, Gini=${ctx.parsed.y.toFixed(3)}`,
        } },
      },
      scales: {
        x: { min: 0, title: { display: true, text: 'CER moyen (%)', font: { size: 11 } } },
        y: { min: 0, max: 1, title: { display: true, text: 'Coefficient de Gini', font: { size: 11 } } },
      },
    },
  });
}

// ── Sprint 10 — Scatter ratio longueur vs score d'ancrage ────────
function buildRatioAnchorScatter() {
  const canvas = document.getElementById('chart-ratio-anchor');
  if (!canvas) return;
  const pts = DATA.ratio_vs_anchor || [];
  if (!pts.length) {
    canvas.parentElement.innerHTML = `<p style="color:var(--text-muted);padding:1rem">Données d'ancrage non disponibles.</p>`;
    return;
  }

  // Zone de danger (ancrage < 0.5 OU ratio > 1.2) dessinée via plugin
  const datasets = pts.map((p, i) => ({
    label: p.engine + (p.is_vlm ? ' 👁' : ''),
    data: [{ x: p.anchor_score, y: p.length_ratio }],
    backgroundColor: engineColor(DATA.engines.findIndex(e => e.name === p.engine)) + 'cc',
    borderColor: engineColor(DATA.engines.findIndex(e => e.name === p.engine)),
    borderWidth: p.is_vlm ? 3 : 1,
    pointRadius: p.is_vlm ? 10 : 7,
    pointStyle: p.is_vlm ? 'star' : 'circle',
  }));

  chartInstances['ratio-anchor'] = new Chart(canvas.getContext('2d'), {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { size: 11 } } },
        tooltip: { callbacks: {
          label: ctx => `${ctx.dataset.label}: ancrage=${(ctx.parsed.x*100).toFixed(1)}%, ratio=${ctx.parsed.y.toFixed(2)}`,
        } },
      },
      scales: {
        x: { min: 0, max: 1, title: { display: true, text: "Score d'ancrage [0–1]", font: { size: 11 } } },
        y: { min: 0, title: { display: true, text: 'Ratio longueur (sortie/GT)', font: { size: 11 } } },
      },
    },
    plugins: [{
      id: 'danger-zones',
      beforeDraw(chart) {
        const { ctx: c, chartArea: { left, top, right, bottom }, scales: { x, y } } = chart;
        c.save();
        // Ancrage < 0.5 (gauche)
        const xHalf = x.getPixelForValue(0.5);
        c.fillStyle = 'rgba(239,68,68,0.07)';
        c.fillRect(left, top, xHalf - left, bottom - top);
        // Ratio > 1.2 (haut)
        const y12 = y.getPixelForValue(1.2);
        if (y12 > top) {
          c.fillRect(left, top, right - left, y12 - top);
        }
        // Lignes de seuil
        c.strokeStyle = 'rgba(239,68,68,0.35)'; c.lineWidth = 1; c.setLineDash([4,4]);
        c.beginPath(); c.moveTo(xHalf, top); c.lineTo(xHalf, bottom); c.stroke();
        if (y12 > top) {
          c.beginPath(); c.moveTo(left, y12); c.lineTo(right, y12); c.stroke();
        }
        c.restore();
      },
    }],
  });
}

function buildDocList() {
  const list = document.getElementById('doc-list');
  list.innerHTML = DATA.documents.map(doc => {
    const c = cerColor(doc.mean_cer); const bg = cerBg(doc.mean_cer);
    return `<div class="doc-list-item" data-doc-id="${esc(doc.doc_id)}"
        onclick="loadDocument('${esc(doc.doc_id)}')">
      <span class="doc-list-label">${esc(doc.doc_id)}</span>
      <span class="doc-list-cer" style="color:${c};background:${bg}">${pct(doc.mean_cer,1)}</span>
    </div>`;
  }).join('');
  if (DATA.documents.length) loadDocument(DATA.documents[0].doc_id);
}

// Zoom
function handleZoom(e) {
  e.preventDefault();
  zoom(e.deltaY < 0 ? 1.15 : 0.87);
}
function zoom(factor) {
  zoomLevel = Math.max(0.5, Math.min(5, zoomLevel * factor));
  applyZoom();
}
function resetZoom() {
  zoomLevel = 1; imgOffset = { x: 0, y: 0 };
  applyZoom();
}
function applyZoom() {
  const img = document.getElementById('doc-image');
  img.style.transform = `scale(${zoomLevel}) translate(${imgOffset.x}px, ${imgOffset.y}px)`;
}
function startDrag(e) {
  if (zoomLevel <= 1) return;
  dragStart = { x: e.clientX - imgOffset.x * zoomLevel, y: e.clientY - imgOffset.y * zoomLevel };
  document.getElementById('doc-image-wrap').style.cursor = 'grabbing';
}
function doDrag(e) {
  if (!dragStart) return;
  imgOffset.x = (e.clientX - dragStart.x) / zoomLevel;
  imgOffset.y = (e.clientY - dragStart.y) / zoomLevel;
  applyZoom();
}
function endDrag() {
  dragStart = null;
  document.getElementById('doc-image-wrap').style.cursor = zoomLevel > 1 ? 'grab' : 'zoom-in';
}

// ── Graphiques ──────────────────────────────────────────────────
let chartsBuilt = false;
let chartInstances = {};

function destroyChart(id) {
  if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; }
}

function buildCharts() {
  if (chartsBuilt) return;
  chartsBuilt = true;
  buildCerHistogram();
  buildRadar();
  buildCerPerDoc();
  buildDurationChart();
  buildQualityCerScatter();
  buildTaxonomyChart();
  // Sprint 7
  buildReliabilityCurves();
  buildBootstrapCIChart();
  buildVennDiagram();
  buildWilcoxonTable();
  buildErrorClusters();
  initCorrelationMatrix();
  // Sprint 10
  buildGiniCerScatter();
  buildRatioAnchorScatter();
}

function buildCerHistogram() {
  destroyChart('cer-hist');
  const ctx = document.getElementById('chart-cer-hist').getContext('2d');
  // Construire histogramme à bins fixes [0-5, 5-10, 10-20, 20-30, 30-50, 50+]
  const bins    = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.01];
  const labels  = ['0–5%', '5–10%', '10–20%', '20–30%', '30–50%', '>50%'];
  const colors  = ['#16a34a','#65a30d','#ca8a04','#ea580c','#dc2626','#9f1239'];

  const datasets = DATA.engines.map((e, ei) => {
    const counts = new Array(labels.length).fill(0);
    e.cer_values.forEach(v => {
      for (let i = 0; i < bins.length - 1; i++) {
        if (v >= bins[i] && v < bins[i+1]) { counts[i]++; break; }
      }
    });
    return {
      label: e.name, data: counts,
      backgroundColor: engineColor(ei) + 'aa',
      borderColor: engineColor(ei),
      borderWidth: 1,
    };
  });

  chartInstances['cer-hist'] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { font: { size: 11 } } } },
      scales: {
        x: { title: { display: true, text: 'Plage CER', font: { size: 11 } } },
        y: { title: { display: true, text: 'Nombre de documents', font: { size: 11 } },
               ticks: { stepSize: 1 } },
      },
    },
  });
}

function buildRadar() {
  destroyChart('radar');
  const ctx = document.getElementById('chart-radar').getContext('2d');
  // Axes : CER, WER, MER, WIL inversés (1 - valeur → plus c'est élevé, mieux c'est)
  const metrics = ['CER', 'WER', 'MER', 'WIL'];
  const keys    = ['cer', 'wer', 'mer', 'wil'];
  const datasets = DATA.engines.map((e, i) => {
    const data = keys.map(k => Math.max(0, (1 - (e[k] || 0)) * 100));
    return {
      label: e.name, data,
      backgroundColor: engineColor(i) + '33',
      borderColor: engineColor(i),
      borderWidth: 2,
      pointRadius: 4,
      pointHoverRadius: 6,
    };
  });

  chartInstances['radar'] = new Chart(ctx, {
    type: 'radar',
    data: { labels: metrics, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { font: { size: 11 } } } },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 20, font: { size: 10 } },
          pointLabels: { font: { size: 12, weight: 'bold' } },
        },
      },
    },
  });
}

function buildCerPerDoc() {
  destroyChart('cer-doc');
  const ctx = document.getElementById('chart-cer-doc').getContext('2d');
  const filteredDocs = DATA.documents.filter(d => !EXCLUDED_DOCS.has(d.doc_id));
  const labels = filteredDocs.map(d => d.doc_id);
  const datasets = DATA.engines.map((e, ei) => {
    const data = filteredDocs.map(doc => {
      const er = doc.engine_results.find(r => r.engine === e.name);
      return er ? er.cer * 100 : null;
    });
    return {
      label: e.name, data,
      borderColor: engineColor(ei),
      backgroundColor: engineColor(ei) + '22',
      tension: 0.3, fill: false,
      pointRadius: 3, pointHoverRadius: 5,
    };
  });

  chartInstances['cer-doc'] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { font: { size: 11 } } } },
      scales: {
        x: { ticks: { maxRotation: 45, font: { size: 10 } } },
        y: { title: { display: true, text: 'CER (%)', font: { size: 11 } }, min: 0 },
      },
    },
  });
}

function buildDurationChart() {
  destroyChart('duration');
  const ctx = document.getElementById('chart-duration').getContext('2d');

  const filteredDocs = DATA.documents.filter(d => !EXCLUDED_DOCS.has(d.doc_id));
  const labels = DATA.engines.map(e => e.name);
  const data   = DATA.engines.map(e => {
    const durs = filteredDocs.flatMap(d => d.engine_results
      .filter(r => r.engine === e.name)
      .map(r => r.duration));
    const mean = durs.length ? durs.reduce((a,b) => a+b, 0) / durs.length : 0;
    return parseFloat(mean.toFixed(3));
  });

  chartInstances['duration'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Durée moy. (s)',
        data,
        backgroundColor: DATA.engines.map((_, i) => engineColor(i) + 'aa'),
        borderColor:     DATA.engines.map((_, i) => engineColor(i)),
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { title: { display: true, text: 'Secondes', font: { size: 11 } }, min: 0 },
      },
    },
  });
}

function buildQualityCerScatter() {
  const ctx = document.getElementById('chart-quality-cer');
  if (!ctx) return;
  const filteredDocs = DATA.documents.filter(d => !EXCLUDED_DOCS.has(d.doc_id));
  // Construire les points : un par document, un dataset par moteur
  const datasets = DATA.engines.map((e, ei) => {
    const points = filteredDocs.flatMap(doc => {
      const er = doc.engine_results.find(r => r.engine === e.name);
      if (!er || er.error || !er.image_quality) return [];
      return [{ x: er.image_quality.quality_score, y: er.cer * 100 }];
    });
    return {
      label: e.name, data: points,
      backgroundColor: engineColor(ei) + 'bb',
      borderColor: engineColor(ei),
      borderWidth: 1, pointRadius: 5, pointHoverRadius: 7,
    };
  }).filter(d => d.data.length > 0);

  if (!datasets.length) { ctx.parentElement.innerHTML = '<p style="color:var(--text-muted);padding:1rem">Aucune donnée de qualité image disponible.</p>'; return; }

  chartInstances['quality-cer'] = new Chart(ctx.getContext('2d'), {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { size: 11 } } },
        tooltip: { callbacks: {
          label: ctx => `${ctx.dataset.label}: qualité=${ctx.parsed.x.toFixed(2)}, CER=${ctx.parsed.y.toFixed(1)}%`,
        } },
      },
      scales: {
        x: { min: 0, max: 1, title: { display: true, text: 'Score qualité image [0–1]', font: { size: 11 } } },
        y: { min: 0, title: { display: true, text: 'CER (%)', font: { size: 11 } } },
      },
    },
  });
}

function buildTaxonomyChart() {
  const ctx = document.getElementById('chart-taxonomy');
  if (!ctx) return;
  const taxLabels = ['Confusion visuelle','Diacritique','Casse','Ligature','Abréviation','Hapax','Segmentation','Hors-vocab.','Lacune'];
  const taxKeys = ['visual_confusion','diacritic_error','case_error','ligature_error','abbreviation_error','hapax','segmentation_error','oov_character','lacuna'];
  const taxColors = ['#6366f1','#f59e0b','#ec4899','#14b8a6','#8b5cf6','#64748b','#f97316','#06b6d4','#ef4444'];

  const datasets = DATA.engines.map((e, ei) => {
    const tax = e.aggregated_taxonomy;
    const data = taxKeys.map(k => tax && tax.counts ? (tax.counts[k] || 0) : 0);
    return {
      label: e.name, data,
      backgroundColor: engineColor(ei) + '99',
      borderColor: engineColor(ei),
      borderWidth: 1,
    };
  });

  chartInstances['taxonomy'] = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: { labels: taxLabels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { font: { size: 11 } } } },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: { title: { display: true, text: "Nb d'erreurs", font: { size: 11 } }, min: 0, ticks: { stepSize: 1 } },
      },
    },
  });
}

// ── Sprint 7 — Courbes de fiabilité ─────────────────────────────
function buildReliabilityCurves() {
  const ctx = document.getElementById('chart-reliability');
  if (!ctx) return;
  const curves = DATA.reliability_curves || [];
  if (!curves.length) { ctx.parentElement.innerHTML = '<p style="color:var(--text-muted);padding:1rem">Données insuffisantes.</p>'; return; }
  const datasets = curves.map((c, i) => {
    const points = (c.points || []).map(p => ({ x: p.pct_docs, y: p.mean_cer * 100 }));
    return {
      label: c.engine, data: points,
      borderColor: engineColor(i), backgroundColor: engineColor(i) + '22',
      tension: 0.3, fill: false, pointRadius: 2, pointHoverRadius: 5,
    };
  });
  destroyChart('reliability');
  chartInstances['reliability'] = new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      parsing: { xAxisKey: 'x', yAxisKey: 'y' },
      plugins: {
        legend: { position: 'top', labels: { font: { size: 11 } } },
        tooltip: { callbacks: {
          title: ([item]) => `${item.parsed.x.toFixed(0)}% docs les plus faciles`,
          label: item => `${item.dataset.label}: CER moy = ${item.parsed.y.toFixed(2)}%`,
        } },
      },
      scales: {
        x: { type:'linear', min:0, max:100,
          title: { display:true, text:'% documents (triés par CER croissant)', font:{ size:11 } } },
        y: { min:0, title: { display:true, text:'CER moyen (%)', font:{ size:11 } } },
      },
    },
  });
}

// ── Sprint 7 — Bootstrap CI ──────────────────────────────────────
function buildBootstrapCIChart() {
  const ctx = document.getElementById('chart-bootstrap-ci');
  if (!ctx) return;
  const cis = DATA.statistics && DATA.statistics.bootstrap_cis || [];
  if (!cis.length) { ctx.parentElement.innerHTML = '<p style="color:var(--text-muted);padding:1rem">Données insuffisantes.</p>'; return; }

  const labels = cis.map(c => c.engine);
  const means  = cis.map(c => (c.mean * 100));
  const lowers = cis.map(c => (c.mean - c.ci_lower) * 100);
  const uppers = cis.map(c => (c.ci_upper - c.mean) * 100);

  destroyChart('bootstrap-ci');
  chartInstances['bootstrap-ci'] = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'CER moyen (%)',
        data: means,
        backgroundColor: cis.map((_, i) => engineColor(i) + 'aa'),
        borderColor:     cis.map((_, i) => engineColor(i)),
        borderWidth: 1,
        errorBars: { symmetric: false },
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => {
              const ci = cis[ctx.dataIndex];
              return `IC 95% : [${(ci.ci_lower*100).toFixed(2)}%, ${(ci.ci_upper*100).toFixed(2)}%]`;
            },
          },
        },
      },
      scales: { y: { min: 0, title: { display:true, text:'CER (%)', font:{size:11} } } },
    },
    plugins: [{
      id: 'errorBars',
      afterDatasetsDraw(chart) {
        const { ctx: c, data, scales: { x, y } } = chart;
        chart.data.datasets[0].data.forEach((val, i) => {
          const ci = cis[i];
          if (!ci) return;
          const xPos = x.getPixelForValue(i);
          const yTop = y.getPixelForValue(ci.ci_upper * 100);
          const yBot = y.getPixelForValue(ci.ci_lower * 100);
          c.save();
          c.strokeStyle = '#374151'; c.lineWidth = 2;
          c.beginPath(); c.moveTo(xPos, yTop); c.lineTo(xPos, yBot); c.stroke();
          c.beginPath(); c.moveTo(xPos-6, yTop); c.lineTo(xPos+6, yTop); c.stroke();
          c.beginPath(); c.moveTo(xPos-6, yBot); c.lineTo(xPos+6, yBot); c.stroke();
          c.restore();
        });
      },
    }],
  });
}

// ── Sprint 7 — Diagramme de Venn ────────────────────────────────
function buildVennDiagram() {
  const container = document.getElementById('venn-container');
  if (!container) return;
  const venn = DATA.venn_data;
  if (!venn || !venn.type) {
    container.innerHTML = '<p style="color:var(--text-muted)">Données insuffisantes pour le diagramme de Venn.</p>';
    return;
  }

  if (venn.type === 'venn2') {
    const total = (venn.only_a || 0) + (venn.both || 0) + (venn.only_b || 0);
    const maxR = 80;
    const rA = Math.sqrt((venn.only_a + venn.both) / (total || 1)) * maxR + 30;
    const rB = Math.sqrt((venn.only_b + venn.both) / (total || 1)) * maxR + 30;
    const overlap = venn.both > 0 ? Math.min(rA, rB) * 0.6 : 0;
    const cxA = 140, cxB = cxA + rA + rB - overlap, cy = 130;
    const w = cxB + rB + 20, h = 260;
    container.innerHTML = `
      <div style="text-align:center">
        <svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="max-width:100%">
          <circle cx="${cxA}" cy="${cy}" r="${rA}" fill="#2563eb" fill-opacity="0.25" stroke="#2563eb" stroke-width="2"/>
          <circle cx="${cxB}" cy="${cy}" r="${rB}" fill="#dc2626" fill-opacity="0.25" stroke="#dc2626" stroke-width="2"/>
          <text x="${cxA - rA*0.5}" y="${cy}" text-anchor="middle" font-size="13" font-weight="bold" fill="#1e40af">${venn.only_a}</text>
          <text x="${(cxA + cxB)/2}" y="${cy}" text-anchor="middle" font-size="13" font-weight="bold" fill="#374151">${venn.both}</text>
          <text x="${cxB + rB*0.5}" y="${cy}" text-anchor="middle" font-size="13" font-weight="bold" fill="#b91c1c">${venn.only_b}</text>
          <text x="${cxA - rA*0.5}" y="${cy + rA + 14}" text-anchor="middle" font-size="11" fill="#2563eb">${esc(venn.label_a)}</text>
          <text x="${cxB + rB*0.5}" y="${cy + rB + 14}" text-anchor="middle" font-size="11" fill="#dc2626">${esc(venn.label_b)}</text>
          <text x="${(cxA+cxB)/2}" y="${cy + Math.min(rA,rB) + 14}" text-anchor="middle" font-size="10" fill="#64748b">commun</text>
        </svg>
        <p style="font-size:.75rem;color:var(--text-muted);margin-top:.25rem">
          Erreurs exclusives ${esc(venn.label_a)} : ${venn.only_a} ·
          Communes : ${venn.both} ·
          Exclusives ${esc(venn.label_b)} : ${venn.only_b}
        </p>
      </div>
    `;
  } else if (venn.type === 'venn3') {
    // Venn 3 cercles simplifié
    const total = (venn.only_a||0)+(venn.only_b||0)+(venn.only_c||0)+(venn.ab||0)+(venn.ac||0)+(venn.bc||0)+(venn.abc||0) || 1;
    container.innerHTML = `
      <div style="text-align:center">
        <svg width="300" height="280" viewBox="0 0 300 280" style="max-width:100%">
          <circle cx="130" cy="110" r="80" fill="#2563eb" fill-opacity="0.2" stroke="#2563eb" stroke-width="1.5"/>
          <circle cx="170" cy="110" r="80" fill="#dc2626" fill-opacity="0.2" stroke="#dc2626" stroke-width="1.5"/>
          <circle cx="150" cy="155" r="80" fill="#16a34a" fill-opacity="0.2" stroke="#16a34a" stroke-width="1.5"/>
          <text x="95" y="95" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e40af">${venn.only_a}</text>
          <text x="205" y="95" text-anchor="middle" font-size="12" font-weight="bold" fill="#b91c1c">${venn.only_b}</text>
          <text x="150" y="230" text-anchor="middle" font-size="12" font-weight="bold" fill="#15803d">${venn.only_c}</text>
          <text x="148" y="108" text-anchor="middle" font-size="11" fill="#374151">${venn.ab}</text>
          <text x="120" y="160" text-anchor="middle" font-size="11" fill="#374151">${venn.ac}</text>
          <text x="180" y="160" text-anchor="middle" font-size="11" fill="#374151">${venn.bc}</text>
          <text x="150" y="145" text-anchor="middle" font-size="11" font-weight="bold" fill="#374151">${venn.abc}</text>
          <text x="95" y="127" text-anchor="middle" font-size="9" fill="#2563eb">${esc((venn.label_a||'').slice(0,10))}</text>
          <text x="205" y="127" text-anchor="middle" font-size="9" fill="#dc2626">${esc((venn.label_b||'').slice(0,10))}</text>
          <text x="150" y="248" text-anchor="middle" font-size="9" fill="#16a34a">${esc((venn.label_c||'').slice(0,10))}</text>
        </svg>
      </div>
    `;
  }
}

// ── Sprint 7 — Table de Wilcoxon ─────────────────────────────────
function buildWilcoxonTable() {
  const container = document.getElementById('wilcoxon-table-container');
  if (!container) return;
  const stats = DATA.statistics && DATA.statistics.pairwise_wilcoxon || [];
  if (!stats.length) {
    container.innerHTML = '<p style="color:var(--text-muted)">Pas assez de données pour les tests statistiques (min 2 concurrents).</p>';
    return;
  }
  const rows = stats.map(s => {
    const sigClass = s.significant ? 'stat-sig' : 'stat-ns';
    const sigLabel = s.significant ? '✓ Significative' : '○ Non significative';
    return `<tr>
      <td style="padding:.4rem .6rem;font-weight:600">${esc(s.engine_a)}</td>
      <td style="padding:.4rem .3rem;color:var(--text-muted)">vs</td>
      <td style="padding:.4rem .6rem;font-weight:600">${esc(s.engine_b)}</td>
      <td style="padding:.4rem .6rem;text-align:right;font-variant-numeric:tabular-nums">${s.n_pairs}</td>
      <td style="padding:.4rem .6rem;text-align:right;font-variant-numeric:tabular-nums">${s.statistic}</td>
      <td style="padding:.4rem .6rem;text-align:right;font-variant-numeric:tabular-nums">${s.p_value}</td>
      <td style="padding:.4rem .75rem"><span class="${sigClass}">${sigLabel}</span></td>
      <td style="padding:.4rem .75rem;font-size:.78rem;color:var(--text-muted);max-width:280px">${esc(s.interpretation)}</td>
    </tr>`;
  }).join('');
  container.innerHTML = `
    <table style="border-collapse:collapse;font-size:.84rem;width:100%">
      <thead><tr style="background:var(--bg)">
        <th style="padding:.4rem .6rem;text-align:left;font-size:.75rem;text-transform:uppercase;letter-spacing:.04em">Concurrent A</th>
        <th></th>
        <th style="padding:.4rem .6rem;text-align:left;font-size:.75rem;text-transform:uppercase;letter-spacing:.04em">Concurrent B</th>
        <th style="padding:.4rem .6rem;text-align:right;font-size:.75rem">N paires</th>
        <th style="padding:.4rem .6rem;text-align:right;font-size:.75rem">W</th>
        <th style="padding:.4rem .6rem;text-align:right;font-size:.75rem">p-value</th>
        <th style="padding:.4rem .75rem;text-align:left;font-size:.75rem">Verdict</th>
        <th style="padding:.4rem .75rem;text-align:left;font-size:.75rem">Interprétation</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// ── Sprint 7 — Clustering des erreurs ───────────────────────────
function buildErrorClusters() {
  const container = document.getElementById('error-clusters-container');
  if (!container) return;
  const clusters = DATA.error_clusters || [];
  if (!clusters.length) {
    container.innerHTML = `<p style="color:var(--text-muted)">Aucun cluster d'erreur détecté.</p>`;
    return;
  }
  const cards = clusters.map(cl => {
    const examplesHtml = (cl.examples || []).slice(0, 3).map(ex => {
      const oldStr = ex.gt_fragment || '';
      const newStr = ex.ocr_fragment || '';
      return `<div class="cluster-ex">
        <span class="ex-old">${esc(oldStr || '∅')}</span>
        <span style="color:var(--text-muted)">→</span>
        <span class="ex-new">${esc(newStr || '∅')}</span>
        <span style="color:var(--text-muted);font-size:.72rem">(${esc(ex.engine || '')})</span>
      </div>`;
    }).join('');
    return `<div class="cluster-card">
      <div class="cluster-label">Cluster #${cl.cluster_id} : ${esc(cl.label)}</div>
      <div class="cluster-count">${cl.count} cas détectés</div>
      <div class="cluster-examples">${examplesHtml}</div>
    </div>`;
  }).join('');
  container.innerHTML = `<div class="cluster-grid">${cards}</div>`;
}

// ── Sprint 7 — Matrice de corrélation ───────────────────────────
function initCorrelationMatrix() {
  const sel = document.getElementById('corr-engine-select');
  if (!sel) return;
  const corrs = DATA.correlation_per_engine || [];
  sel.innerHTML = '';
  corrs.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.engine; opt.textContent = c.engine;
    sel.appendChild(opt);
  });
  renderCorrelationMatrix();
}

function renderCorrelationMatrix() {
  const container = document.getElementById('corr-matrix-container');
  if (!container) return;
  const sel = document.getElementById('corr-engine-select');
  const engineName = sel && sel.value;
  const corrs = DATA.correlation_per_engine || [];
  const entry = corrs.find(c => c.engine === engineName) || corrs[0];
  if (!entry || !entry.labels || !entry.matrix) {
    container.innerHTML = '<p style="color:var(--text-muted)">Données insuffisantes.</p>';
    return;
  }
  const labels = entry.labels;
  const matrix = entry.matrix;
  const n = labels.length;

  const labelNames = {
    cer: 'CER', wer: 'WER', mer: 'MER', wil: 'WIL',
    quality_score: 'Qualité img', sharpness: 'Netteté',
    ligature: 'Ligatures', diacritic: 'Diacritiques',
  };
  function corrColor(r) {
    if (r >= 0.7)  return 'background:#dcfce7;color:#14532d';
    if (r >= 0.3)  return 'background:#f0fdf4;color:#166534';
    if (r >= -0.3) return 'background:#f8fafc;color:#374151';
    if (r >= -0.7) return 'background:#fef2f2;color:#991b1b';
    return 'background:#fee2e2;color:#7f1d1d';
  }

  const headerRow = '<tr><th></th>' + labels.map(l =>
    `<th>${esc(labelNames[l] || l)}</th>`).join('') + '</tr>';
  const dataRows = matrix.map((row, i) =>
    '<tr><th style="text-align:right">' + esc(labelNames[labels[i]] || labels[i]) + '</th>' +
    row.map((v, j) => {
      const style = corrColor(v);
      const display = i === j ? '1.00' : v.toFixed(2);
      return `<td style="${style}">${display}</td>`;
    }).join('') + '</tr>'
  ).join('');

  container.innerHTML = `<table class="corr-table"><thead>${headerRow}</thead><tbody>${dataRows}</tbody></table>`;
}

// ── Sprint 7 — URL stateful ──────────────────────────────────────
function updateURL(view, params) {
  const hash = '#' + view + (params ? '?' + new URLSearchParams(params).toString() : '');
  history.replaceState(null, '', hash);
}

function readURLState() {
  const hash = location.hash.slice(1);
  const [view, query] = hash.split('?');
  const params = query ? Object.fromEntries(new URLSearchParams(query)) : {};
  return { view: view || 'ranking', params };
}

// ── Sprint 17 — Aide Critical Difference Diagram ────────────────
function toggleCDDHelp() {
  const el = document.getElementById('cdd-help');
  if (!el) return;
  el.hidden = !el.hidden;
}

// ── Sprint 20 — Glossaire contextuel (panneau latéral) ──────────
function openGlossary(termKey) {
  if (!window.GLOSSARY) return;
  const entry = GLOSSARY[termKey];
  const panel = document.getElementById('glossary-panel');
  const title = document.getElementById('glossary-panel-title');
  const body = document.getElementById('glossary-panel-body');
  if (!panel || !title || !body) return;

  if (!entry) {
    title.textContent = termKey;
    body.innerHTML = '<p class="glossary-empty">' +
      (I18N.glossary_empty || 'Aucune entrée pour ce terme.') + '</p>';
  } else {
    title.textContent = entry.title || termKey;
    body.innerHTML = '';
    const fields = [
      ['definition', I18N.glossary_definition || 'Définition'],
      ['measures',   I18N.glossary_measures   || 'Ce que la métrique mesure'],
      ['usage',      I18N.glossary_usage      || "Cas d'usage"],
      ['limits',     I18N.glossary_limits     || 'Limites'],
      ['reference',  I18N.glossary_reference  || 'Référence'],
    ];
    fields.forEach(([k, label]) => {
      if (!entry[k]) return;
      const h = document.createElement('h4'); h.textContent = label;
      const p = document.createElement('p'); p.textContent = entry[k];
      body.appendChild(h); body.appendChild(p);
    });
  }
  panel.hidden = false;
  panel.setAttribute('aria-hidden', 'false');
  document.body.classList.add('side-panel-open');
}

function closeGlossary() {
  const panel = document.getElementById('glossary-panel');
  if (!panel) return;
  panel.hidden = true;
  panel.setAttribute('aria-hidden', 'true');
  if (!document.querySelector('.side-panel:not([hidden])')) {
    document.body.classList.remove('side-panel-open');
  }
}

function injectGlossaryButtons() {
  if (!window.GLOSSARY) return;
  document.querySelectorAll('th[data-glossary-key]').forEach(th => {
    if (th.querySelector('.glossary-btn')) return;
    const key = th.getAttribute('data-glossary-key');
    if (!GLOSSARY[key]) return;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'glossary-btn';
    btn.textContent = '?';
    btn.title = (I18N.glossary_tooltip || 'Définition');
    btn.setAttribute('aria-label', btn.title);
    btn.onclick = (ev) => {
      ev.stopPropagation();  // ne pas déclencher le tri de colonne
      openGlossary(key);
    };
    th.appendChild(btn);
  });
}

// ── Sprint 20 — Panneau "Mode avancé" (personnalisation) ────────
const _CUSTOM_COLS = [
  'cer', 'cer_diplomatic', 'wer', 'mer', 'wil',
  'ligature_score', 'diacritic_score', 'gini', 'anchor_score',
];
let _CUSTOM_STATE = {
  hiddenColumns: new Set(),
  strataFilter: {},
  weightsEnabled: false,
  weights: {},
};

function openCustomize() {
  const panel = document.getElementById('customize-panel');
  if (!panel) return;
  _populateCustomize();
  panel.hidden = false;
  panel.setAttribute('aria-hidden', 'false');
  document.body.classList.add('side-panel-open');
}

function closeCustomize() {
  const panel = document.getElementById('customize-panel');
  if (!panel) return;
  panel.hidden = true;
  panel.setAttribute('aria-hidden', 'true');
  if (!document.querySelector('.side-panel:not([hidden])')) {
    document.body.classList.remove('side-panel-open');
  }
}

function _populateCustomize() {
  const colList = document.getElementById('customize-columns-list');
  colList.innerHTML = '';
  _CUSTOM_COLS.forEach(col => {
    const label = (I18N['col_' + col] || col);
    const id = 'custom-col-' + col;
    const wrap = document.createElement('label');
    wrap.className = 'custom-col-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.id = id;
    cb.checked = !_CUSTOM_STATE.hiddenColumns.has(col);
    cb.addEventListener('change', () => {
      if (cb.checked) _CUSTOM_STATE.hiddenColumns.delete(col);
      else _CUSTOM_STATE.hiddenColumns.add(col);
      applyColumnVisibility();
      updateCustomURL();
    });
    wrap.appendChild(cb);
    wrap.appendChild(document.createTextNode(' ' + label));
    colList.appendChild(wrap);
  });

  // Strates : détection sur documents[].script_type
  const strata = {};
  (DATA.documents || []).forEach(d => {
    const s = d.script_type;
    if (!s) return;
    strata[s] = (strata[s] || 0) + 1;
  });
  const filtersList = document.getElementById('customize-filters-list');
  filtersList.innerHTML = '';
  const keys = Object.keys(strata);
  if (keys.length === 0) {
    const p = document.createElement('p');
    p.className = 'custom-note';
    p.textContent = I18N.customize_filters_empty ||
      'Aucune strate détectée dans les métadonnées du corpus.';
    filtersList.appendChild(p);
  } else {
    keys.sort().forEach(k => {
      const wrap = document.createElement('label');
      wrap.className = 'custom-col-row';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = _CUSTOM_STATE.strataFilter[k] !== false;
      cb.addEventListener('change', () => {
        _CUSTOM_STATE.strataFilter[k] = cb.checked;
        applyStrataFilter();
        updateCustomURL();
      });
      wrap.appendChild(cb);
      wrap.appendChild(document.createTextNode(
        ' ' + k + ' (' + strata[k] + ')'
      ));
      filtersList.appendChild(wrap);
    });
  }

  _renderCustomWeightsControls();
}

function toggleCustomWeights() {
  _CUSTOM_STATE.weightsEnabled = !_CUSTOM_STATE.weightsEnabled;
  _renderCustomWeightsControls();
  applyCompositeScore();
  updateCustomURL();
}

function _renderCustomWeightsControls() {
  const container = document.getElementById('custom-weights-controls');
  const toggle = document.getElementById('custom-weights-toggle');
  const list = document.getElementById('custom-weights-list');
  const formula = document.getElementById('custom-formula');
  if (!container || !list || !formula || !toggle) return;

  toggle.textContent = _CUSTOM_STATE.weightsEnabled
    ? (I18N.customize_weights_disable || 'Désactiver')
    : (I18N.customize_weights_enable  || 'Activer');
  container.hidden = !_CUSTOM_STATE.weightsEnabled;
  if (!_CUSTOM_STATE.weightsEnabled) return;

  list.innerHTML = '';
  const metrics = ['cer', 'wer', 'mer', 'wil',
                   'ligature_score', 'diacritic_score',
                   'gini', 'anchor_score'];
  metrics.forEach(m => {
    const w = _CUSTOM_STATE.weights[m] || 0;
    const row = document.createElement('div');
    row.className = 'custom-weight-row';
    row.innerHTML = '<span>' + (I18N['col_' + m] || m) + '</span>' +
      '<input type="range" min="0" max="100" step="5" value="' + w * 100 + '" ' +
      'data-metric="' + m + '">' +
      '<output>' + (w * 100).toFixed(0) + ' %</output>';
    const slider = row.querySelector('input');
    const output = row.querySelector('output');
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value) / 100;
      _CUSTOM_STATE.weights[m] = v;
      output.textContent = slider.value + ' %';
      renderCompositeFormula();
      applyCompositeScore();
      updateCustomURL();
    });
    list.appendChild(row);
  });
  renderCompositeFormula();
}

function renderCompositeFormula() {
  const formula = document.getElementById('custom-formula');
  if (!formula) return;
  const terms = [];
  Object.keys(_CUSTOM_STATE.weights).forEach(m => {
    const w = _CUSTOM_STATE.weights[m];
    if (w && w > 0) {
      terms.push(w.toFixed(2) + ' × ' + m);
    }
  });
  if (terms.length === 0) {
    formula.innerHTML = '<em>' + (I18N.customize_weights_none || 'Aucun poids non nul — score composite inactif.') + '</em>';
  } else {
    formula.innerHTML = '<code>score = ' + terms.join(' + ') + '</code>';
  }
}

function applyColumnVisibility() {
  _CUSTOM_COLS.forEach(col => {
    const hidden = _CUSTOM_STATE.hiddenColumns.has(col);
    document.querySelectorAll('th[data-col="' + col + '"], td[data-col="' + col + '"]').forEach(el => {
      el.style.display = hidden ? 'none' : '';
    });
  });
}

function applyStrataFilter() {
  // Effet : cache les documents dont le script_type est désactivé dans la galerie
  const active = new Set(Object.keys(_CUSTOM_STATE.strataFilter)
    .filter(k => _CUSTOM_STATE.strataFilter[k] !== false));
  // Si rien n'est filtré, pas de traitement
  if (active.size === 0) return;
  document.querySelectorAll('.gallery-card').forEach(card => {
    const s = card.dataset.scriptType;
    if (!s) return;
    card.style.display = active.has(s) ? '' : 'none';
  });
}

function applyCompositeScore() {
  const headTh = document.querySelector('th[data-col="composite"]');
  if (!_CUSTOM_STATE.weightsEnabled) {
    // Retirer colonne si présente
    if (headTh) headTh.remove();
    document.querySelectorAll('td[data-col="composite"]').forEach(td => td.remove());
    return;
  }

  const weights = _CUSTOM_STATE.weights;
  const weightKeys = Object.keys(weights).filter(k => weights[k] > 0);
  if (weightKeys.length === 0) {
    if (headTh) headTh.remove();
    document.querySelectorAll('td[data-col="composite"]').forEach(td => td.remove());
    return;
  }

  // Injecter colonne dans le tableau si absente
  const tbl = document.querySelector('#view-ranking table');
  if (!tbl) return;
  const thead = tbl.querySelector('thead tr');
  if (thead && !thead.querySelector('th[data-col="composite"]')) {
    const th = document.createElement('th');
    th.dataset.col = 'composite';
    th.className = 'sortable';
    th.innerHTML = (I18N.customize_composite_col || 'Score') +
      '<i class="sort-icon">↕</i>';
    thead.appendChild(th);
  }

  // Calculer le score pour chaque ligne moteur
  const rows = tbl.querySelectorAll('tbody tr');
  rows.forEach(tr => {
    const name = tr.dataset.engine;
    const engine = (DATA.engines || []).find(e => e.name === name);
    if (!engine) return;
    let score = 0;
    weightKeys.forEach(k => {
      const v = engine[k];
      if (v == null) return;
      // Pour les métriques "plus petit = mieux" (CER, WER…), on inverse
      const invert = ['cer', 'wer', 'mer', 'wil', 'gini', 'length_ratio'].includes(k);
      const val = invert ? (1 - Math.min(1, v)) : v;
      score += weights[k] * val;
    });
    let td = tr.querySelector('td[data-col="composite"]');
    if (!td) {
      td = document.createElement('td');
      td.dataset.col = 'composite';
      tr.appendChild(td);
    }
    td.textContent = score.toFixed(3);
  });
}

function resetCustomization() {
  _CUSTOM_STATE = {
    hiddenColumns: new Set(),
    strataFilter: {},
    weightsEnabled: false,
    weights: {},
  };
  applyColumnVisibility();
  applyStrataFilter();
  applyCompositeScore();
  _populateCustomize();
  updateCustomURL();
}

function updateCustomURL() {
  // Sérialise _CUSTOM_STATE dans l'URL (paramètre ``view`` existant)
  const params = new URLSearchParams(window.location.search);
  const hc = [..._CUSTOM_STATE.hiddenColumns].join(',');
  if (hc) params.set('hidden', hc); else params.delete('hidden');
  const inactive = Object.keys(_CUSTOM_STATE.strataFilter)
    .filter(k => _CUSTOM_STATE.strataFilter[k] === false).join(',');
  if (inactive) params.set('strata_off', inactive);
  else params.delete('strata_off');
  const w = Object.entries(_CUSTOM_STATE.weights)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => k + ':' + v.toFixed(2));
  if (_CUSTOM_STATE.weightsEnabled && w.length) {
    params.set('w', w.join(','));
  } else {
    params.delete('w');
  }
  const newUrl = window.location.pathname + '?' + params.toString() + window.location.hash;
  window.history.replaceState({}, '', newUrl);
}

function restoreCustomFromURL() {
  const params = new URLSearchParams(window.location.search);
  const hc = params.get('hidden');
  if (hc) hc.split(',').filter(Boolean).forEach(c => _CUSTOM_STATE.hiddenColumns.add(c));
  const strataOff = params.get('strata_off');
  if (strataOff) strataOff.split(',').filter(Boolean).forEach(s => {
    _CUSTOM_STATE.strataFilter[s] = false;
  });
  const w = params.get('w');
  if (w) {
    w.split(',').forEach(pair => {
      const [k, v] = pair.split(':');
      const num = parseFloat(v);
      if (k && !isNaN(num) && num > 0) _CUSTOM_STATE.weights[k] = num;
    });
    if (Object.keys(_CUSTOM_STATE.weights).length > 0) {
      _CUSTOM_STATE.weightsEnabled = true;
    }
  }
}

// ── Sprint 19 — Vue Pareto coût/qualité ─────────────────────────
let _paretoChart = null;
let _paretoAxis = 'cost';

function setParetoAxis(axis) {
  _paretoAxis = axis;
  document.querySelectorAll('.pareto-toggle').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.axis === axis);
  });
  renderParetoChart();
  renderParetoAssumptions();
}

function _paretoAxisConfig(axis) {
  const pareto = (DATA.pareto || {})[axis] || {};
  const xKey = axis === 'cost' ? 'cost' : (axis === 'speed' ? 'dur' : 'co2');
  const xLabel = pareto.axis_label ||
    (I18N['pareto_axis_' + axis] || axis);
  return { pareto, xKey, xLabel };
}

function renderParetoChart() {
  const canvas = document.getElementById('pareto-chart');
  if (!canvas || !window.Chart || !DATA.pareto) return;

  const { pareto, xKey, xLabel } = _paretoAxisConfig(_paretoAxis);
  const points = pareto.points || [];
  const frontNames = new Set(pareto.front || []);

  if (_paretoChart) { _paretoChart.destroy(); _paretoChart = null; }
  if (points.length === 0) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#64748b';
    ctx.font = '13px system-ui, sans-serif';
    ctx.fillText(I18N.pareto_empty || 'Données insuffisantes pour cette vue.', 10, 30);
    return;
  }

  const frontPts = points.filter(p => frontNames.has(p.engine));
  const otherPts = points.filter(p => !frontNames.has(p.engine));

  _paretoChart = new Chart(canvas.getContext('2d'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: I18N.pareto_front_label || 'Front Pareto',
          data: frontPts.map(p => ({ x: p[xKey], y: p.cer * 100, engine: p.engine })),
          backgroundColor: '#16a34a',
          borderColor: '#166534',
          pointRadius: 8,
          pointHoverRadius: 10,
        },
        {
          label: I18N.pareto_dominated_label || 'Dominés',
          data: otherPts.map(p => ({ x: p[xKey], y: p.cer * 100, engine: p.engine })),
          backgroundColor: '#94a3b8',
          borderColor: '#64748b',
          pointRadius: 6,
          pointHoverRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: ctx => {
              const p = ctx.raw;
              return p.engine + ' — CER ' + p.y.toFixed(2) + ' %, ' +
                     xLabel + ' : ' + p.x.toFixed(2);
            },
          },
        },
      },
      scales: {
        x: {
          type: _paretoAxis === 'cost' ? 'logarithmic' : 'linear',
          title: { display: true, text: xLabel },
        },
        y: {
          title: { display: true, text: I18N.col_cer || 'CER (%)' },
          ticks: { callback: v => v + ' %' },
        },
      },
    },
  });
}

function renderParetoAssumptions() {
  const ul = document.getElementById('pareto-assumptions-list');
  if (!ul) return;
  ul.innerHTML = '';
  (DATA.engines || []).forEach(e => {
    const c = e.cost || {};
    const parts = [];
    if (c.cost_per_1k_pages_eur != null) {
      parts.push((c.cost_per_1k_pages_eur).toFixed(2) + ' €/1000 pages');
    }
    if (c.type) parts.push(c.type);
    if (c.pricing_source_url) {
      parts.push('<a href="' + c.pricing_source_url + '" target="_blank" rel="noopener">' +
                 (c.pricing_date || 'source') + '</a>');
    }
    const assumptions = (c.assumptions || []).join(' ');
    const li = document.createElement('li');
    li.innerHTML = '<strong>' + e.name + '</strong> — ' + parts.join(' · ') +
                   (assumptions ? ' <em>' + assumptions + '</em>' : '');
    ul.appendChild(li);
  });
}

// ── Sprint 7 — Mode présentation ────────────────────────────────
let presentMode = false;
function togglePresentMode() {
  presentMode = !presentMode;
  document.body.classList.toggle('present-mode', presentMode);
  const btn = document.getElementById('btn-present');
  if (btn) {
    btn.classList.toggle('active', presentMode);
    btn.textContent = presentMode ? '⊡ Normal' : '⊞ Présentation';
  }
}

// ── Sprint 7 — Export CSV ────────────────────────────────────────
function _buildCSVRows(docs) {
  const header = ['doc_id','engine','cer','wer','mer','wil','duration','ligature_score','diacritic_score','difficulty_score','gini','anchor_score','length_ratio','is_hallucinating'];
  const rows = [header];
  docs.forEach(doc => {
    doc.engine_results.forEach(er => {
      rows.push([
        doc.doc_id,
        er.engine,
        er.cer !== null ? (er.cer * 100).toFixed(4) : '',
        er.wer !== null ? (er.wer * 100).toFixed(4) : '',
        er.mer !== null ? (er.mer * 100).toFixed(4) : '',
        er.wil !== null ? (er.wil * 100).toFixed(4) : '',
        er.duration !== null ? er.duration : '',
        er.ligature_score !== null ? er.ligature_score : '',
        er.diacritic_score !== null ? er.diacritic_score : '',
        doc.difficulty_score !== undefined ? (doc.difficulty_score * 100).toFixed(2) : '',
        er.line_metrics ? er.line_metrics.gini.toFixed(6) : '',
        er.hallucination_metrics ? er.hallucination_metrics.anchor_score.toFixed(6) : '',
        er.hallucination_metrics ? er.hallucination_metrics.length_ratio.toFixed(4) : '',
        er.hallucination_metrics ? (er.hallucination_metrics.is_hallucinating ? '1' : '0') : '',
      ]);
    });
  });
  return rows.map(r => r.map(v => JSON.stringify(String(v ?? ''))).join(',')).join('\n');
}

function _downloadCSV(content, filename) {
  const blob = new Blob(['\ufeff' + content], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a); a.click();
  setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 100);
}

function exportCSV() {
  // Feuille 1 : tous les documents
  const corpusSlug = DATA.meta.corpus_name.replace(/\s+/g,'-');
  _downloadCSV(_buildCSVRows(DATA.documents), `picarones_metrics_${corpusSlug}.csv`);

  // Feuille 2 : documents filtrés (exclusions robustes actives)
  const cerThreshold   = parseInt(document.getElementById('robust-cer').value) / 100;
  const anchorThreshold = parseFloat(document.getElementById('robust-anchor').value);
  const ratioThreshold  = parseFloat(document.getElementById('robust-ratio').value);
  const filteredDocs = DATA.documents.filter(doc => {
    // Exclure si doc est dans _manualExclusions
    if (_manualExclusions.has(doc.doc_id)) return false;
    // Exclure si tous les moteurs le détectent comme problématique
    return doc.engine_results.some(er => {
      if (!er || er.error) return false;
      if (cerThreshold < 1.0 && er.cer !== null && er.cer > cerThreshold) return false;
      const hm = er.hallucination_metrics;
      if (hm && hm.anchor_score < anchorThreshold) return false;
      if (hm && hm.length_ratio > ratioThreshold) return false;
      return true;
    });
  });
  // Télécharger avec un délai pour ne pas bloquer le premier download
  setTimeout(() => {
    _downloadCSV(_buildCSVRows(filteredDocs), `picarones_metrics_${corpusSlug}_robust.csv`);
  }, 400);
}

// ── Vue Caractères ───────────────────────────────────────────────
let charViewBuilt = false;

function initCharView() {
  charViewBuilt = true;
  // Remplir le sélecteur de moteur
  const sel = document.getElementById('char-engine-select');
  sel.innerHTML = '';
  DATA.engines.forEach(e => {
    const opt = document.createElement('option');
    opt.value = e.name; opt.textContent = e.name;
    sel.appendChild(opt);
  });
  renderCharView();
}

function renderCharView() {
  const engineName = document.getElementById('char-engine-select').value;
  const eng = DATA.engines.find(e => e.name === engineName);
  if (!eng) return;

  // Scores ligatures / diacritiques
  const scoresRow = document.getElementById('char-scores-row');
  const ligScore = eng.ligature_score;
  const diacScore = eng.diacritic_score;
  scoresRow.innerHTML = `
    <div class="stat">Ligatures <b>${_scoreBadge(ligScore, 'Ligatures')}</b></div>
    <div class="stat">Diacritiques <b>${_scoreBadge(diacScore, 'Diacritiques')}</b></div>
    ${eng.aggregated_structure ? `
    <div class="stat">Précision lignes <b>${_scoreBadge(eng.aggregated_structure.mean_line_accuracy, 'Précision nb lignes')}</b></div>
    <div class="stat">Ordre lecture <b>${_scoreBadge(eng.aggregated_structure.mean_reading_order_score, 'Score ordre de lecture')}</b></div>
    ` : ''}
    ${eng.aggregated_image_quality ? `
    <div class="stat">Qualité image moy. <b>${_scoreBadge(eng.aggregated_image_quality.mean_quality_score, 'Qualité image moyenne')}</b></div>
    ` : ''}
  `;

  // Matrice de confusion heatmap
  renderConfusionHeatmap(eng);

  // Détail ligatures
  renderLigatureDetail(eng);

  // Taxonomie détaillée
  renderTaxonomyDetail(eng);
}

function renderConfusionHeatmap(eng) {
  const container = document.getElementById('confusion-heatmap');
  const cm = eng.aggregated_confusion;
  if (!cm || !cm.matrix) {
    container.innerHTML = '<p style="color:var(--text-muted)">Aucune donnée de confusion disponible.</p>';
    return;
  }

  // Collecter les top confusions (substitutions uniquement, hors ∅)
  const pairs = [];
  for (const [gt, ocrs] of Object.entries(cm.matrix)) {
    if (gt === '∅') continue;
    for (const [ocr, cnt] of Object.entries(ocrs)) {
      if (ocr !== gt && ocr !== '∅' && cnt > 0) {
        pairs.push({ gt, ocr, cnt });
      }
    }
  }
  pairs.sort((a,b) => b.cnt - a.cnt);
  const top = pairs.slice(0, 30);

  if (!top.length) {
    container.innerHTML = '<p style="color:var(--text-muted)">Aucune substitution détectée.</p>';
    return;
  }

  // Heatmap sous forme de tableau compact
  const maxCnt = top[0].cnt;
  const rows = top.map(p => {
    const intensity = Math.round((p.cnt / maxCnt) * 200 + 55);  // 55–255
    const bg = `rgb(${intensity},50,50)`;
    const fg = intensity > 150 ? '#fff' : '#222';
    return `<tr onclick="showConfusionExamples('${esc(p.gt)}','${esc(p.ocr)}')" style="cursor:pointer" title="GT='${esc(p.gt)}' → OCR='${esc(p.ocr)}' : ${p.cnt} fois">
      <td style="font-family:monospace;font-size:1.1rem;padding:.3rem .6rem;text-align:center">${esc(p.gt)}</td>
      <td style="padding:.1rem .3rem;color:var(--text-muted)">→</td>
      <td style="font-family:monospace;font-size:1.1rem;padding:.3rem .6rem;text-align:center">${esc(p.ocr)}</td>
      <td style="padding:.3rem 1rem">
        <div style="display:flex;align-items:center;gap:.5rem">
          <div style="width:${Math.round(p.cnt/maxCnt*120)}px;height:12px;border-radius:3px;background:${bg}"></div>
          <span style="font-size:.8rem;color:var(--text-muted)">${p.cnt}×</span>
        </div>
      </td>
    </tr>`;
  }).join('');

  container.innerHTML = `
    <p style="font-size:.75rem;color:var(--text-muted);margin-bottom:.5rem">
      Cliquer sur une ligne pour voir les exemples dans la vue Document.
      Total substitutions : <b>${cm.total_substitutions}</b>
      · Insertions : <b>${cm.total_insertions}</b>
      · Suppressions : <b>${cm.total_deletions}</b>
    </p>
    <table style="border-collapse:collapse;font-size:.85rem">
      <thead><tr>
        <th style="padding:.3rem .6rem;text-align:left">GT</th>
        <th></th>
        <th style="padding:.3rem .6rem;text-align:left">OCR</th>
        <th style="padding:.3rem 1rem;text-align:left">Fréquence</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function showConfusionExamples(gtChar, ocrChar) {
  // Naviguer vers la vue Document en cherchant un exemple de cette confusion
  showView('document');
  const docWithConfusion = DATA.documents.find(doc =>
    doc.engine_results.some(er => {
      const h = er.hypothesis || '';
      const g = doc.ground_truth || '';
      return g.includes(gtChar) && h.includes(ocrChar);
    })
  );
  if (docWithConfusion) loadDocument(docWithConfusion.doc_id);
}

function renderLigatureDetail(eng) {
  const container = document.getElementById('ligature-detail');
  // Agrégation sur tous les documents pour ce moteur
  const ligData = {};
  DATA.documents.forEach(doc => {
    const er = doc.engine_results.find(r => r.engine === eng.name);
    if (!er || !er.ligature_score) return;
    // On n'a que le score global par doc; pour le détail, utiliser aggregated_char_scores
  });

  const agg = eng.aggregated_char_scores;
  if (!agg || !agg.ligature || !agg.ligature.per_ligature) {
    const overallScore = eng.ligature_score;
    if (overallScore !== null && overallScore !== undefined) {
      container.innerHTML = `<div class="stat">Score global ligatures : ${_scoreBadge(overallScore, 'Ligatures')}</div>`;
    } else {
      container.innerHTML = '<p style="color:var(--text-muted)">Aucune donnée ligature disponible (pas de ligatures dans le corpus).</p>';
    }
    return;
  }

  const perLig = agg.ligature.per_ligature;
  if (!Object.keys(perLig).length) {
    container.innerHTML = '<p style="color:var(--text-muted)">Aucune ligature trouvée dans le corpus GT.</p>';
    return;
  }

  const rows = Object.entries(perLig)
    .sort((a,b) => b[1].gt_count - a[1].gt_count)
    .map(([lig, d]) => {
      const sc = d.score;
      const color = sc >= 0.9 ? '#16a34a' : sc >= 0.7 ? '#ca8a04' : '#dc2626';
      const barW = Math.round(sc * 120);
      return `<tr>
        <td style="font-family:monospace;font-size:1.2rem;padding:.3rem .6rem">${esc(lig)}</td>
        <td style="padding:.3rem .6rem;font-size:.8rem;color:var(--text-muted)">${esc(lig.codePointAt(0).toString(16).toUpperCase().padStart(4,'0'))}</td>
        <td style="padding:.3rem .6rem">${d.gt_count} GT</td>
        <td style="padding:.3rem .6rem">${d.ocr_correct} corrects</td>
        <td style="padding:.3rem 1rem">
          <div style="display:flex;align-items:center;gap:.5rem">
            <div style="width:${barW}px;height:10px;border-radius:3px;background:${color}"></div>
            <span style="color:${color};font-weight:600">${(sc*100).toFixed(0)}%</span>
          </div>
        </td>
      </tr>`;
    }).join('');

  container.innerHTML = `
    <table style="border-collapse:collapse;font-size:.85rem">
      <thead><tr>
        <th style="padding:.3rem .6rem;text-align:left">Ligature</th>
        <th style="padding:.3rem .6rem;text-align:left">Unicode</th>
        <th style="padding:.3rem .6rem">GT</th>
        <th style="padding:.3rem .6rem">Corrects</th>
        <th style="padding:.3rem 1rem;text-align:left">Score</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderTaxonomyDetail(eng) {
  const container = document.getElementById('taxonomy-detail');
  const tax = eng.aggregated_taxonomy;
  if (!tax || !tax.counts) {
    container.innerHTML = '<p style="color:var(--text-muted)">Aucune donnée taxonomique disponible.</p>';
    return;
  }

  const classNames = {
    visual_confusion: '1 — Confusion visuelle',
    diacritic_error: '2 — Erreur diacritique',
    case_error: '3 — Erreur de casse',
    ligature_error: '4 — Ligature',
    abbreviation_error: '5 — Abréviation',
    hapax: '6 — Hapax',
    segmentation_error: '7 — Segmentation',
    oov_character: '8 — Hors-vocabulaire',
    lacuna: '9 — Lacune',
  };
  const total = tax.total_errors || 1;
  const maxCnt = Math.max(...Object.values(tax.counts));

  const rows = Object.entries(tax.counts)
    .filter(([, cnt]) => cnt > 0)
    .sort((a,b) => b[1]-a[1])
    .map(([cls, cnt]) => {
      const pctVal = (cnt / total * 100).toFixed(1);
      const barW = maxCnt > 0 ? Math.round(cnt/maxCnt * 200) : 0;
      return `<tr>
        <td style="padding:.3rem .6rem;font-size:.85rem">${esc(classNames[cls] || cls)}</td>
        <td style="padding:.3rem .6rem;text-align:right;font-variant-numeric:tabular-nums">${cnt}</td>
        <td style="padding:.3rem 1rem">
          <div style="display:flex;align-items:center;gap:.5rem">
            <div style="width:${barW}px;height:10px;border-radius:3px;background:#6366f1"></div>
            <span style="color:var(--text-muted);font-size:.8rem">${pctVal}%</span>
          </div>
        </td>
      </tr>`;
    }).join('');

  container.innerHTML = `
    <p style="font-size:.75rem;color:var(--text-muted);margin-bottom:.5rem">Total : <b>${tax.total_errors}</b> erreurs classifiées.</p>
    <table style="border-collapse:collapse;font-size:.85rem;min-width:400px">
      <thead><tr>
        <th style="padding:.3rem .6rem;text-align:left">Classe</th>
        <th style="padding:.3rem .6rem;text-align:right">N</th>
        <th style="padding:.3rem 1rem;text-align:left">Proportion</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// ── Init ────────────────────────────────────────────────────────
function applyI18n() {
  // Applique les traductions aux éléments avec data-i18n (textContent)
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (I18N[key] !== undefined) el.textContent = I18N[key];
  });
  // Options de select avec data-i18n-opt
  document.querySelectorAll('[data-i18n-opt]').forEach(el => {
    const key = el.getAttribute('data-i18n-opt');
    if (I18N[key] !== undefined) el.textContent = I18N[key];
  });
  // Tooltips des th via id
  const thMap = {
    'th-cer-diplo':  'col_cer_diplo_title',
    'th-ligatures':  'col_ligatures_title',
    'th-diacritics': 'col_diacritics_title',
    'th-gini':       'col_gini_title',
    'th-anchor':     'col_anchor_title',
    'th-overnorm':   'col_overnorm_title',
  };
  Object.entries(thMap).forEach(([id, key]) => {
    const el = document.getElementById(id);
    if (el && I18N[key]) el.title = I18N[key];
  });
}

function init() {
  // i18n
  applyI18n();

  // Méta nav
  const d = new Date(DATA.meta.run_date);
  const locale = I18N.date_locale || 'fr-FR';
  const fmt = d.toLocaleDateString(locale, { year:'numeric', month:'short', day:'numeric' });
  document.getElementById('nav-meta').textContent =
    DATA.meta.corpus_name + ' · ' + fmt;
  document.getElementById('footer-date').textContent =
    (I18N.footer_generated || 'Rapport généré le') + ' ' + fmt;

  // Sélecteur moteur galerie
  const sel = document.getElementById('gallery-engine-select');
  DATA.engines.forEach(e => {
    const opt = document.createElement('option');
    opt.value = e.name; opt.textContent = e.name;
    sel.appendChild(opt);
  });

  renderRanking();
  renderRobustMetrics();
  renderGallery();
  buildDocList();
  renderParetoChart();
  renderParetoAssumptions();
  injectGlossaryButtons();
  restoreCustomFromURL();
  applyColumnVisibility();
  applyStrataFilter();
  applyCompositeScore();

  // Restaurer l'état depuis l'URL
  const { view, params } = readURLState();
  if (view && view !== 'ranking') {
    _switchView(view);  // appel direct pour ne pas écraser l'URL
    if (view === 'document' && params.doc) {
      loadDocument(params.doc);
    }
  }

  // Gérer le bouton retour
  window.addEventListener('popstate', () => {
    const { view: v, params: p } = readURLState();
    _switchView(v || 'ranking');
    if ((v === 'document') && p.doc) loadDocument(p.doc);
  });
}

// ─── Sprint A6 (B-9) — accessibilité des graphiques Chart.js ──────────
//
// Les <canvas> Chart.js ne sont **pas** accessibles aux lecteurs d'écran
// par défaut (le rendu est purement pixel). Pour respecter WCAG 1.1.1
// (Non-text Content) niveau A, on ajoute :
//
// 1. ``role="img"`` + ``aria-label`` (déjà posés statiquement dans le
//    HTML via le helper Python ``_enrich_canvas_with_aria``) ;
// 2. une table de données jumelle générée à la demande à partir de
//    l'instance Chart.js, avec un bouton "Voir les données" qui la
//    révèle pour TOUS (utile aussi pour la copie / vérification).
//
// Cette fonction est idempotente : on peut l'appeler plusieurs fois
// sans dupliquer les boutons (test ``data-a11y-attached``).
function attachChartA11y() {
  const canvases = document.querySelectorAll('canvas[data-a11y-label]');
  canvases.forEach(canvas => {
    if (canvas.dataset.a11yAttached === '1') return;
    canvas.dataset.a11yAttached = '1';

    const id = canvas.id;
    if (!id) return;

    // Bouton "Voir les données" en dessous du canvas.
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn-toggle-data';
    btn.setAttribute('data-i18n', 'view_data');
    btn.textContent = (typeof I18N !== 'undefined' && I18N.view_data)
      ? I18N.view_data : 'Voir les données';
    btn.setAttribute('aria-controls', id + '-data');
    btn.setAttribute('aria-expanded', 'false');

    // Conteneur de table (caché visuellement mais lu par les AT via
    // aria-describedby ; révélé visuellement au clic via .is-revealed).
    const wrapper = document.createElement('div');
    wrapper.id = id + '-data';
    wrapper.className = 'chart-data-table visually-hidden';
    wrapper.setAttribute('role', 'region');
    wrapper.setAttribute('aria-label',
      ((typeof I18N !== 'undefined' && I18N.chart_data_caption)
        ? I18N.chart_data_caption
        : 'Données du graphique')
      + ' : ' + (canvas.dataset.a11yLabel || id));

    // Lien aria-describedby pour que le lecteur d'écran annonce
    // l'existence de la table dès qu'il atteint le canvas.
    canvas.setAttribute('aria-describedby', wrapper.id);

    btn.addEventListener('click', () => {
      const expanded = wrapper.classList.toggle('is-revealed');
      btn.setAttribute('aria-expanded', expanded ? 'true' : 'false');
      btn.textContent = expanded
        ? ((typeof I18N !== 'undefined' && I18N.hide_data) ? I18N.hide_data : 'Masquer les données')
        : ((typeof I18N !== 'undefined' && I18N.view_data) ? I18N.view_data : 'Voir les données');
      // Génération paresseuse du tableau au premier clic.
      if (expanded && !wrapper.dataset.populated) {
        _populateChartDataTable(wrapper, id);
        wrapper.dataset.populated = '1';
      }
    });

    canvas.parentElement.appendChild(btn);
    canvas.parentElement.appendChild(wrapper);
  });
}

function _populateChartDataTable(wrapper, canvasId) {
  const chart = (typeof chartInstances !== 'undefined') ? chartInstances[canvasId] : null;
  if (!chart || !chart.data) {
    wrapper.innerHTML = '<p>' +
      ((typeof I18N !== 'undefined' && I18N.chart_no_data)
        ? I18N.chart_no_data : 'Aucune donnée disponible')
      + '</p>';
    return;
  }
  const labels = chart.data.labels || [];
  const datasets = chart.data.datasets || [];

  // En-tête : colonne libellé puis une colonne par dataset.
  let html = '<table class="chart-data-table is-revealed">';
  html += '<thead><tr><th scope="col">—</th>';
  datasets.forEach(ds => {
    html += '<th scope="col">' + esc(ds.label || '') + '</th>';
  });
  html += '</tr></thead><tbody>';
  // Une ligne par label.
  for (let i = 0; i < labels.length; i++) {
    html += '<tr><th scope="row">' + esc(String(labels[i])) + '</th>';
    datasets.forEach(ds => {
      const v = ds.data ? ds.data[i] : '';
      html += '<td>' + esc(String(v == null ? '' : v)) + '</td>';
    });
    html += '</tr>';
  }
  // Cas particulier : pas de labels (scatter, radar) — on dump les datasets.
  if (labels.length === 0 && datasets.length > 0) {
    datasets.forEach(ds => {
      html += '<tr><th scope="row">' + esc(ds.label || '') + '</th><td>' +
              esc(JSON.stringify(ds.data).slice(0, 200)) + '</td></tr>';
    });
  }
  html += '</tbody></table>';
  wrapper.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', () => {
  init();
  // Délai pour laisser les charts s'instancier au switch de vue.
  // Les boutons sont posés sur les canvas déjà visibles ; pour les
  // canvas qui se créent au premier showView('analyses'), on rappelle
  // attachChartA11y depuis showView aussi.
  setTimeout(attachChartA11y, 200);
});
