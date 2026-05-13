// ─── CSRF — Sprint A4 (B-11) ─────────────────────────────────────────────
//
// Pattern « double-submit cookie » + signature HMAC-SHA256 (cf.
// picarones/web/security.py).  En mode public (PICARONES_CSRF_REQUIRED
// désactivé), le serveur retourne enabled=false et on ne pose aucun
// header — rétrocompat HuggingFace Space.
//
// On wrappe ``fetch`` globalement pour injecter automatiquement
// ``X-CSRF-Token`` sur toutes les méthodes mutantes vers la même origine.
const CSRF_COOKIE = "picarones_csrf";
const CSRF_HEADER = "X-CSRF-Token";
const CSRF_PROTECTED = new Set(["POST", "PUT", "PATCH", "DELETE"]);

function _readCookie(name) {
  const m = document.cookie.match(new RegExp("(^|; )" + name + "=([^;]+)"));
  return m ? decodeURIComponent(m[2]) : null;
}

async function _ensureCsrfToken() {
  if (_readCookie(CSRF_COOKIE)) return _readCookie(CSRF_COOKIE);
  // Mode public : ce GET retourne enabled=false sans poser de cookie.
  // Mode institutionnel : le serveur pose le cookie en réponse.
  try {
    const r = await fetch("/api/csrf/token", {credentials: "same-origin"});
    if (!r.ok) return null;
    const body = await r.json();
    return body.token || null;
  } catch (e) {
    return null;
  }
}

const _origFetch = window.fetch.bind(window);
window.fetch = async function(input, init) {
  init = init || {};
  const method = (init.method || "GET").toUpperCase();
  const url = typeof input === "string" ? input : (input.url || "");
  // Same-origin only (URL relative ou matchant location.origin).
  const isSameOrigin =
    !url.startsWith("http://") && !url.startsWith("https://")
    || url.startsWith(window.location.origin);
  if (CSRF_PROTECTED.has(method) && isSameOrigin) {
    const token = await _ensureCsrfToken();
    if (token) {
      const headers = new Headers(init.headers || {});
      if (!headers.has(CSRF_HEADER)) headers.set(CSRF_HEADER, token);
      init.headers = headers;
    }
  }
  return _origFetch(input, init);
};

// ─── i18n ────────────────────────────────────────────────────────────────────
const T = {
  fr: {
    app_title: "Picarones",
    nav_benchmark: "Benchmark",
    nav_reports: "Rapports",
    nav_engines: "Moteurs",
    nav_import: "Import",
    loading: "Chargement…",
    search: "Rechercher",
    all: "Tous",
    cancel: "Annuler",
    bench_corpus_title: "1. Corpus",
    bench_corpus_label: "Chemin vers le dossier corpus (paires image / .gt.txt)",
    bench_browse: "Parcourir",
    corpus_tab_browse: "📁 Parcourir",
    corpus_tab_upload: "⬆ Uploader",
    upload_zip_mode: "Archive ZIP",
    upload_files_mode: "Fichiers individuels",
    upload_drop_zip: "Glissez un .zip ici ou cliquez pour sélectionner",
    upload_drop_files: "Glissez des images + .gt.txt ou cliquez pour sélectionner",
    upload_uploading: "Upload en cours…",
    upload_success: "Corpus chargé avec succès",
    upload_no_corpus: "Aucun corpus uploadé.",
    upload_select: "Utiliser ce corpus",
    upload_delete: "Supprimer",
    upload_pairs: "paires",
    upload_missing_gt: "GT manquant(s)",
    bench_engines_title: "2. Moteurs et pipelines",
    bench_ocr_title: "2. Moteurs OCR",
    bench_llm_title: "3. Modèles LLM",
    bench_compose_title: "4. Concurrents à benchmarker",
    bench_options_title: "5. Options",
    compose_ocr_only: "OCR seul",
    compose_pipeline: "Pipeline OCR+LLM",
    compose_postcorrection: "Post-correction (corpus OCR)",
    corpus_has_ocr: "Ce corpus contient des fichiers OCR pré-calculés (.ocr.txt) — post-correction disponible.",
    corpus_no_ocr_warn: "Ce corpus ne contient pas de fichiers .ocr.txt — uploadez un corpus triplet pour la post-correction.",
    compose_ocr_engine: "Moteur OCR",
    compose_ocr_model: "Modèle / Langue",
    compose_llm_provider: "Provider LLM",
    compose_llm_model: "Modèle LLM",
    compose_mode: "Mode pipeline",
    compose_prompt: "Prompt",
    compose_add: "+ Ajouter",
    compose_empty: "Aucun concurrent ajouté.",
    mode_text_only: "Post-correction texte",
    mode_text_image: "Post-correction image+texte",
    mode_zero_shot: "Zero-shot",
    bench_norm_label: "Profil de normalisation",
    bench_lang_label: "Langue (Tesseract)",
    bench_output_label: "Dossier de sortie",
    bench_name_label: "Nom du rapport (optionnel)",
    bench_start: "▶ Lancer le benchmark",
    bench_cancel: "✕ Annuler",
    bench_progress_title: "Progression",
    bench_log: "Journal",
    bench_result_title: "Résultats",
    bench_synthesis_title: "Synthèse narrative",
    bench_open_report: "Ouvrir le rapport",
    reports_title: "Rapports générés",
    reports_dir_label: "Dossier de rapports",
    reports_refresh: "Rafraîchir",
    engines_ocr_title: "Moteurs OCR",
    engines_llm_title: "LLMs disponibles",
    import_htr_title: "Import HTR-United",
    import_htr_desc: "Catalogue communautaire de corpus HTR/OCR pour documents patrimoniaux.",
    htr_demo_badge: "Mode démo",
    htr_demo_note: "le catalogue distant est inaccessible ; affichage d'un échantillon embarqué.  Pour le catalogue complet, vérifier la connectivité réseau du serveur.",
    import_hf_title: "Import HuggingFace Datasets",
    import_hf_desc: "Datasets OCR/HTR publics depuis HuggingFace Hub (IAM, RIMES, CATMuS, Gallica…).",
    import_search_label: "Recherche",
    import_lang_filter: "Langue",
    import_script_filter: "Type d'écriture",
    import_tag_filter: "Tags",
    import_modal_title: "Importer le corpus",
    import_output_dir: "Dossier de destination",
    import_max_samples: "Nombre max de documents",
    import_confirm: "Importer",
    available: "disponible",
    not_installed: "non installé",
    configured: "configuré",
    missing_key: "clé manquante",
    running: "actif",
    not_running: "inactif",
    no_reports: "Aucun rapport trouvé.",
    lines: "lignes",
    centuries: "siècles",
  },
  en: {
    app_title: "Picarones",
    nav_benchmark: "Benchmark",
    nav_reports: "Reports",
    nav_engines: "Engines",
    nav_import: "Import",
    loading: "Loading…",
    search: "Search",
    all: "All",
    cancel: "Cancel",
    bench_corpus_title: "1. Corpus",
    bench_corpus_label: "Path to corpus directory (image / .gt.txt pairs)",
    bench_browse: "Browse",
    corpus_tab_browse: "📁 Browse",
    corpus_tab_upload: "⬆ Upload",
    upload_zip_mode: "ZIP archive",
    upload_files_mode: "Individual files",
    upload_drop_zip: "Drop a .zip here or click to select",
    upload_drop_files: "Drop images + .gt.txt files or click to select",
    upload_uploading: "Uploading…",
    upload_success: "Corpus loaded successfully",
    upload_no_corpus: "No corpus uploaded.",
    upload_select: "Use this corpus",
    upload_delete: "Delete",
    upload_pairs: "pairs",
    upload_missing_gt: "missing GT",
    bench_engines_title: "2. Engines & pipelines",
    bench_ocr_title: "2. OCR Engines",
    bench_llm_title: "3. LLM Models",
    bench_compose_title: "4. Competitors",
    bench_options_title: "5. Options",
    compose_ocr_only: "OCR only",
    compose_pipeline: "OCR+LLM Pipeline",
    compose_postcorrection: "Post-correction (corpus OCR)",
    corpus_has_ocr: "This corpus contains pre-computed OCR files (.ocr.txt) — post-correction available.",
    corpus_no_ocr_warn: "This corpus has no .ocr.txt files — upload a triplet corpus for post-correction.",
    compose_ocr_engine: "OCR Engine",
    compose_ocr_model: "Model / Language",
    compose_llm_provider: "LLM Provider",
    compose_llm_model: "LLM Model",
    compose_mode: "Pipeline mode",
    compose_prompt: "Prompt",
    compose_add: "+ Add",
    compose_empty: "No competitors added.",
    mode_text_only: "Text post-correction",
    mode_text_image: "Image+text post-correction",
    mode_zero_shot: "Zero-shot",
    bench_norm_label: "Normalization profile",
    bench_lang_label: "Language (Tesseract)",
    bench_output_label: "Output directory",
    bench_name_label: "Report name (optional)",
    bench_start: "▶ Start benchmark",
    bench_cancel: "✕ Cancel",
    bench_progress_title: "Progress",
    bench_log: "Log",
    bench_result_title: "Results",
    bench_synthesis_title: "Narrative synthesis",
    bench_open_report: "Open report",
    reports_title: "Generated reports",
    reports_dir_label: "Reports directory",
    reports_refresh: "Refresh",
    engines_ocr_title: "OCR Engines",
    engines_llm_title: "Available LLMs",
    import_htr_title: "Import from HTR-United",
    import_htr_desc: "Community catalogue of HTR/OCR datasets for heritage documents.",
    htr_demo_badge: "Demo mode",
    htr_demo_note: "the remote catalogue is unreachable; showing an embedded sample.  For the full catalogue, check the server's network connectivity.",
    import_hf_title: "Import from HuggingFace Datasets",
    import_hf_desc: "Public OCR/HTR datasets from HuggingFace Hub (IAM, RIMES, CATMuS, Gallica…).",
    import_search_label: "Search",
    import_lang_filter: "Language",
    import_script_filter: "Script type",
    import_tag_filter: "Tags",
    import_modal_title: "Import corpus",
    import_output_dir: "Output directory",
    import_max_samples: "Max documents",
    import_confirm: "Import",
    available: "available",
    not_installed: "not installed",
    configured: "configured",
    missing_key: "key missing",
    running: "running",
    not_running: "not running",
    no_reports: "No reports found.",
    lines: "lines",
    centuries: "centuries",
  },
};
let lang = "fr";
function t(key) { return (T[lang][key]) || key; }
function toggleLang() {
  lang = lang === "fr" ? "en" : "fr";
  document.getElementById("lang-btn").textContent = lang === "fr" ? "EN" : "FR";
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const k = el.getAttribute("data-i18n");
    if (T[lang][k]) el.textContent = T[lang][k];
  });
}

// ─── Navigation ──────────────────────────────────────────────────────────────
function showView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  const view = document.getElementById("view-" + name);
  if (view) view.classList.add("active");
  const btns = document.querySelectorAll(".nav-btn");
  const idx = ["benchmark","reports","engines","import"].indexOf(name);
  if (btns[idx]) btns[idx].classList.add("active");

  if (name === "reports") loadReports();
  if (name === "engines") loadEngines();
  if (name === "import") { searchHTRUnited(); searchHuggingFace(); }
}

// ─── Status / version ────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    document.getElementById("app-version").textContent = "v" + d.version;
  } catch(e) {}
}

// ─── Models cache & fetching ─────────────────────────────────────────────────
let _modelsCache = {};
let _enginesData = null;
let _competitors = [];
let _refreshIntervalId = null;
let _pendingOCREngine = null;   // garde contre les réponses obsolètes (race condition)

async function fetchModels(provider, capability) {
  const cacheKey = capability ? `${provider}__${capability}` : provider;
  if (_modelsCache[cacheKey]) return _modelsCache[cacheKey];
  const url = capability ? `/api/models/${provider}?capability=${capability}` : `/api/models/${provider}`;
  const r = await fetch(url);
  const d = await r.json();
  // Support both new format (objects with id+capabilities) and old format (flat strings)
  let models = d.model_ids || d.models || [];
  if (models.length > 0 && typeof models[0] === "object") {
    models = models.map(m => m.id || m);
  }
  _modelsCache[cacheKey] = models;
  return models;
}

function populateSelect(selectId, models, spinnerId) {
  const sel = document.getElementById(selectId);
  if (spinnerId) { const sp = document.getElementById(spinnerId); if (sp) sp.style.display = "none"; }
  if (!sel) return;
  // Handle both string arrays and object arrays
  const items = models.map(m => typeof m === "object" ? (m.id || m) : m);
  sel.innerHTML = items.length === 0
    ? '<option value="">— aucun modèle —</option>'
    : items.map(m => `<option value="${m}">${m}</option>`).join("");
}

// ─── Benchmark sections (OCR + LLM status + composer init) ───────────────────
async function loadBenchmarkSections() {
  try {
    const r = await fetch("/api/engines");
    const d = await r.json();
    _enginesData = d;
    renderOCREnginesSection(d.engines);
    renderLLMSection(d.llms);
  } catch(e) {
    document.getElementById("ocr-engines-status-list").innerHTML =
      `<div style="color:var(--danger);font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

function _makeProviderRow(eng, msId) {
  const dotCls = eng.available ? "status-ok" : (eng.status === "not_running" ? "status-warn" : "status-err");
  let statusLabel;
  if (eng.available) statusLabel = eng.version ? eng.version : (lang === "fr" ? "disponible" : "available");
  else if (eng.status === "missing_key") statusLabel = eng.key_env ? `<code style="font-size:11px;color:var(--warning)">${eng.key_env}</code>` : (lang === "fr" ? "clé manquante" : "key missing");
  else if (eng.status === "not_running") statusLabel = lang === "fr" ? "inactif" : "not running";
  else statusLabel = lang === "fr" ? "non installé" : "not installed";

  const row = document.createElement("div");
  row.className = "provider-row";
  row.innerHTML = `
    <div class="provider-label"><span class="engine-status ${dotCls}"></span><strong>${eng.label}</strong></div>
    <div class="provider-status">${statusLabel}</div>
    <div class="provider-model-select" id="${msId}">${eng.available ? '<span class="spinner"></span>' : ""}</div>`;
  return row;
}

async function renderOCREnginesSection(engines) {
  const container = document.getElementById("ocr-engines-status-list");
  container.innerHTML = "";
  for (const eng of engines) {
    const msId = `ms-ocr-${eng.id}`;
    container.appendChild(_makeProviderRow(eng, msId));
    if (eng.available) {
      fetchModels(eng.id).then(models => {
        const div = document.getElementById(msId);
        if (!div) return;
        div.innerHTML = models.length === 0
          ? `<span style="color:var(--text-muted);font-size:11px;">—</span>`
          : `<span style="font-size:12px;">${models.slice(0,5).join(", ")}${models.length > 5 ? ` +${models.length-5}` : ""}</span>`;
      }).catch(() => {
        const div = document.getElementById(msId);
        if (div) div.innerHTML = `<span style="color:var(--danger);font-size:11px;">Erreur API</span>`;
      });
    }
  }
}

async function renderLLMSection(llms) {
  const container = document.getElementById("llm-status-list");
  container.innerHTML = "";
  for (const llm of llms) {
    const msId = `ms-llm-${llm.id}`;
    container.appendChild(_makeProviderRow(llm, msId));
    if (llm.available) {
      fetchModels(llm.id).then(models => {
        const div = document.getElementById(msId);
        if (!div) return;
        div.innerHTML = models.length === 0
          ? `<span style="color:var(--text-muted);font-size:11px;">—</span>`
          : `<span style="font-size:12px;">${models.slice(0,3).join(", ")}${models.length > 3 ? ` +${models.length-3}` : ""}</span>`;
      }).catch(() => {
        const div = document.getElementById(msId);
        if (div) div.innerHTML = `<span style="color:var(--danger);font-size:11px;">Erreur API</span>`;
      });
    }
  }
}

function startAutoRefresh() {
  if (_refreshIntervalId) clearInterval(_refreshIntervalId);
  _refreshIntervalId = setInterval(async () => {
    try {
      const r = await fetch("/api/engines");
      const d = await r.json();
      if (!_enginesData || JSON.stringify(d) !== JSON.stringify(_enginesData)) {
        _modelsCache = {};
        _enginesData = d;
        renderOCREnginesSection(d.engines);
        renderLLMSection(d.llms);
      }
    } catch(e) {}
  }, 10000);
}

// ─── Competitor composer ──────────────────────────────────────────────────────
async function onComposeOCRChange() {
  const engine = document.getElementById("compose-ocr-engine").value;
  _pendingOCREngine = engine;   // marquer la requête courante
  const sp = document.getElementById("sp-ocr-model");
  // Google Vision et Azure ont des listes statiques — pas d'appel API nécessaire
  if (engine === "google_vision") {
    sp.style.display = "none";
    populateSelect("compose-ocr-model", ["document_text_detection", "text_detection"], null);
    return;
  }
  if (engine === "azure_doc_intel") {
    sp.style.display = "none";
    populateSelect("compose-ocr-model", ["prebuilt-document", "prebuilt-read"], null);
    return;
  }
  // Tesseract : langues installées ; Mistral OCR : modèles vision (API dynamique)
  sp.style.display = "inline-block";
  try {
    const models = await fetchModels(engine);
    if (_pendingOCREngine !== engine) return;  // réponse obsolète, abandonner
    populateSelect("compose-ocr-model", models, "sp-ocr-model");
  } catch(e) {
    if (_pendingOCREngine !== engine) return;
    sp.style.display = "none";
    document.getElementById("compose-ocr-model").innerHTML = '<option value="">Erreur</option>';
  }
}

async function onComposeLLMChange() {
  const provider = document.getElementById("compose-llm-provider").value;
  const composeMode = document.querySelector("input[name=compose-mode]:checked").value;
  const pipelineMode = document.getElementById("compose-pipeline-mode").value;
  // Apply capability filter for modes requiring vision
  const needsVision = (pipelineMode === "text_and_image" || pipelineMode === "zero_shot");
  const capability = (composeMode === "postcorrection" || composeMode === "pipeline") && needsVision ? "vision" : "";
  _loadLLMModelsWithCapability(provider, capability);
}

function onComposeModeChange() {
  const mode = document.querySelector("input[name=compose-mode]:checked").value;
  const ocrSection = document.getElementById("compose-ocr-section");
  const pipelineSection = document.getElementById("compose-pipeline-section");

  if (mode === "ocr") {
    ocrSection.style.display = "flex";
    pipelineSection.style.display = "none";
  } else if (mode === "pipeline") {
    ocrSection.style.display = "flex";
    pipelineSection.style.display = "block";
    // Reload LLM models without capability filter
    onComposeLLMChange();
  } else if (mode === "postcorrection") {
    ocrSection.style.display = "none";
    pipelineSection.style.display = "block";
    // Reload LLM models with capability filter based on pipeline mode
    onComposePipelineModeChange();
  }
}

function onComposePipelineModeChange() {
  const composeMode = document.querySelector("input[name=compose-mode]:checked").value;
  if (composeMode !== "postcorrection" && composeMode !== "pipeline") return;
  const pipelineMode = document.getElementById("compose-pipeline-mode").value;
  // Filter by vision capability for modes that need images
  const needsVision = (pipelineMode === "text_and_image" || pipelineMode === "zero_shot");
  const capability = needsVision ? "vision" : "";
  const provider = document.getElementById("compose-llm-provider").value;
  // Clear cache for this provider to re-fetch with new capability filter
  const cacheKey = capability ? `${provider}__${capability}` : provider;
  delete _modelsCache[cacheKey];
  _loadLLMModelsWithCapability(provider, capability);
}

async function _loadLLMModelsWithCapability(provider, capability) {
  document.getElementById("sp-llm-model").style.display = "inline-block";
  try {
    const models = await fetchModels(provider, capability);
    populateSelect("compose-llm-model", models, "sp-llm-model");
  } catch(e) {
    document.getElementById("sp-llm-model").style.display = "none";
    document.getElementById("compose-llm-model").innerHTML = '<option value="">Erreur</option>';
  }
}

async function loadComposePrompts() {
  document.getElementById("sp-prompt").style.display = "inline-block";
  try {
    const models = await fetchModels("prompts");
    populateSelect("compose-prompt", models, "sp-prompt");
  } catch(e) {
    document.getElementById("sp-prompt").style.display = "none";
  }
}

function addCompetitor() {
  const mode = document.querySelector("input[name=compose-mode]:checked").value;
  const errEl = document.getElementById("compose-error");

  const comp = { name: "", engine_name: "", ocr_model: "",
                  llm_provider: "", llm_model: "", pipeline_mode: "", prompt_file: "" };

  if (mode === "postcorrection") {
    // Post-correction : OCR vient du corpus (.ocr.txt)
    comp.engine_name = "corpus";
    comp.llm_provider = document.getElementById("compose-llm-provider").value;
    comp.llm_model = document.getElementById("compose-llm-model").value;
    comp.pipeline_mode = document.getElementById("compose-pipeline-mode").value;
    comp.prompt_file = document.getElementById("compose-prompt").value;
    if (!comp.llm_provider || !comp.llm_model) {
      errEl.textContent = lang === "fr" ? "Sélectionnez un provider et un modèle LLM." : "Select an LLM provider and model.";
      return;
    }
    const modeLabel = {"text_only":"texte","text_and_image":"img+texte","zero_shot":"zero-shot"}[comp.pipeline_mode] || comp.pipeline_mode;
    comp.name = `📝 ${comp.llm_model} [${modeLabel}]`;
  } else if (mode === "pipeline") {
    const ocrEngine = document.getElementById("compose-ocr-engine").value;
    const ocrModel = document.getElementById("compose-ocr-model").value;
    if (!ocrEngine) {
      errEl.textContent = lang === "fr" ? "Sélectionnez un moteur OCR." : "Select an OCR engine.";
      return;
    }
    comp.engine_name = ocrEngine;
    comp.ocr_model = ocrModel;
    comp.llm_provider = document.getElementById("compose-llm-provider").value;
    comp.llm_model = document.getElementById("compose-llm-model").value;
    comp.pipeline_mode = document.getElementById("compose-pipeline-mode").value;
    comp.prompt_file = document.getElementById("compose-prompt").value;
    if (!comp.llm_provider) {
      errEl.textContent = lang === "fr" ? "Sélectionnez un provider LLM." : "Select an LLM provider.";
      return;
    }
    comp.name = `${ocrEngine}${ocrModel ? ":"+ocrModel : ""} → ${comp.llm_model || comp.llm_provider}`;
  } else {
    // OCR seul
    const ocrEngine = document.getElementById("compose-ocr-engine").value;
    const ocrModel = document.getElementById("compose-ocr-model").value;
    if (!ocrEngine) {
      errEl.textContent = lang === "fr" ? "Sélectionnez un moteur OCR." : "Select an OCR engine.";
      return;
    }
    comp.engine_name = ocrEngine;
    comp.ocr_model = ocrModel;
    comp.name = `${ocrEngine}${ocrModel ? " ("+ocrModel+")" : ""}`;
  }

  errEl.textContent = "";
  _competitors.push(comp);
  renderCompetitors();
}

function removeCompetitor(idx) {
  _competitors.splice(idx, 1);
  renderCompetitors();
}

function renderCompetitors() {
  const container = document.getElementById("competitors-list");
  if (_competitors.length === 0) {
    container.innerHTML = `<div style="color:var(--text-muted);font-size:12px;">${t("compose_empty")}</div>`;
    return;
  }
  container.innerHTML = _competitors.map((c, i) => {
    const isCorpusOCR = c.engine_name === "corpus" || (c.engine_name === "" && c.llm_provider);
    const isPipeline = !!c.llm_provider && !isCorpusOCR;
    let badge, detail;
    if (isCorpusOCR) {
      badge = "📝 Post-correction";
      detail = `corpus_ocr → ${c.llm_provider}:${c.llm_model} [${c.pipeline_mode}]`;
    } else if (isPipeline) {
      badge = "⛓ Pipeline";
      detail = `${c.engine_name}:${c.ocr_model} → ${c.llm_provider}:${c.llm_model} [${c.pipeline_mode}]`;
    } else {
      badge = "🔍 OCR";
      detail = `${c.engine_name}:${c.ocr_model}`;
    }
    return `<div class="competitor-card">
      <div class="competitor-info">
        <span class="competitor-badge">${badge}</span>
        <span class="competitor-name">${c.name}</span>
        <span class="competitor-detail">${detail}</span>
      </div>
      <button class="btn btn-danger btn-sm" onclick="removeCompetitor(${i})">✕</button>
    </div>`;
  }).join("");
}

// ─── Normalization profiles ──────────────────────────────────────────────────
let _normProfilesData = [];
async function loadNormProfiles() {
  try {
    const r = await fetch("/api/normalization/profiles");
    const d = await r.json();
    _normProfilesData = d.profiles || [];
    const sel = document.getElementById("norm-profile");
    sel.innerHTML = "";
    _normProfilesData.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${p.name} — ${p.description}`;
      if (p.id === "nfc") opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => {
      const p = _normProfilesData.find(x => x.id === sel.value);
      if (p && p.exclude_chars && p.exclude_chars.length) {
        document.getElementById("char-exclude").value = p.exclude_chars.join(", ");
      }
    });
  } catch(e) {}
}

// ─── File browser ────────────────────────────────────────────────────────────
let _fbVisible = false;
function openFileBrowser() {
  _fbVisible = !_fbVisible;
  const c = document.getElementById("file-browser-container");
  c.style.display = _fbVisible ? "block" : "none";
  if (_fbVisible) browsePath(".");
}
async function browsePath(path) {
  try {
    const r = await fetch(`/api/corpus/browse?path=${encodeURIComponent(path)}`);
    const d = await r.json();
    document.getElementById("fb-current-path").textContent = d.current_path;
    const fb = document.getElementById("file-browser");
    fb.innerHTML = "";
    if (d.parent_path) {
      const up = document.createElement("div");
      up.className = "fb-item";
      up.innerHTML = `<span class="fb-icon">⬆</span><span class="fb-name">..</span>`;
      up.onclick = () => browsePath(d.parent_path);
      fb.appendChild(up);
    }
    d.items.filter(i => i.is_dir).forEach(item => {
      const el = document.createElement("div");
      el.className = "fb-item";
      const hasCorpus = item.has_corpus ? `<span class="fb-badge" style="color:var(--success)">✓ ${item.gt_count} GT</span>` : "";
      el.innerHTML = `<span class="fb-icon">📁</span><span class="fb-name">${item.name}</span>${hasCorpus}`;
      el.onclick = () => {
        if (item.has_corpus) {
          document.getElementById("corpus-path").value = item.path;
          document.getElementById("corpus-info").textContent = `✓ ${item.gt_count} documents GT trouvés.`;
          _fbVisible = false;
          document.getElementById("file-browser-container").style.display = "none";
        } else {
          browsePath(item.path);
        }
      };
      fb.appendChild(el);
    });
    if (fb.children.length === 0) {
      fb.innerHTML = '<div style="padding:12px; color: var(--text-muted); font-size:12px;">Dossier vide</div>';
    }
  } catch(e) {
    document.getElementById("file-browser").innerHTML =
      `<div style="padding:12px; color: var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

// ─── Benchmark ───────────────────────────────────────────────────────────────
let _currentJobId = null;
let _eventSource = null;

async function startBenchmark() {
  const corpusPath = document.getElementById("corpus-path").value.trim();
  if (!corpusPath) {
    alert(lang === "fr" ? "Veuillez sélectionner un dossier corpus." : "Please select a corpus directory.");
    return;
  }
  if (_competitors.length === 0) {
    alert(lang === "fr" ? "Ajoutez au moins un concurrent (Section 4)." : "Add at least one competitor (Section 4).");
    return;
  }

  const payload = {
    corpus_path: corpusPath,
    competitors: _competitors,
    normalization_profile: document.getElementById("norm-profile").value,
    char_exclude: document.getElementById("char-exclude").value.trim(),
    output_dir: document.getElementById("output-dir").value,
    report_name: document.getElementById("report-name").value,
  };

  document.getElementById("start-btn").disabled = true;
  document.getElementById("cancel-btn").style.display = "inline-flex";
  document.getElementById("bench-progress-section").style.display = "block";
  document.getElementById("bench-result-section").style.display = "none";
  document.getElementById("bench-log").textContent = "";
  document.getElementById("engine-progress-list").innerHTML = "";
  document.getElementById("bench-status-text").textContent = lang === "fr" ? "Démarrage…" : "Starting…";

  try {
    const r = await fetch("/api/benchmark/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || "Erreur serveur");
    }
    const d = await r.json();
    _currentJobId = d.job_id;
    _startSSE(_currentJobId);
  } catch(e) {
    appendLog(`Erreur : ${e.message}`, "error");
    document.getElementById("start-btn").disabled = false;
    document.getElementById("cancel-btn").style.display = "none";
    document.getElementById("bench-status-text").textContent = "";
  }
}

function _startSSE(jobId) {
  if (_eventSource) _eventSource.close();
  const pl = document.getElementById("engine-progress-list");
  pl.innerHTML = "";
  const seenEngines = {};

  _eventSource = new EventSource(`/api/benchmark/${jobId}/stream`);

  _eventSource.addEventListener("start", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message, "success");
    document.getElementById("bench-status-text").textContent = lang === "fr" ? "En cours…" : "Running…";
  });

  _eventSource.addEventListener("log", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message);
  });

  _eventSource.addEventListener("warning", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message, "warn");
  });

  _eventSource.addEventListener("progress", e => {
    const d = JSON.parse(e.data);
    const pct = Math.round(d.progress * 100);
    const engId = d.engine.replace(/[^a-z0-9_-]/gi, "_");
    if (!seenEngines[engId]) {
      seenEngines[engId] = true;
      const div = document.createElement("div");
      div.style = "margin-bottom: 8px;";
      div.innerHTML = `<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
        <span>${d.engine}</span><span id="eng-pct-${engId}">0%</span></div>
        <div class="progress-bar-outer"><div class="progress-bar-inner" id="eng-bar-${engId}" style="width:0%"></div></div>`;
      pl.appendChild(div);
    }
    const bar = document.getElementById(`eng-bar-${engId}`);
    const pctEl = document.getElementById(`eng-pct-${engId}`);
    if (bar) bar.style.width = pct + "%";
    if (pctEl) pctEl.textContent = pct + "%";
    document.getElementById("bench-status-text").textContent =
      `${pct}% — ${d.engine} (${d.processed}/${d.total})`;
  });

  _eventSource.addEventListener("complete", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message, "success");
    _showResults(d);
    _finishBenchmark();
  });

  _eventSource.addEventListener("error", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message, "error");
    _finishBenchmark();
  });

  _eventSource.addEventListener("cancelled", e => {
    appendLog(lang === "fr" ? "Benchmark annulé." : "Benchmark cancelled.", "warn");
    _finishBenchmark();
  });

  _eventSource.addEventListener("done", e => { _finishBenchmark(); });
  _eventSource.onerror = () => { if (_currentJobId) _finishBenchmark(); };
}

function _showResults(data) {
  const section = document.getElementById("bench-result-section");
  section.style.display = "block";
  if (data.output_html) {
    const link = document.getElementById("bench-report-link");
    link.href = `/reports/${data.output_html.split("/").pop()}`;
  }
  if (data.ranking) {
    let html = `<table><thead><tr><th>#</th><th>${lang==="fr"?"Moteur":"Engine"}</th><th>CER</th><th>WER</th><th>${lang==="fr"?"Docs":"Docs"}</th></tr></thead><tbody>`;
    data.ranking.forEach((row, i) => {
      const cer = row.mean_cer != null ? (row.mean_cer*100).toFixed(2)+"%" : "N/A";
      const wer = row.mean_wer != null ? (row.mean_wer*100).toFixed(2)+"%" : "N/A";
      html += `<tr><td>${i+1}</td><td>${row.engine}</td><td>${cer}</td><td>${wer}</td><td>${row.total_docs || ""}</td></tr>`;
    });
    html += "</tbody></table>";
    document.getElementById("bench-ranking-table").innerHTML = html;
  }
  // Phase 6 chantier post-rewrite : appel à
  // /api/benchmark/{job_id}/synthesis_preview pour afficher la
  // synthèse narrative (moteur narratif côté serveur) sans avoir à
  // ouvrir le rapport HTML.  Avant : endpoint existait + testé serveur
  // mais zéro appel depuis l'UI (code zombie typique post-rewrite).
  if (_currentJobId) {
    _loadSynthesisPreview(_currentJobId);
  }
}

async function _loadSynthesisPreview(jobId) {
  /** GET /api/benchmark/{jobId}/synthesis_preview et injecte les
   * phrases dans #bench-synthesis-sentences.  En cas d'erreur (job
   * sans synthèse, JSON manquant, narratif indisponible) on masque
   * la section silencieusement — la synthèse est un bonus, pas un
   * bloquant. */
  const section = document.getElementById("bench-synthesis-section");
  const list = document.getElementById("bench-synthesis-sentences");
  if (!section || !list) return;
  section.style.display = "none";
  list.innerHTML = "";
  try {
    const r = await fetch(
      `/api/benchmark/${encodeURIComponent(jobId)}/synthesis_preview?lang=${encodeURIComponent(lang)}`,
    );
    if (!r.ok) return;
    const d = await r.json();
    const sentences = Array.isArray(d.sentences) ? d.sentences : [];
    if (sentences.length === 0) return;
    list.innerHTML = sentences
      .map(s => `<li>${_escapeHtml(String(s))}</li>`)
      .join("");
    section.style.display = "block";
  } catch (e) {
    // Synthèse optionnelle — on n'ennuie pas l'utilisateur.
  }
}

function _escapeHtml(s) {
  /** Helper local : on injecte les phrases dans innerHTML donc il
   * faut neutraliser les balises HTML potentielles (les phrases
   * narratives peuvent contenir des noms de moteurs avec ``<`` ou ``>``
   * théoriquement). */
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function _finishBenchmark() {
  if (_eventSource) { _eventSource.close(); _eventSource = null; }
  document.getElementById("start-btn").disabled = false;
  document.getElementById("cancel-btn").style.display = "none";
  document.getElementById("bench-status-text").textContent = "";
}

async function cancelBenchmark() {
  if (!_currentJobId) return;
  await fetch(`/api/benchmark/${_currentJobId}/cancel`, {method: "POST"});
}

function appendLog(msg, cls) {
  const box = document.getElementById("bench-log");
  const line = document.createElement("div");
  if (cls === "error") line.className = "log-error";
  else if (cls === "warn") line.className = "log-warn";
  else if (cls === "success") line.className = "log-success";
  line.textContent = msg;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

// ─── Reports ─────────────────────────────────────────────────────────────────
async function loadReports() {
  const dir = document.getElementById("reports-dir").value || ".";
  const container = document.getElementById("reports-list");
  container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${t("loading")}</div>`;
  try {
    const r = await fetch(`/api/reports?reports_dir=${encodeURIComponent(dir)}`);
    const d = await r.json();
    if (d.reports.length === 0) {
      container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${t("no_reports")}</div>`;
      return;
    }
    let html = `<table><thead><tr><th>${lang==="fr"?"Fichier":"File"}</th><th>${lang==="fr"?"Taille":"Size"}</th><th>${lang==="fr"?"Modifié":"Modified"}</th><th></th></tr></thead><tbody>`;
    d.reports.forEach(rep => {
      const date = new Date(rep.modified).toLocaleString(lang === "fr" ? "fr-FR" : "en-US");
      html += `<tr><td>${rep.filename}</td><td>${rep.size_kb} Ko</td><td>${date}</td>
        <td><a href="${rep.url}" target="_blank" class="btn btn-primary btn-sm">${lang==="fr"?"Ouvrir":"Open"}</a></td></tr>`;
    });
    html += "</tbody></table>";
    container.innerHTML = html;
  } catch(e) {
    container.innerHTML = `<div style="color: var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

// ─── Engines status ──────────────────────────────────────────────────────────
async function loadEngines() {
  try {
    const r = await fetch("/api/engines");
    const d = await r.json();

    // OCR
    let html = `<table><thead><tr><th>ID</th><th>${lang==="fr"?"Nom":"Name"}</th><th>Version</th><th>Statut</th></tr></thead><tbody>`;
    d.engines.forEach(e => {
      const cls = e.available ? "badge-ok" : "badge-err";
      const lbl = e.available ? t("available") : t("not_installed");
      html += `<tr><td><code>${e.id}</code></td><td>${e.label}</td><td>${e.version||"—"}</td>
        <td><span class="badge ${cls}">${lbl}</span></td></tr>`;
    });
    html += "</tbody></table>";
    document.getElementById("engines-ocr-list").innerHTML = html;

    // LLMs
    let llmHtml = `<table><thead><tr><th>ID</th><th>${lang==="fr"?"Nom":"Name"}</th><th>Statut</th><th>${lang==="fr"?"Détail":"Detail"}</th></tr></thead><tbody>`;
    d.llms.forEach(e => {
      const cls = e.available ? "badge-ok" : "badge-warn";
      const statusKey = e.status === "configured" ? "configured"
        : e.status === "running" ? "running"
        : e.status === "not_running" ? "not_running"
        : "missing_key";
      const lbl = t(statusKey);
      let detail = "";
      if (e.key_env) detail = `<code style="font-size:11px;">${e.key_env}</code>`;
      if (e.models && e.models.length > 0) detail = e.models.slice(0, 3).join(", ");
      llmHtml += `<tr><td><code>${e.id}</code></td><td>${e.label}</td>
        <td><span class="badge ${cls}">${lbl}</span></td><td>${detail}</td></tr>`;
    });
    llmHtml += "</tbody></table>";
    document.getElementById("engines-llm-list").innerHTML = llmHtml;
  } catch(e) {
    document.getElementById("engines-ocr-list").innerHTML =
      `<div style="color: var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

// ─── HTR-United ──────────────────────────────────────────────────────────────
function _updateHtrDemoBanner(isDemo) {
  /** Affiche / masque le bandeau "Mode démo" sous le titre HTR-United.
   *
   * Phase 4.4 du chantier post-rewrite : l'endpoint
   * ``/api/htr-united/catalogue`` retourne désormais le champ
   * ``is_demo`` (``true`` quand le serveur ne peut pas joindre le
   * catalogue distant et fallback sur l'échantillon embarqué).  Avant,
   * l'UI annonçait "Catalogue HTR-United" sans distinguer mode démo
   * vs catalogue complet, vecteur de confusion utilisateur. */
  const el = document.getElementById("htr-demo-banner");
  if (!el) return;
  el.style.display = isDemo ? "block" : "none";
}

async function initHTRFilters() {
  try {
    const r = await fetch("/api/htr-united/catalogue");
    const d = await r.json();
    _updateHtrDemoBanner(Boolean(d.is_demo));
    const langSel = document.getElementById("htr-lang-filter");
    const scriptSel = document.getElementById("htr-script-filter");
    langSel.innerHTML = `<option value="">${t("all")}</option>`;
    d.available_languages.forEach(l => {
      langSel.innerHTML += `<option value="${l}">${l}</option>`;
    });
    scriptSel.innerHTML = `<option value="">${t("all")}</option>`;
    d.available_scripts.forEach(s => {
      scriptSel.innerHTML += `<option value="${s}">${s}</option>`;
    });
  } catch(e) {}
}

async function searchHTRUnited() {
  const q = document.getElementById("htr-search").value;
  const lang2 = document.getElementById("htr-lang-filter").value;
  const script = document.getElementById("htr-script-filter").value;
  const container = document.getElementById("htr-results");
  container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${t("loading")}</div>`;
  try {
    const url = `/api/htr-united/catalogue?query=${encodeURIComponent(q)}&language=${encodeURIComponent(lang2)}&script=${encodeURIComponent(script)}`;
    const r = await fetch(url);
    const d = await r.json();
    _updateHtrDemoBanner(Boolean(d.is_demo));
    if (d.entries.length === 0) {
      container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${lang==="fr"?"Aucun résultat.":"No results."}</div>`;
      return;
    }
    container.innerHTML = d.entries.map(e => {
      const tags = [...e.language, ...e.script].map(s => `<span class="ds-tag">${s}</span>`).join("");
      return `<div class="ds-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
          <h4>${e.title}</h4>
          <button class="btn btn-primary btn-sm" onclick="openImportModal('htr', '${e.id}', '${e.title.replace(/'/g,"\\'")}')">
            ${lang==="fr"?"Importer":"Import"}
          </button>
        </div>
        <p>${e.description}</p>
        <p style="color: var(--text-muted);">${e.institution} — ${e.lines.toLocaleString()} ${t("lines")} — ${e.format}</p>
        <div class="ds-meta">${tags}</div>
      </div>`;
    }).join("");
  } catch(e) {
    container.innerHTML = `<div style="color: var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

async function searchHuggingFace() {
  const q = document.getElementById("hf-search").value;
  const langFilter = document.getElementById("hf-lang-filter").value;
  const tags = document.getElementById("hf-tags").value;
  const container = document.getElementById("hf-results");
  container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${t("loading")}</div>`;
  try {
    const url = `/api/huggingface/search?query=${encodeURIComponent(q)}&language=${encodeURIComponent(langFilter)}&tags=${encodeURIComponent(tags)}`;
    const r = await fetch(url);
    const d = await r.json();
    if (d.datasets.length === 0) {
      container.innerHTML = `<div style="color: var(--text-muted); font-size:12px;">${lang==="fr"?"Aucun résultat.":"No results."}</div>`;
      return;
    }
    container.innerHTML = d.datasets.map(ds => {
      const tags2 = ds.tags.slice(0,5).map(s => `<span class="ds-tag">${s}</span>`).join("");
      return `<div class="ds-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
          <h4>${ds.title}</h4>
          <button class="btn btn-primary btn-sm" onclick="openImportModal('hf', '${ds.dataset_id.replace(/'/g,"\\'")}', '${ds.title.replace(/'/g,"\\'")}')">
            ${lang==="fr"?"Importer":"Import"}
          </button>
        </div>
        <p>${ds.description}</p>
        <p style="color: var(--text-muted);">${ds.institution||ds.dataset_id} ${ds.downloads ? "— " + ds.downloads.toLocaleString() + " téléchargements" : ""}</p>
        <div class="ds-meta">${tags2}</div>
      </div>`;
    }).join("");
  } catch(e) {
    container.innerHTML = `<div style="color: var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

// ─── Import modal ─────────────────────────────────────────────────────────────
function openImportModal(type, id, title) {
  document.getElementById("import-modal-type").value = type;
  document.getElementById("import-modal-id").value = id;
  document.getElementById("import-modal-title").textContent = `${t("import_modal_title")} : ${title}`;
  document.getElementById("import-modal-status").innerHTML = "";
  document.getElementById("import-modal").style.display = "flex";
}
function closeImportModal() {
  document.getElementById("import-modal").style.display = "none";
}
async function confirmImport() {
  const type = document.getElementById("import-modal-type").value;
  const id = document.getElementById("import-modal-id").value;
  const outputDir = document.getElementById("import-modal-output").value;
  const maxSamples = parseInt(document.getElementById("import-modal-max").value);
  const statusDiv = document.getElementById("import-modal-status");
  statusDiv.innerHTML = `<div class="alert alert-info"><span class="spinner"></span> ${lang==="fr"?"Import en cours…":"Importing…"}</div>`;

  try {
    let url, body;
    if (type === "htr") {
      url = "/api/htr-united/import";
      body = {entry_id: id, output_dir: outputDir, max_samples: maxSamples};
    } else {
      url = "/api/huggingface/import";
      body = {dataset_id: id, output_dir: outputDir, max_samples: maxSamples};
    }
    const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Erreur");
    const msg = lang === "fr"
      ? `✓ Import terminé. ${d.files_imported || 0} fichiers dans <code>${d.output_dir}</code>`
      : `✓ Import done. ${d.files_imported || 0} files in <code>${d.output_dir}</code>`;
    statusDiv.innerHTML = `<div class="alert alert-success">${msg}</div>`;
    // Suggestion de corpus path
    document.getElementById("corpus-path").value = d.output_dir;
  } catch(e) {
    statusDiv.innerHTML = `<div class="alert alert-error">Erreur : ${e.message}</div>`;
  }
}

// ─── Corpus upload ────────────────────────────────────────────────────────────
let _uploadMode = "zip";  // "zip" | "files"

function switchCorpusTab(tab) {
  document.getElementById("corpus-tab-browse").style.display = tab === "browse" ? "block" : "none";
  document.getElementById("corpus-tab-upload").style.display = tab === "upload" ? "block" : "none";
  document.getElementById("ctab-browse").classList.toggle("active", tab === "browse");
  document.getElementById("ctab-upload").classList.toggle("active", tab === "upload");
  if (tab === "upload") loadUploadedCorpora();
}

function onUploadModeChange() {
  _uploadMode = document.querySelector("input[name=upload-mode]:checked").value;
  const input = document.getElementById("upload-file-input");
  if (_uploadMode === "zip") {
    input.accept = ".zip";
    input.multiple = false;
    document.getElementById("upload-dropzone-text").textContent = t("upload_drop_zip");
  } else {
    input.accept = ".jpg,.jpeg,.png,.tif,.tiff,.webp,.gt.txt,.txt";
    input.multiple = true;
    document.getElementById("upload-dropzone-text").textContent = t("upload_drop_files");
  }
}

function onFileInputChange(event) {
  const files = Array.from(event.target.files);
  if (files.length > 0) uploadCorpus(files);
}

function onDropFiles(event) {
  event.preventDefault();
  document.getElementById("upload-dropzone").classList.remove("dragover");
  const files = Array.from(event.dataTransfer.files);
  if (files.length > 0) uploadCorpus(files);
}

async function uploadCorpus(files) {
  const progressContainer = document.getElementById("upload-progress-container");
  const progressBar = document.getElementById("upload-progress-bar");
  const progressText = document.getElementById("upload-progress-text");
  const previewEl = document.getElementById("upload-preview");

  progressContainer.style.display = "block";
  progressBar.style.width = "10%";
  progressText.textContent = t("upload_uploading");
  previewEl.innerHTML = "";

  const fd = new FormData();
  for (const f of files) fd.append("files", f);

  try {
    // Simulate progress during upload
    let pct = 10;
    const timer = setInterval(() => {
      pct = Math.min(pct + 5, 85);
      progressBar.style.width = pct + "%";
    }, 200);

    const r = await fetch("/api/corpus/upload", {method: "POST", body: fd});
    clearInterval(timer);
    progressBar.style.width = "100%";

    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || "Erreur serveur");
    }
    const d = await r.json();
    progressText.textContent = `✓ ${t("upload_success")} — ${d.doc_count} ${t("upload_pairs")}`;
    progressBar.style.background = "var(--success)";

    // Show preview
    renderUploadPreview(d, previewEl);

    // Show corpus OCR notice if triplet corpus
    _updateCorpusOCRNotice(d);

    // Set corpus path and auto-select
    setCorpusPath(d.corpus_path, `upload:${d.corpus_id} (${d.doc_count} docs)`);

    // Refresh list
    loadUploadedCorpora();
  } catch(e) {
    progressBar.style.width = "100%";
    progressBar.style.background = "var(--danger)";
    progressText.textContent = `✗ ${e.message}`;
  }
}

function renderUploadPreview(data, container) {
  const missingBadge = data.has_missing_gt
    ? `<span class="badge badge-err" style="margin-left:8px;">${data.missing_gt.length} ${t("upload_missing_gt")}</span>`
    : "";
  const ocrBadge = (data.has_ocr_text && data.ocr_text_count > 0)
    ? `<span class="badge" style="margin-left:8px; background:#dcfce7; color:#16a34a;">📝 ${data.ocr_text_count} .ocr.txt</span>`
    : "";
  let html = `<div class="corpus-preview">
    <div class="corpus-preview-header">
      <span>📄 ${data.doc_count} ${t("upload_pairs")}</span>${ocrBadge}${missingBadge}
    </div>`;
  for (const p of data.pairs) {
    html += `<div class="corpus-preview-pair">
      <span style="color:var(--text-muted);">🖼</span><span>${p.image}</span>
      <span style="color:var(--text-muted); margin-left:auto;">↔</span>
      <span style="color:var(--success);">${p.gt}</span>
    </div>`;
  }
  if (data.total_pairs > data.pairs.length) {
    html += `<div class="corpus-preview-more">… et ${data.total_pairs - data.pairs.length} autres paires</div>`;
  }
  for (const w of (data.warnings || [])) {
    html += `<div style="padding:5px 12px; font-size:11px; color:var(--warning);">⚠ ${w}</div>`;
  }
  html += `</div>`;
  container.innerHTML = html;
}

function setCorpusPath(path, label) {
  document.getElementById("corpus-path").value = path;
  document.getElementById("corpus-info").textContent = `✓ ${label}`;
}

function _updateCorpusOCRNotice(corpusData) {
  const notice = document.getElementById("corpus-ocr-notice");
  if (!notice) return;
  if (corpusData && corpusData.has_ocr_text && corpusData.ocr_text_count > 0) {
    notice.style.display = "block";
    notice.innerHTML = `📝 ${t("corpus_has_ocr")} <strong>(${corpusData.ocr_text_count} fichiers .ocr.txt)</strong>`;
  } else {
    notice.style.display = "none";
  }
}

async function loadUploadedCorpora() {
  const container = document.getElementById("uploads-list");
  try {
    const r = await fetch("/api/corpus/uploads");
    const d = await r.json();
    if (d.uploads.length === 0) {
      container.innerHTML = `<div style="color:var(--text-muted); font-size:12px;">${t("upload_no_corpus")}</div>`;
      return;
    }
    const currentPath = document.getElementById("corpus-path").value;
    container.innerHTML = d.uploads.map(u => {
      const isSelected = u.corpus_path === currentPath;
      const missing = u.has_missing_gt
        ? `<span class="badge badge-warn" style="margin-left:6px;">${t("upload_missing_gt")}</span>` : "";
      return `<div class="upload-corpus-item${isSelected ? " selected" : ""}"
                   onclick="setCorpusPath('${u.corpus_path}', 'upload (${u.doc_count} docs)'); loadUploadedCorpora()">
        <span class="upload-corpus-label">
          <strong>${u.doc_count} ${t("upload_pairs")}</strong>${missing}
          <span style="display:block; font-size:11px; color:var(--text-muted); font-family:monospace;">${u.corpus_path}</span>
        </span>
        <button class="btn btn-danger btn-sm" onclick="event.stopPropagation(); deleteUploadedCorpus('${u.corpus_id}')"
                title="${t("upload_delete")}">✕</button>
      </div>`;
    }).join("");
  } catch(e) {
    container.innerHTML = `<div style="color:var(--danger); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

async function deleteUploadedCorpus(corpusId) {
  try {
    await fetch(`/api/corpus/uploads/${corpusId}`, {method: "DELETE"});
    loadUploadedCorpora();
    // Clear corpus path if it was the deleted one
    const p = document.getElementById("corpus-path").value;
    if (p.includes(corpusId)) {
      document.getElementById("corpus-path").value = "";
      document.getElementById("corpus-info").textContent = "";
    }
  } catch(e) {}
}

// ─── Config save / load ──────────────────────────────────────────────────────
// Bindings UI pour /api/config/save et /api/config/load (Phase 4.3 du
// chantier post-rewrite).  Avant ce wiring, les endpoints existaient
// côté serveur (avec tests dédiés) mais aucun bouton ne les appelait —
// code zombie typique post-rewrite.

function _gatherCurrentConfig() {
  /** Sérialise l'état UI courant en dict compatible
   * ``/api/config/save``.  Inclut les compétiteurs composés
   * (_competitors), les options de normalisation et le profil de
   * langue rapport. */
  return {
    label: document.getElementById("report-name").value || "picarones-config",
    corpus_path: document.getElementById("corpus-path").value,
    competitors: _competitors,
    normalization_profile: document.getElementById("norm-profile").value,
    char_exclude: document.getElementById("char-exclude").value,
    output_dir: document.getElementById("output-dir").value,
    report_name: document.getElementById("report-name").value,
  };
}

async function saveConfigToFile() {
  /** POST la config courante à /api/config/save et déclenche le
   * téléchargement du JSON retourné. */
  const cfg = _gatherCurrentConfig();
  try {
    const r = await fetch("/api/config/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(cfg),
    });
    if (!r.ok) {
      const detail = await r.text();
      alert(lang === "fr"
        ? "Erreur sauvegarde config : " + detail
        : "Save config error: " + detail);
      return;
    }
    const blob = await r.blob();
    // Reconstitue le filename depuis le header Content-Disposition.
    const cd = r.headers.get("Content-Disposition") || "";
    const m = cd.match(/filename="([^"]+)"/);
    const filename = m ? m[1] : "picarones-config.json";
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (e) {
    alert(lang === "fr"
      ? "Erreur sauvegarde config : " + e.message
      : "Save config error: " + e.message);
  }
}

function loadConfigFromFile() {
  /** Déclenche le sélecteur de fichier — l'utilisateur choisit un
   * JSON, ``onConfigFileSelected`` fait le reste. */
  document.getElementById("config-file-input").click();
}

async function onConfigFileSelected(event) {
  /** Lit le fichier JSON, POST à /api/config/load pour validation +
   * upgrade éventuel, puis restaure l'état UI depuis le dict retourné. */
  const file = event.target.files[0];
  if (!file) return;
  // Reset l'input pour permettre un re-chargement du même fichier.
  event.target.value = "";
  try {
    const text = await file.text();
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      alert(lang === "fr"
        ? "Fichier JSON invalide : " + e.message
        : "Invalid JSON file: " + e.message);
      return;
    }
    const r = await fetch("/api/config/load", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(parsed),
    });
    if (!r.ok) {
      const detail = await r.text();
      alert(lang === "fr"
        ? "Erreur chargement config : " + detail
        : "Load config error: " + detail);
      return;
    }
    const result = await r.json();
    _applyConfig(result.config || {});
  } catch (e) {
    alert(lang === "fr"
      ? "Erreur chargement config : " + e.message
      : "Load config error: " + e.message);
  }
}

function _applyConfig(cfg) {
  /** Restaure l'état UI depuis un dict de config validé serveur.
   * Champs inconnus = ignorés silencieusement (responsabilité de
   * ``filter_config`` côté serveur). */
  if (typeof cfg.corpus_path === "string") {
    document.getElementById("corpus-path").value = cfg.corpus_path;
  }
  if (typeof cfg.normalization_profile === "string") {
    document.getElementById("norm-profile").value = cfg.normalization_profile;
  }
  if (typeof cfg.char_exclude === "string") {
    document.getElementById("char-exclude").value = cfg.char_exclude;
  }
  if (typeof cfg.output_dir === "string") {
    document.getElementById("output-dir").value = cfg.output_dir;
  }
  if (typeof cfg.report_name === "string") {
    document.getElementById("report-name").value = cfg.report_name;
  }
  if (Array.isArray(cfg.competitors)) {
    _competitors = cfg.competitors;
    renderCompetitors();
  }
}

// ─── Init ────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  loadStatus();
  loadNormProfiles();
  initHTRFilters();
  // Load OCR engines, LLM models, initialize composer
  await loadBenchmarkSections();
  onComposeOCRChange();      // Pre-populate Tesseract languages
  loadComposePrompts();       // Pre-load prompt files
  startAutoRefresh();         // Auto-detect new API keys every 10 s
  // Close modal on backdrop click
  document.getElementById("import-modal").addEventListener("click", e => {
    if (e.target === document.getElementById("import-modal")) closeImportModal();
  });
});
