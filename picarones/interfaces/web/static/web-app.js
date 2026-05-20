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
    nav_library: "Bibliothèque",
    nav_reports: "Rapports",
    nav_engines: "Moteurs",
    nav_import: "Import",
    nav_history: "Historique",
    nav_diagnose: "Diagnostic",
    nav_economics: "Coûts",
    nav_robustness: "Robustesse",
    nav_edition: "Édition",
    nav_compare: "Comparer",
    sys_title: "Système",
    sys_version: "Version",
    sys_mode: "Mode",
    sys_job: "Tâche",
    sys_idle: "au repos",
    sys_pipeline: "Pipeline",
    sys_engines_online: "Moteurs en ligne",
    sys_llms_online: "LLM en ligne",
    sys_details: "Système · détails",
    loading: "Chargement…",
    search: "Rechercher",
    all: "Tous",
    cancel: "Annuler",
    bench_hero_title: "Banc d'essai",
    bench_hero_desc: "Comparaison de pipelines OCR / HTR / VLM sur corpus patrimonial.",
    bench_corpus_title: "Corpus",
    bench_corpus_label: "Chemin vers le dossier corpus (paires image / .gt.txt)",
    bench_compose_desc: "Pipelines à comparer — OCR seul, OCR → LLM, ou VLM zero-shot",
    bench_options_desc: "Normalisation, exclusions, destination du rapport",
    bench_run_title: "Exécuter",
    bench_run_desc: "Lancer le benchmark sur la file de concurrents",
    queue_label: "QUEUE",
    competitors_word: "concurrent(s)",
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
    compose_max_image_dim: "Image max (px)",
    compose_max_image_dim_hint: "0 = pleine résolution (défaut, méthodo inchangée). > 0 réduit l'image envoyée au VLM (modes image) pour éviter les 429 — change la méthodo, run fingerprinté à part.",
    compose_add: "+ Ajouter",
    compose_empty: "Aucun concurrent ajouté.",
    mode_text_only: "Post-correction texte",
    mode_text_image: "Post-correction image+texte",
    mode_zero_shot: "Zero-shot",
    bench_norm_label: "Profil de normalisation",
    bench_lang_label: "Langue (Tesseract)",
    bench_char_exclude_label: "Caractères à ignorer",
    bench_output_label: "Dossier de sortie",
    bench_name_label: "Nom du rapport (optionnel)",
    bench_start: "▶ Lancer le benchmark",
    bench_cancel: "✕ Annuler",
    bench_config_save: "💾 Sauvegarder config",
    bench_config_load: "📂 Charger config",
    bench_progress_title: "Progression",
    bench_log: "Journal",
    bench_result_title: "Résultats",
    bench_synthesis_title: "Synthèse narrative",
    bench_open_report: "Ouvrir le rapport",
    reports_title: "Rapports générés",
    reports_hero_desc: "Rapports HTML produits par les benchmarks — tri par date.",
    reports_dir_label: "Dossier de rapports",
    reports_refresh: "Rafraîchir",
    engines_hero_title: "Moteurs",
    engines_hero_desc: "État des adapters OCR/HTR et providers LLM disponibles dans l'environnement.",
    engines_ocr_title: "Moteurs OCR",
    engines_ocr_desc: "Tesseract, Pero, Kraken, Calamari, Mistral OCR, Google Vision, Azure DI.",
    engines_llm_title: "LLMs disponibles",
    engines_llm_desc: "Providers texte / vision configurés via variables d'environnement.",
    library_hero_title: "Bibliothèque de corpus",
    library_hero_desc: "Corpus locaux et catalogues distants — toute la matière en un seul endroit.",
    library_my_corpora: "Mes corpus",
    library_discover: "Découvrir",
    library_local_title: "Corpus locaux",
    library_local_desc: "Téléversés, importés depuis HTR-United / HuggingFace, ou pointés depuis le filesystem.",
    library_remote_title: "Sources distantes",
    library_remote_desc: "Catalogues, datasets, manifestes",
    library_drop_zip: "Glissez un .zip ici ou cliquez pour sélectionner",
    library_local_empty: "Aucun corpus local. Importez-en un.",
    library_use_in_benchmark: "Utiliser dans Benchmark",
    bench_corpus_sub: "Choisir un corpus chargé dans la Bibliothèque.",
    bench_corpus_pick: "Corpus",
    bench_corpus_pick_placeholder: "— Choisir un corpus —",
    bench_corpus_open_library: "Ouvrir la Bibliothèque",
    bench_corpus_advanced: "Mode expert",
    bench_corpus_free_label: "Chemin libre (filesystem)",
    bench_corpus_free_hint: "non recommandé hors usage local",
    sys_lang: "Langue",
    bench_profile_label: "Profil de run",
    bench_profile_hint: "Métriques activées et sections du rapport HTML.",
    bench_profile_minimal: "minimal — CER/WER seulement",
    bench_profile_standard: "standard — défaut",
    bench_profile_philological: "philological — édition critique (MUFI, abréviations)",
    bench_profile_diagnostics: "diagnostics — diagnostic approfondi (leviers, image_predictive)",
    bench_profile_economics: "economics — coûts API LLM + throughput",
    bench_profile_full: "full — tout activé",
    import_modal_desc: "Sélectionnez la destination et le nombre maximum de documents à télécharger.",
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
    nav_library: "Library",
    nav_reports: "Reports",
    nav_engines: "Engines",
    nav_import: "Import",
    nav_history: "History",
    nav_diagnose: "Diagnose",
    nav_economics: "Costs",
    nav_robustness: "Robustness",
    nav_edition: "Edition",
    nav_compare: "Compare",
    sys_title: "System",
    sys_version: "Version",
    sys_mode: "Mode",
    sys_job: "Job",
    sys_idle: "idle",
    sys_pipeline: "Pipeline",
    sys_engines_online: "Engines online",
    sys_llms_online: "LLMs online",
    sys_details: "System · details",
    loading: "Loading…",
    search: "Search",
    all: "All",
    cancel: "Cancel",
    bench_hero_title: "Benchmark",
    bench_hero_desc: "Compare OCR / HTR / VLM pipelines on heritage corpora.",
    bench_corpus_title: "Corpus",
    bench_corpus_label: "Path to corpus directory (image / .gt.txt pairs)",
    bench_compose_desc: "Pipelines to compare — OCR only, OCR → LLM, or VLM zero-shot",
    bench_options_desc: "Normalization, exclusions, report destination",
    bench_run_title: "Execute",
    bench_run_desc: "Run the benchmark on the queued competitors",
    queue_label: "QUEUE",
    competitors_word: "competitor(s)",
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
    compose_max_image_dim: "Max image (px)",
    compose_max_image_dim_hint: "0 = full resolution (default, methodology unchanged). > 0 shrinks the image sent to the VLM (image modes) to avoid 429s — changes methodology, fingerprinted as a separate run.",
    compose_add: "+ Add",
    compose_empty: "No competitors added.",
    mode_text_only: "Text post-correction",
    mode_text_image: "Image+text post-correction",
    mode_zero_shot: "Zero-shot",
    bench_norm_label: "Normalization profile",
    bench_lang_label: "Language (Tesseract)",
    bench_char_exclude_label: "Characters to ignore",
    bench_output_label: "Output directory",
    bench_name_label: "Report name (optional)",
    bench_start: "▶ Start benchmark",
    bench_cancel: "✕ Cancel",
    bench_config_save: "💾 Save config",
    bench_config_load: "📂 Load config",
    bench_progress_title: "Progress",
    bench_log: "Log",
    bench_result_title: "Results",
    bench_synthesis_title: "Narrative synthesis",
    bench_open_report: "Open report",
    reports_title: "Generated reports",
    reports_hero_desc: "HTML reports produced by benchmarks — sorted by date.",
    reports_dir_label: "Reports directory",
    reports_refresh: "Refresh",
    engines_hero_title: "Engines",
    engines_hero_desc: "Status of OCR/HTR adapters and available LLM providers.",
    engines_ocr_title: "OCR Engines",
    engines_ocr_desc: "Tesseract, Pero, Kraken, Calamari, Mistral OCR, Google Vision, Azure DI.",
    engines_llm_title: "Available LLMs",
    engines_llm_desc: "Text / vision providers configured via environment variables.",
    library_hero_title: "Corpus library",
    library_hero_desc: "Local corpora and remote catalogues — all the material in one place.",
    library_my_corpora: "My corpora",
    library_discover: "Discover",
    library_local_title: "Local corpora",
    library_local_desc: "Uploaded, imported from HTR-United / HuggingFace, or pointed at from the filesystem.",
    library_remote_title: "Remote sources",
    library_remote_desc: "Catalogues, datasets, manifests",
    library_drop_zip: "Drop a .zip here or click to select",
    library_local_empty: "No local corpus. Import one.",
    library_use_in_benchmark: "Use in Benchmark",
    bench_corpus_sub: "Pick a corpus loaded in the Library.",
    bench_corpus_pick: "Corpus",
    bench_corpus_pick_placeholder: "— Pick a corpus —",
    bench_corpus_open_library: "Open the Library",
    bench_corpus_advanced: "Expert mode",
    bench_corpus_free_label: "Free path (filesystem)",
    bench_corpus_free_hint: "not recommended outside local use",
    sys_lang: "Language",
    bench_profile_label: "Run profile",
    bench_profile_hint: "Activated metrics and HTML report sections.",
    bench_profile_minimal: "minimal — CER/WER only",
    bench_profile_standard: "standard — default",
    bench_profile_philological: "philological — critical edition (MUFI, abbreviations)",
    bench_profile_diagnostics: "diagnostics — in-depth diagnosis (levers, image_predictive)",
    bench_profile_economics: "economics — LLM API costs + throughput",
    bench_profile_full: "full — everything enabled",
    import_modal_desc: "Pick the destination directory and the maximum number of documents to download.",
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
function setLang(next) {
  if (next !== "fr" && next !== "en") return;
  lang = next;
  const legacyBtn = document.getElementById("lang-btn");
  if (legacyBtn) legacyBtn.textContent = lang === "fr" ? "EN" : "FR";
  const frBtn = document.getElementById("lang-btn-fr");
  const enBtn = document.getElementById("lang-btn-en");
  if (frBtn) frBtn.classList.toggle("on", lang === "fr");
  if (enBtn) enBtn.classList.toggle("on", lang === "en");
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const k = el.getAttribute("data-i18n");
    if (T[lang][k]) el.textContent = T[lang][k];
  });
}
function toggleLang() { setLang(lang === "fr" ? "en" : "fr"); }

// Bouton "Systeme · details" en bas de la sidebar — ouvre la vue
// Engines comme proxy panneau Systeme (regroupera plus tard
// engines + history + autres workflows secondaires).
function openSystemPanel() {
  showView("engines");
}

// ─── Navigation ──────────────────────────────────────────────────────────────
function showView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav-btn, .nav-item").forEach(b => b.classList.remove("active"));
  const view = document.getElementById("view-" + name);
  if (view) view.classList.add("active");
  const btn = document.querySelector('[data-view="' + name + '"]');
  if (btn) btn.classList.add("active");

  if (name === "reports") loadReports();
  if (name === "engines") loadEngines();
  if (name === "import" || name === "library") loadLibraryLocalCorpora();
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
    _refreshSysCounts(d);
  } catch(e) {
    const c = document.getElementById("ocr-engines-status-list");
    if (c) c.innerHTML = `<div style="color:var(--err);font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

function _refreshSysCounts(d) {
  if (!d) return;
  const engOn = (d.engines || []).filter(e => e.available).length;
  const engTot = (d.engines || []).length;
  const llmOn = (d.llms || []).filter(l => l.available).length;
  const llmTot = (d.llms || []).length;
  const eEl = document.getElementById("sys-engines-online");
  const lEl = document.getElementById("sys-llms-online");
  const pcEl = document.getElementById("sys-pipeline-counts");
  if (eEl) eEl.textContent = `${engOn} / ${engTot}`;
  if (lEl) lEl.textContent = `${llmOn} / ${llmTot}`;
  if (pcEl) pcEl.textContent = `${engOn}+${llmOn} · ${engTot + llmTot}`;
}

function _makeProviderRow(eng, msId) {
  const dotCls = eng.available ? "on" : (eng.status === "not_running" ? "warn" : "off");
  let statusLabel;
  if (eng.available) statusLabel = eng.version ? eng.version : (lang === "fr" ? "disponible" : "available");
  else if (eng.status === "missing_key") statusLabel = eng.key_env ? `<code class="mono" style="font-size:11px;color:var(--clay-deep)">${_escapeHtml(eng.key_env)}</code>` : (lang === "fr" ? "clé manquante" : "key missing");
  else if (eng.status === "not_running") statusLabel = lang === "fr" ? "inactif" : "not running";
  else statusLabel = lang === "fr" ? "non installé" : "not installed";

  const row = document.createElement("div");
  row.style = "display:grid; grid-template-columns:1fr auto auto; gap:14px; align-items:center; padding:10px 14px; border-bottom:1px solid var(--g-50); font-size:13px;";
  row.innerHTML = `
    <div style="display:flex; align-items:center; gap:10px;"><span class="dot ${dotCls}"></span><strong>${_escapeHtml(eng.label)}</strong></div>
    <div class="mono" style="font-size:11.5px; color:var(--g-500);">${statusLabel}</div>
    <div id="${msId}" class="mono" style="font-size:11.5px; min-width:120px; text-align:right;">${eng.available ? '<span class="dot busy"></span>' : ""}</div>`;
  return row;
}

async function renderOCREnginesSection(engines) {
  const container = document.getElementById("ocr-engines-status-list");
  if (!container) return;
  container.innerHTML = "";
  for (const eng of engines) {
    const msId = `ms-ocr-${eng.id}`;
    container.appendChild(_makeProviderRow(eng, msId));
    if (eng.available) {
      fetchModels(eng.id).then(models => {
        const div = document.getElementById(msId);
        if (!div) return;
        div.innerHTML = models.length === 0
          ? `<span style="color:var(--g-400);font-size:11px;">—</span>`
          : `<span style="font-size:12px;">${models.slice(0,5).join(", ")}${models.length > 5 ? ` +${models.length-5}` : ""}</span>`;
      }).catch(() => {
        const div = document.getElementById(msId);
        if (div) div.innerHTML = `<span style="color:var(--err);font-size:11px;">Erreur API</span>`;
      });
    }
  }
}

async function renderLLMSection(llms) {
  const container = document.getElementById("llm-status-list");
  if (!container) return;
  container.innerHTML = "";
  for (const llm of llms) {
    const msId = `ms-llm-${llm.id}`;
    container.appendChild(_makeProviderRow(llm, msId));
    if (llm.available) {
      fetchModels(llm.id).then(models => {
        const div = document.getElementById(msId);
        if (!div) return;
        div.innerHTML = models.length === 0
          ? `<span style="color:var(--g-400);font-size:11px;">—</span>`
          : `<span style="font-size:12px;">${models.slice(0,3).join(", ")}${models.length > 3 ? ` +${models.length-3}` : ""}</span>`;
      }).catch(() => {
        const div = document.getElementById(msId);
        if (div) div.innerHTML = `<span style="color:var(--err);font-size:11px;">Erreur API</span>`;
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
        _refreshSysCounts(d);
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

function _getComposeMode() {
  // Nouveau segmented control XerOCR (boutons + data-mode).
  const onBtn = document.querySelector("#compose-mode-tabs button.on");
  if (onBtn && onBtn.dataset.mode) return onBtn.dataset.mode;
  // Fallback legacy : input[name=compose-mode]:checked.
  const r = document.querySelector("input[name=compose-mode]:checked");
  return r ? r.value : "ocr";
}

function setComposeMode(mode) {
  const tabs = document.getElementById("compose-mode-tabs");
  if (tabs) {
    tabs.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.mode === mode));
  }
  // Sync legacy radio si encore present.
  const radio = document.querySelector(`input[name=compose-mode][value="${mode}"]`);
  if (radio) radio.checked = true;
  onComposeModeChange();
}

async function onComposeLLMChange() {
  const provider = document.getElementById("compose-llm-provider").value;
  const composeMode = _getComposeMode();
  const pipelineMode = document.getElementById("compose-pipeline-mode").value;
  // Apply capability filter for modes requiring vision
  const needsVision = (pipelineMode === "text_and_image" || pipelineMode === "zero_shot");
  const capability = (composeMode === "postcorrection" || composeMode === "pipeline") && needsVision ? "vision" : "";
  _loadLLMModelsWithCapability(provider, capability);
}

function onComposeModeChange() {
  const mode = _getComposeMode();
  const ocrSection = document.getElementById("compose-ocr-section");
  const pipelineSection = document.getElementById("compose-pipeline-section");

  if (mode === "ocr") {
    if (ocrSection) ocrSection.style.display = "grid";
    if (pipelineSection) pipelineSection.style.display = "none";
  } else if (mode === "pipeline") {
    if (ocrSection) ocrSection.style.display = "grid";
    if (pipelineSection) pipelineSection.style.display = "block";
    onComposeLLMChange();
  } else if (mode === "postcorrection") {
    if (ocrSection) ocrSection.style.display = "none";
    if (pipelineSection) pipelineSection.style.display = "block";
    onComposePipelineModeChange();
  }
}

function onComposePipelineModeChange() {
  const composeMode = _getComposeMode();
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
  const mode = _getComposeMode();
  const errEl = document.getElementById("compose-error");

  const comp = { name: "", engine_name: "", ocr_model: "",
                  llm_provider: "", llm_model: "", pipeline_mode: "", prompt_file: "",
                  max_image_dimension: 0 };
  const _maxImgDim = parseInt((document.getElementById("compose-max-image-dim") || {}).value, 10);
  const maxImgDim = Number.isFinite(_maxImgDim) && _maxImgDim > 0 ? _maxImgDim : 0;

  if (mode === "postcorrection") {
    // Post-correction : OCR vient du corpus (.ocr.txt)
    comp.engine_name = "corpus";
    comp.llm_provider = document.getElementById("compose-llm-provider").value;
    comp.llm_model = document.getElementById("compose-llm-model").value;
    comp.pipeline_mode = document.getElementById("compose-pipeline-mode").value;
    comp.prompt_file = document.getElementById("compose-prompt").value;
    comp.max_image_dimension = maxImgDim;
    if (!comp.llm_provider || !comp.llm_model) {
      errEl.textContent = lang === "fr" ? "Sélectionnez un provider et un modèle LLM." : "Select an LLM provider and model.";
      return;
    }
    const modeLabel = {"text_only":"texte","text_and_image":"img+texte","zero_shot":"zero-shot"}[comp.pipeline_mode] || comp.pipeline_mode;
    const promptStem = (comp.prompt_file || "").split("/").pop().replace(/\.txt$/i, "");
    comp.name = `📝 ${comp.llm_model} [${modeLabel}${promptStem ? "/" + promptStem : ""}]`;
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
    comp.max_image_dimension = maxImgDim;
    if (!comp.llm_provider) {
      errEl.textContent = lang === "fr" ? "Sélectionnez un provider LLM." : "Select an LLM provider.";
      return;
    }
    const modeLabel = {"text_only":"texte","text_and_image":"img+texte","zero_shot":"zero-shot"}[comp.pipeline_mode] || comp.pipeline_mode;
    const promptStem = (comp.prompt_file || "").split("/").pop().replace(/\.txt$/i, "");
    comp.name = `${ocrEngine}${ocrModel ? ":"+ocrModel : ""} → ${comp.llm_model || comp.llm_provider} [${modeLabel}${promptStem ? "/" + promptStem : ""}]`;
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
  if (!container) return;
  _updateCompetitorCounts();
  if (_competitors.length === 0) {
    container.innerHTML = `<div class="empty">${t("compose_empty")}</div>`;
    return;
  }
  const colors = ["ink", "fern", "slate", "clay", "butter"];
  container.innerHTML = _competitors.map((c, i) => {
    const isCorpusOCR = c.engine_name === "corpus" || (c.engine_name === "" && c.llm_provider);
    const isPipeline = !!c.llm_provider && !isCorpusOCR;
    let kind, chain;
    if (isCorpusOCR) {
      kind = t("compose_postcorrection");
      chain = `corpus_ocr → ${c.llm_provider}:${c.llm_model} [${c.pipeline_mode}]`;
    } else if (isPipeline) {
      kind = "OCR → LLM";
      chain = `${c.engine_name}:${c.ocr_model} → ${c.llm_provider}:${c.llm_model} [${c.pipeline_mode}]`;
    } else {
      kind = "OCR";
      chain = `${c.engine_name}:${c.ocr_model}`;
    }
    const cid = "C" + String(i + 1).padStart(2, "0");
    const color = colors[i % colors.length];
    return `<div class="competitor">
      <span class="c-id ${color}">${cid}</span>
      <div>
        <div class="c-name">${_escapeHtml(c.name)}</div>
        <div class="c-chain"><span class="tag tag-mono">${_escapeHtml(kind)}</span><span class="mono" style="color:var(--g-500); font-size:11.5px;">${_escapeHtml(chain)}</span></div>
      </div>
      <button class="btn btn-ghost btn-sm" type="button" onclick="removeCompetitor(${i})" title="${t("upload_delete")}">✕</button>
    </div>`;
  }).join("");
}

function _updateCompetitorCounts() {
  const n = _competitors.length;
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set("hero-competitors", n);
  set("competitors-count", n);
  set("competitors-count-inline", n);
}

function _escapeHtml(s) {
  return String(s || "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
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
      const heroNorm = document.getElementById("hero-norm");
      if (heroNorm) heroNorm.textContent = sel.value || "—";
    });
    // Init hero-norm avec la valeur par défaut.
    const heroNorm = document.getElementById("hero-norm");
    if (heroNorm) heroNorm.textContent = sel.value || "—";
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
      up.className = "file-row";
      up.innerHTML = `<span class="icon">⬆</span><span>..</span><span class="meta"></span><span class="meta"></span>`;
      up.onclick = () => browsePath(d.parent_path);
      fb.appendChild(up);
    }
    d.items.filter(i => i.is_dir).forEach(item => {
      const el = document.createElement("div");
      el.className = "file-row";
      const hasCorpus = item.has_corpus ? `<span class="tag tag-fern">✓ ${item.gt_count} GT</span>` : "";
      el.innerHTML = `<span class="icon">📁</span><span>${_escapeHtml(item.name)}</span><span class="meta"></span>${hasCorpus}`;
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
      fb.innerHTML = '<div class="empty">Dossier vide</div>';
    }
  } catch(e) {
    document.getElementById("file-browser").innerHTML =
      `<div class="empty" style="color:var(--err);">Erreur : ${_escapeHtml(e.message)}</div>`;
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
    profile: (document.getElementById("run-profile") || {}).value || "standard",
  };

  document.getElementById("start-btn").disabled = true;
  document.getElementById("cancel-btn").style.display = "inline-flex";
  document.getElementById("bench-progress-section").style.display = "block";
  document.getElementById("bench-result-section").style.display = "none";
  document.getElementById("bench-log").textContent = "";
  document.getElementById("engine-progress-list").innerHTML = "";
  document.getElementById("bench-status-text").textContent = lang === "fr" ? "Démarrage…" : "Starting…";
  _setBenchState("RUN");

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
    const colors = ["fern", "slate", "clay", "butter", "ink"];
    if (!seenEngines[engId]) {
      const idx = Object.keys(seenEngines).length;
      seenEngines[engId] = colors[idx % colors.length];
      const color = seenEngines[engId];
      const div = document.createElement("div");
      div.style = "display:grid; grid-template-columns:1fr 60px; gap:14px; align-items:center; margin-bottom:10px; font-size:12.5px;";
      div.innerHTML = `<div>
          <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span>${_escapeHtml(d.engine)}</span>
            <span class="mono" style="color:var(--g-500);" id="eng-status-${engId}">${d.processed}/${d.total}</span>
          </div>
          <div class="progress ${color}"><div class="progress-bar" id="eng-bar-${engId}" style="width:0%"></div></div>
        </div>
        <span class="num" style="text-align:right; color:var(--g-500);" id="eng-pct-${engId}">0%</span>`;
      pl.appendChild(div);
    }
    const bar = document.getElementById(`eng-bar-${engId}`);
    const pctEl = document.getElementById(`eng-pct-${engId}`);
    const statusEl = document.getElementById(`eng-status-${engId}`);
    if (bar) bar.style.width = pct + "%";
    if (pctEl) pctEl.textContent = pct + "%";
    if (statusEl) statusEl.textContent = `${d.processed}/${d.total}`;
    document.getElementById("bench-status-text").textContent =
      `${pct}% — ${d.engine} (${d.processed}/${d.total})`;
  });

  _eventSource.addEventListener("complete", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message, "success");
    _showResults(d);
    _finishBenchmark(true);
  });

  _eventSource.addEventListener("error", e => {
    const d = JSON.parse(e.data);
    appendLog(d.message, "error");
    _finishBenchmark(false);
  });

  _eventSource.addEventListener("cancelled", e => {
    appendLog(lang === "fr" ? "Benchmark annulé." : "Benchmark cancelled.", "warn");
    _finishBenchmark(false);
  });

  _eventSource.addEventListener("done", e => { _finishBenchmark(true); });
  _eventSource.onerror = () => { if (_currentJobId) _finishBenchmark(false); };
}

function _showResults(data) {
  const section = document.getElementById("bench-result-section");
  section.style.display = "block";
  if (data.output_html) {
    const link = document.getElementById("bench-report-link");
    link.href = `/reports/${data.output_html.split("/").pop()}`;
  }
  if (data.ranking) {
    let html = `<table class="data"><thead><tr>
      <th style="width:60px">#</th>
      <th>${lang==="fr"?"Moteur":"Engine"}</th>
      <th class="num-cell">CER</th>
      <th class="num-cell">WER</th>
      <th class="num-cell">${lang==="fr"?"Docs":"Docs"}</th>
    </tr></thead><tbody>`;
    data.ranking.forEach((row, i) => {
      const cer = row.mean_cer != null ? (row.mean_cer*100).toFixed(2) : "—";
      const wer = row.mean_wer != null ? (row.mean_wer*100).toFixed(2) : "—";
      const pillCls = i === 0 ? "rank-pill first" : "rank-pill";
      html += `<tr>
        <td><span class="${pillCls}">#${i+1}</span></td>
        <td style="font-weight:500">${_escapeHtml(row.engine)}</td>
        <td class="num-cell">${cer}</td>
        <td class="num-cell">${wer}</td>
        <td class="num-cell">${row.total_docs || ""}</td>
      </tr>`;
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

function _setBenchState(state) {
  // state ∈ {"READY","RUN","DONE"}
  const heroEl = document.getElementById("hero-state");
  const sysJob = document.getElementById("sys-job");
  if (heroEl) heroEl.textContent = state;
  if (sysJob) {
    const label = state === "RUN" ? (lang === "fr" ? "en cours" : "running")
                : state === "DONE" ? (lang === "fr" ? "terminé" : "done")
                : (lang === "fr" ? "au repos" : "idle");
    sysJob.textContent = label;
  }
}

function _finishBenchmark(success) {
  if (_eventSource) { _eventSource.close(); _eventSource = null; }
  const startBtn = document.getElementById("start-btn");
  const cancelBtn = document.getElementById("cancel-btn");
  const status = document.getElementById("bench-status-text");
  if (startBtn) startBtn.disabled = false;
  if (cancelBtn) cancelBtn.style.display = "none";
  if (status) status.textContent = "";
  _setBenchState(success ? "DONE" : "READY");
}

async function cancelBenchmark() {
  if (!_currentJobId) return;
  await fetch(`/api/benchmark/${_currentJobId}/cancel`, {method: "POST"});
}

function appendLog(msg, cls) {
  const box = document.getElementById("bench-log");
  if (!box) return;
  const ts = new Date().toISOString().slice(11, 19);
  const line = document.createElement("div");
  // Mappage : success → ok, error → err, warn → warn, info → (default), blue/slate → slate.
  // Les class CSS XerOCR sont `.ts`, `.ok`, `.warn`, `.err`, `.blue`, `.slate`.
  const map = { success: "ok", error: "err", warn: "warn", blue: "blue", slate: "slate" };
  const lvl = map[cls] || "";
  line.innerHTML = `<span class="ts">${ts}</span><span${lvl ? ` class="${lvl}"` : ""}>${_escapeHtml(msg)}</span>`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

// ─── Reports ─────────────────────────────────────────────────────────────────
async function loadReports() {
  const dir = document.getElementById("reports-dir").value || ".";
  const container = document.getElementById("reports-list");
  if (!container) return;
  container.innerHTML = `<div class="empty">${t("loading")}</div>`;
  try {
    const r = await fetch(`/api/reports?reports_dir=${encodeURIComponent(dir)}`);
    const d = await r.json();
    const setMeta = (n) => {
      const a = document.getElementById("reports-aside");
      const c = document.getElementById("reports-count");
      if (a) a.textContent = `${n} REPORT${n === 1 ? "" : "S"}`;
      if (c) c.textContent = n;
    };
    setMeta(d.reports.length);
    if (d.reports.length === 0) {
      container.innerHTML = `<div class="empty">${t("no_reports")}</div>`;
      return;
    }
    let html = `<table class="data"><thead><tr>
      <th>${lang === "fr" ? "Fichier" : "File"}</th>
      <th class="num-cell">${lang === "fr" ? "Taille" : "Size"}</th>
      <th>${lang === "fr" ? "Modifié" : "Modified"}</th>
      <th></th>
    </tr></thead><tbody>`;
    d.reports.forEach(rep => {
      const date = new Date(rep.modified).toLocaleString(lang === "fr" ? "fr-FR" : "en-US");
      html += `<tr>
        <td class="mono" style="color:var(--g-700);">${_escapeHtml(rep.filename)}</td>
        <td class="num-cell">${rep.size_kb} Ko</td>
        <td class="mono" style="color:var(--g-500); font-size:11.5px;">${_escapeHtml(date)}</td>
        <td style="text-align:right;"><a href="${rep.url}" target="_blank" class="btn btn-primary btn-sm">${lang === "fr" ? "Ouvrir" : "Open"}</a></td>
      </tr>`;
    });
    html += "</tbody></table>";
    container.innerHTML = html;
  } catch(e) {
    container.innerHTML = `<div class="empty" style="color:var(--err);">Erreur : ${_escapeHtml(e.message)}</div>`;
  }
}

// ─── Engines status ──────────────────────────────────────────────────────────
async function loadEngines() {
  try {
    const r = await fetch("/api/engines");
    const d = await r.json();
    _refreshSysCounts(d);

    const setHero = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    setHero("engines-count-total", `${d.engines.filter(e => e.available).length}/${d.engines.length}`);
    setHero("llms-count-total", `${d.llms.filter(l => l.available).length}/${d.llms.length}`);

    // OCR
    let html = `<table class="data"><thead><tr>
      <th>ID</th>
      <th>${lang === "fr" ? "Nom" : "Name"}</th>
      <th>Version</th>
      <th>Statut</th>
    </tr></thead><tbody>`;
    d.engines.forEach(e => {
      const tagCls = e.available ? "tag-fern" : "tag-clay";
      const lbl = e.available ? t("available") : t("not_installed");
      html += `<tr>
        <td class="mono" style="color:var(--g-500); font-size:11.5px;">${_escapeHtml(e.id)}</td>
        <td style="font-weight:500">${_escapeHtml(e.label)}</td>
        <td class="mono" style="color:var(--g-500); font-size:11.5px;">${_escapeHtml(e.version || "—")}</td>
        <td><span class="tag ${tagCls}">${_escapeHtml(lbl)}</span></td>
      </tr>`;
    });
    html += "</tbody></table>";
    const oList = document.getElementById("engines-ocr-list");
    if (oList) oList.innerHTML = html;

    // LLMs
    let llmHtml = `<table class="data"><thead><tr>
      <th>ID</th>
      <th>${lang === "fr" ? "Nom" : "Name"}</th>
      <th>Statut</th>
      <th>${lang === "fr" ? "Détail" : "Detail"}</th>
    </tr></thead><tbody>`;
    d.llms.forEach(e => {
      const tagCls = e.available ? "tag-fern" : "tag-butter";
      const statusKey = e.status === "configured" ? "configured"
        : e.status === "running" ? "running"
        : e.status === "not_running" ? "not_running"
        : "missing_key";
      const lbl = t(statusKey);
      let detail = "—";
      if (e.key_env) detail = `<code class="mono" style="font-size:11px; color:var(--clay-deep);">${_escapeHtml(e.key_env)}</code>`;
      if (e.models && e.models.length > 0) detail = `<span class="mono" style="font-size:11.5px; color:var(--g-500);">${_escapeHtml(e.models.slice(0, 3).join(", "))}</span>`;
      llmHtml += `<tr>
        <td class="mono" style="color:var(--g-500); font-size:11.5px;">${_escapeHtml(e.id)}</td>
        <td style="font-weight:500">${_escapeHtml(e.label)}</td>
        <td><span class="tag ${tagCls}">${_escapeHtml(lbl)}</span></td>
        <td>${detail}</td>
      </tr>`;
    });
    llmHtml += "</tbody></table>";
    const lList = document.getElementById("engines-llm-list");
    if (lList) lList.innerHTML = llmHtml;
  } catch(e) {
    const oList = document.getElementById("engines-ocr-list");
    if (oList) oList.innerHTML = `<div class="empty" style="color:var(--err);">Erreur : ${_escapeHtml(e.message)}</div>`;
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
  container.innerHTML = `<div style="color: var(--g-400); font-size:12px;">${t("loading")}</div>`;
  try {
    const url = `/api/htr-united/catalogue?query=${encodeURIComponent(q)}&language=${encodeURIComponent(lang2)}&script=${encodeURIComponent(script)}`;
    const r = await fetch(url);
    const d = await r.json();
    _updateHtrDemoBanner(Boolean(d.is_demo));
    if (d.entries.length === 0) {
      container.innerHTML = `<div style="color: var(--g-400); font-size:12px;">${lang==="fr"?"Aucun résultat.":"No results."}</div>`;
      return;
    }
    container.innerHTML = d.entries.map(e => {
      // Defensif : certains champs peuvent etre des objets (license,
      // script structures) qui sortent en "[object Object]" si on
      // les passe directement a innerHTML.  On extrait juste le label.
      const _str = v => typeof v === "string"
        ? v
        : (v && (v.name || v.id || v.label) ? (v.name || v.id || v.label) : "");
      const tags = [...(e.language || []), ...(e.script || [])]
        .map(_str).filter(Boolean)
        .map(s => `<span class="tag tag-slate">${_escapeHtml(s)}</span>`).join("");
      const idAttr = _escapeAttr(e.id);
      const titleAttr = _escapeAttr(e.title);
      return `<div class="ds-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px;">
          <div class="ds-name">${_escapeHtml(e.title)}</div>
          <button class="btn btn-primary btn-sm" type="button" onclick="openImportModal('htr', '${idAttr}', '${titleAttr}')">
            ${lang === "fr" ? "Importer" : "Import"}
          </button>
        </div>
        <div class="ds-desc">${_escapeHtml(e.description)}</div>
        <div class="ds-meta">
          <span>${_escapeHtml(e.institution)}</span>
          <span><b class="num">${e.lines.toLocaleString()}</b> ${t("lines")}</span>
          <span>${_escapeHtml(e.format)}</span>
        </div>
        <div class="ds-tags">${tags}</div>
      </div>`;
    }).join("");
  } catch(e) {
    container.innerHTML = `<div style="color: var(--err); font-size:12px;">Erreur : ${e.message}</div>`;
  }
}

async function searchHuggingFace() {
  const q = document.getElementById("hf-search").value;
  const langFilter = document.getElementById("hf-lang-filter").value;
  const tags = document.getElementById("hf-tags").value;
  const container = document.getElementById("hf-results");
  container.innerHTML = `<div style="color: var(--g-400); font-size:12px;">${t("loading")}</div>`;
  try {
    const url = `/api/huggingface/search?query=${encodeURIComponent(q)}&language=${encodeURIComponent(langFilter)}&tags=${encodeURIComponent(tags)}`;
    const r = await fetch(url);
    const d = await r.json();
    if (d.datasets.length === 0) {
      container.innerHTML = `<div style="color: var(--g-400); font-size:12px;">${lang==="fr"?"Aucun résultat.":"No results."}</div>`;
      return;
    }
    container.innerHTML = d.datasets.map(ds => {
      const _str = v => typeof v === "string"
        ? v
        : (v && (v.name || v.id || v.label) ? (v.name || v.id || v.label) : "");
      const tags2 = (ds.tags || []).slice(0, 5).map(_str).filter(Boolean)
        .map(s => `<span class="tag tag-slate">${_escapeHtml(s)}</span>`).join("");
      const idAttr = _escapeAttr(ds.dataset_id);
      const titleAttr = _escapeAttr(ds.title);
      return `<div class="ds-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px;">
          <div class="ds-name">${_escapeHtml(ds.title)}</div>
          <button class="btn btn-primary btn-sm" type="button" onclick="openImportModal('hf', '${idAttr}', '${titleAttr}')">
            ${lang === "fr" ? "Importer" : "Import"}
          </button>
        </div>
        <div class="ds-desc">${_escapeHtml(ds.description)}</div>
        <div class="ds-meta">
          <span>${_escapeHtml(ds.institution || ds.dataset_id)}</span>
          ${ds.downloads ? `<span><b class="num">${ds.downloads.toLocaleString()}</b> ${lang === "fr" ? "téléchargements" : "downloads"}</span>` : ""}
        </div>
        <div class="ds-tags">${tags2}</div>
      </div>`;
    }).join("");
  } catch(e) {
    container.innerHTML = `<div style="color: var(--err); font-size:12px;">Erreur : ${e.message}</div>`;
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
  statusDiv.innerHTML = `<div class="surface-flat" style="padding:10px 14px; font-size:12.5px;"><span class="dot busy"></span> ${lang === "fr" ? "Import en cours…" : "Importing…"}</div>`;

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
    statusDiv.innerHTML = `<div class="surface-flat" style="padding:10px 14px; font-size:12.5px; color:var(--fern-deep); background:var(--fern-soft);">${msg}</div>`;
    // Suggestion de corpus path
    document.getElementById("corpus-path").value = d.output_dir;
  } catch(e) {
    statusDiv.innerHTML = `<div class="surface-flat" style="padding:10px 14px; font-size:12.5px; color:var(--err); background:oklch(0.95 0.03 28);">Erreur : ${_escapeHtml(e.message)}</div>`;
  }
}

// ─── Library (Mes corpus / Decouvrir) ────────────────────────────────────────
// Etat partage Library : liste des corpus locaux (cachee + index pour
// hydrater le dropdown du Benchmark).
let _libraryLocalCorpora = [];

function switchLibraryPane(pane) {
  const local = document.getElementById("library-pane-local");
  const disc = document.getElementById("library-pane-discover");
  // flex (pas block) pour activer le flex-direction:column + gap inline.
  if (local) local.style.display = pane === "local" ? "flex" : "none";
  if (disc) disc.style.display = pane === "discover" ? "flex" : "none";
  const lt = document.getElementById("ltab-local");
  const dt = document.getElementById("ltab-discover");
  if (lt) lt.classList.toggle("on", pane === "local");
  if (dt) dt.classList.toggle("on", pane === "discover");
  if (pane === "discover") {
    // Lazy-load au premier affichage.
    if (!_libraryDiscoverInited) {
      _libraryDiscoverInited = true;
      searchHTRUnited();
    }
  }
}
let _libraryDiscoverInited = false;

function switchLibrarySource(source) {
  document.querySelectorAll("#library-source-switch .source-chip").forEach(b => {
    b.classList.toggle("on", b.dataset.source === source);
  });
  const htr = document.getElementById("library-source-htr-united");
  const hf = document.getElementById("library-source-huggingface");
  if (htr) htr.style.display = source === "htr-united" ? "block" : "none";
  if (hf) hf.style.display = source === "huggingface" ? "block" : "none";
  if (source === "huggingface" && !_libraryHfInited) {
    _libraryHfInited = true;
    searchHuggingFace();
  }
}
let _libraryHfInited = false;

async function loadLibraryLocalCorpora() {
  /** GET /api/corpus/uploads → rend la grille des corpus locaux
   *  (pane "Mes corpus") ET alimente le dropdown du Benchmark. */
  const list = document.getElementById("library-local-list");
  try {
    const r = await fetch("/api/corpus/uploads");
    const d = await r.json();
    _libraryLocalCorpora = d.uploads || [];
    _updateLibraryHeroStats();
    loadCorpusOptions();
    if (!list) return;
    if (_libraryLocalCorpora.length === 0) {
      list.innerHTML = `<div class="empty">${t("library_local_empty")}</div>`;
      return;
    }
    list.innerHTML = _libraryLocalCorpora.map(u => {
      const missing = u.has_missing_gt
        ? `<span class="tag tag-butter">${_escapeHtml(t("upload_missing_gt"))}</span>` : "";
      const docs = (u.doc_count || 0).toLocaleString();
      return `<div class="ds-card">
        <div style="display:flex; justify-content:space-between; align-items:baseline; gap:10px;">
          <div class="ds-name">${_escapeHtml(u.corpus_id)}</div>
          ${missing}
        </div>
        <div class="ds-meta">
          <span>PAIRES · <b class="num">${docs}</b></span>
          <span class="mono" style="font-size:11px; color:var(--g-400);">${_escapeHtml(u.corpus_path)}</span>
        </div>
        <div class="row" style="margin-top:10px; justify-content:space-between;">
          <button class="btn btn-sm btn-primary" type="button" onclick="useCorpusInBenchmark('${_escapeAttr(u.corpus_path)}', '${_escapeAttr(u.corpus_id)}')">
            <span data-i18n="library_use_in_benchmark">Utiliser dans Benchmark</span>
          </button>
          <button class="btn btn-ghost btn-sm" type="button" onclick="deleteLibraryCorpus('${_escapeAttr(u.corpus_id)}')" title="${_escapeAttr(t("upload_delete"))}">✕</button>
        </div>
      </div>`;
    }).join("");
  } catch (e) {
    if (list) list.innerHTML = `<div class="empty" style="color:var(--err);">Erreur : ${_escapeHtml(e.message)}</div>`;
  }
}

function _updateLibraryHeroStats() {
  const setHero = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  setHero("hero-local-count", _libraryLocalCorpora.length);
  const totalPages = _libraryLocalCorpora.reduce((s, c) => s + (c.doc_count || 0), 0);
  setHero("hero-local-pages", totalPages.toLocaleString());
  const aside = document.getElementById("library-local-aside");
  if (aside) aside.textContent = `${_libraryLocalCorpora.length} CORPORA · ${totalPages.toLocaleString()} PAIRES`;
}

function _escapeAttr(s) {
  return String(s || "").replace(/'/g, "&#39;").replace(/"/g, "&quot;");
}

async function deleteLibraryCorpus(corpusId) {
  if (!confirm(lang === "fr" ? `Supprimer le corpus « ${corpusId} » ?` : `Delete corpus "${corpusId}"?`)) return;
  try {
    await fetch(`/api/corpus/uploads/${encodeURIComponent(corpusId)}`, { method: "DELETE" });
    loadLibraryLocalCorpora();
    // Reset corpus benchmark si on vient de virer le selectionne.
    const sel = document.getElementById("corpus-select");
    if (sel && sel.value && sel.value.includes(corpusId)) {
      sel.value = "";
      onCorpusSelectChange();
    }
  } catch (e) { /* silent */ }
}

function onLibraryUploadInput(event) {
  const files = Array.from(event.target.files || []);
  if (files.length > 0) uploadCorpusToLibrary(files);
}

function onLibraryDrop(event) {
  event.preventDefault();
  document.getElementById("library-dropzone").classList.remove("active");
  const files = Array.from(event.dataTransfer.files || []);
  if (files.length > 0) uploadCorpusToLibrary(files);
}

async function uploadCorpusToLibrary(files) {
  const progressContainer = document.getElementById("library-upload-progress");
  const progressBar = document.getElementById("library-upload-bar");
  const progressText = document.getElementById("library-upload-text");
  if (progressContainer) progressContainer.style.display = "block";
  if (progressBar) { progressBar.style.width = "10%"; progressBar.style.background = ""; }
  if (progressText) progressText.textContent = t("upload_uploading");

  const fd = new FormData();
  for (const f of files) fd.append("files", f);

  try {
    let pct = 10;
    const timer = setInterval(() => {
      pct = Math.min(pct + 5, 85);
      if (progressBar) progressBar.style.width = pct + "%";
    }, 200);

    const r = await fetch("/api/corpus/upload", { method: "POST", body: fd });
    clearInterval(timer);
    if (progressBar) progressBar.style.width = "100%";

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    const d = await r.json();
    if (progressText) progressText.textContent = `✓ ${t("upload_success")} — ${d.doc_count} ${t("upload_pairs")}`;
    if (progressBar) progressBar.style.background = "var(--ok)";
    // Recharge la liste (qui aussi refresh le dropdown Benchmark).
    loadLibraryLocalCorpora();
  } catch (e) {
    if (progressBar) { progressBar.style.width = "100%"; progressBar.style.background = "var(--err)"; }
    if (progressText) progressText.textContent = `✗ ${_escapeHtml(e.message)}`;
  }
}

// ─── Corpus picker in Benchmark (dropdown alimente par Library) ──────────────
function loadCorpusOptions() {
  /** Hydrate #corpus-select avec les corpus locaux de Library. */
  const sel = document.getElementById("corpus-select");
  if (!sel) return;
  const previous = sel.value;
  sel.innerHTML = `<option value="">${t("bench_corpus_pick_placeholder")}</option>`;
  for (const u of _libraryLocalCorpora) {
    const opt = document.createElement("option");
    opt.value = u.corpus_path;
    const docs = (u.doc_count || 0).toLocaleString();
    opt.textContent = `${u.corpus_id}  ·  ${docs} ${t("upload_pairs")}`;
    sel.appendChild(opt);
  }
  // Restaure la selection si toujours dispo.
  if (previous && Array.from(sel.options).some(o => o.value === previous)) {
    sel.value = previous;
  }
  const meta = document.getElementById("corpus-select-meta");
  if (meta) meta.textContent = `${_libraryLocalCorpora.length} ${lang === "fr" ? "disponible(s)" : "available"}`;
}

function onCorpusSelectChange() {
  const sel = document.getElementById("corpus-select");
  const pathInput = document.getElementById("corpus-path");
  const info = document.getElementById("corpus-info");
  if (!sel) return;
  const v = sel.value || "";
  if (pathInput) pathInput.value = v;
  if (v) {
    const u = _libraryLocalCorpora.find(c => c.corpus_path === v);
    if (info && u) info.textContent = `✓ ${u.corpus_id} (${(u.doc_count || 0).toLocaleString()} ${t("upload_pairs")})`;
  } else if (info) {
    info.textContent = "";
  }
}

function toggleCorpusFreeInput() {
  const free = document.getElementById("corpus-free-input");
  if (!free) return;
  free.style.display = free.style.display === "none" ? "block" : "none";
}

function useCorpusInBenchmark(corpusPath, corpusId) {
  /** Appele depuis Library : bascule sur la vue Benchmark et selectionne
   *  le corpus dans le dropdown. */
  showView("benchmark");
  // Garantit que le dropdown est synchronise avant de fixer la valeur.
  loadCorpusOptions();
  const sel = document.getElementById("corpus-select");
  if (sel) {
    sel.value = corpusPath;
    onCorpusSelectChange();
  }
  const info = document.getElementById("corpus-info");
  if (info) info.textContent = `✓ ${corpusId}`;
}

// Helper retro-compat : conserve par les anciens callers de setCorpusPath
// (file browser, upload preview, etc.).  Met a jour l'input free + l'info.
function setCorpusPath(path, label) {
  const pathInput = document.getElementById("corpus-path");
  const info = document.getElementById("corpus-info");
  if (pathInput) pathInput.value = path;
  if (info) info.textContent = `✓ ${label}`;
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
    profile: (document.getElementById("run-profile") || {}).value || "standard",
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
  if (typeof cfg.profile === "string") {
    const sel = document.getElementById("run-profile");
    if (sel) sel.value = cfg.profile;
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
  loadLibraryLocalCorpora();  // Hydrate Library + Benchmark corpus dropdown
  _setBenchState("READY");
  _updateCompetitorCounts();
  // Hero norm sync on profile change
  const normSel = document.getElementById("norm-profile");
  if (normSel) {
    normSel.addEventListener("change", () => {
      const h = document.getElementById("hero-norm");
      if (h) h.textContent = normSel.value || "—";
    });
  }
  // Close modal on backdrop click
  const importModal = document.getElementById("import-modal");
  if (importModal) {
    importModal.addEventListener("click", e => {
      if (e.target === importModal) closeImportModal();
    });
  }
});
