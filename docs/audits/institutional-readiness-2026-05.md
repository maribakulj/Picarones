# Audit institutionnel BnF — état de Picarones au 2 mai 2026

> Audit réalisé sur la branche `claude/audit-institutional-readiness-8Cw4w`
> à partir du commit `06aac6a` (merge PR #50). Méthode : 6 agents
> d'exploration en parallèle (architecture, code/sécurité, tests,
> documentation, CI/CD, web/i18n/accessibilité), suivis d'une vérification
> manuelle des findings critiques (`grep`, lecture ligne à ligne).
> Lint `ruff check picarones/ tests/` : passe. Suite complète :
> **3 356 passed, 3 skipped, 0 failed** en 3 min 04 s.
>
> **Cible** : adoption institutionnelle (BnF, BL, KBR, Archives nationales)
> et publication scientifique citable (JOSS, arXiv).
>
> **Verdict global** : **non prêt** pour estampille institutionnelle ou
> citation académique sans remédiation. Le code est solide, l'architecture
> claire, la couverture de tests inégalée. Les bloqueurs sont concentrés
> sur trois axes : **(1) communication scientifique** (CITATION, JOSS,
> citations primaires des méthodes statistiques), **(2) gouvernance et
> ops institutionnelles** (CSRF, accessibilité WCAG, déploiement, RGPD),
> **(3) hygiène d'intégration continue** (lock file, scanners de sécurité,
> seuil de couverture).
>
> **Effort estimé pour atteindre le niveau BnF** : 6 à 10 semaines
> calendaires (1 ETP), hors rédaction du papier JOSS qui suit son propre
> calendrier (8 à 12 semaines de revue par les pairs).

---

## 1. Résumé par sévérité

| Sévérité | Compte | Domaines |
|---|---|---|
| **BLOCKER** | 11 | Architecture (2), violation règle propre (3), publication scientifique (3), accessibilité (2), sécurité web (1) |
| **MAJOR** | 18 | CI/CD (6), documentation (5), tests (3), reproductibilité (2), web/UX (2) |
| **MINOR** | 17 | Polissage (DX, packaging, i18n résiduel, cache Docker, formats locales…) |
| **Faux positifs** | 1 | « SQL injection » dans `jobs.py:235` — détaillé en §6 |

Tous les findings sont accompagnés de la citation `fichier:ligne` exacte
et d'une esquisse de correction. Les efforts indiqués sont en
*personne-jours* (PJ) pour un ingénieur familier du repo.

---

## 2. Bloqueurs — à corriger avant tout estampillage institutionnel

### B-1 — Violation Cercle 2 → Cercle 3 dans `measurements/statistics.py`

**Fichier** : `picarones/measurements/statistics.py:861`

```python
def _extract_error_pairs(gt: str, hyp: str) -> list[tuple[str, str]]:
    from picarones.report.diff_utils import compute_word_diff   # ← Cercle 3 !
```

**Problème** : violation directe de la règle architecturale documentée
dans `CLAUDE.md` et `docs/architecture.md` (« les imports vont
uniquement de l'extérieur vers l'intérieur »). Un module de mesures
(Cercle 2) ne doit jamais dépendre du rendu (Cercle 3). Le import
est *paresseux* (à l'intérieur de la fonction), donc il ne casse pas
le démarrage, mais il rend le module `statistics` inutilisable
pour quiconque consomme Picarones sans la couche `report` (par exemple
un pipeline d'analyse en notebook ou un service externe).

**Correctif** : extraire `compute_word_diff` (et toute la famille
`diff_utils`) dans `picarones/core/diff_utils.py`. Le rendu HTML peut
continuer à le ré-exporter pour rétrocompatibilité.

**Effort** : 0,5 PJ. **Risque** : faible — le module diff_utils a déjà
ses tests dans `tests/report/test_diff_utils.py`, à déplacer.

---

### B-2 — Violation Cercle 2 → Cercle 3 dans `measurements/difficulty.py`

**Fichier** : `picarones/measurements/difficulty.py:195`

```python
def difficulty_color(score: float) -> str:
    from picarones.report.colors import COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE, COLOR_RED
```

**Problème** : identique à B-1. Pire : la fonction renvoie une couleur
CSS, donc c'est une logique purement de présentation qui s'est glissée
dans le module métier `difficulty`.

**Correctif** : déplacer `difficulty_color` dans
`picarones/report/difficulty_render.py` (à créer) et ne laisser dans
`difficulty.py` que la logique de scoring numérique. Les appelants du
côté `report/` font alors `from picarones.report.difficulty_render
import difficulty_color`.

**Effort** : 0,5 PJ.

---

### B-3 — Trois `except Exception: pass` qui violent la règle « jamais »

**Fichiers et lignes** :
- `picarones/extras/importers/huggingface.py:266` (recherche API silencieusement avalée)
- `picarones/extras/importers/huggingface.py:416` (échec de sauvegarde d'image silencieux)
- `picarones/extras/importers/htr_united.py:448` (parsing YAML silencieux → fallback démo)

**Problème** : règle écrite noir sur blanc dans `CLAUDE.md` :

> **Ne jamais mettre `except Exception: pass`** : remplacer par
> `logger.warning("[module] fonctionnalité dégradée : %s", e)`.

Conséquences concrètes pour un archiviste BnF qui importe un corpus
HTR-United : si le YAML distant est mal-formé, l'utilisateur reçoit un
catalogue de démo *sans aucun avertissement* — il croit consulter le
catalogue institutionnel. Pour une mainteneur, c'est un bug invisible
qui peut survivre des années.

**Correctif** : remplacer chaque `pass` par
`logger.warning("[importers] <opération> a échoué (mode dégradé) : %s", e)`
+ ajouter un `Fact` dans la synthèse du rapport quand le fallback est
déclenché côté utilisateur.

**Effort** : 0,5 PJ. **Test** : 1 cas par site (mock l'échec → vérifier
le log).

---

### B-4 — Aucune `CITATION.cff` ni preprint scientifique

**Fichiers manquants** : `CITATION.cff`, `paper.md` (JOSS), pas de DOI
dans le `README`, pas d'`ORCID` listés, pas de `.zenodo.json`.

**Problème** : pour qu'un article scientifique cite Picarones, il faut
au minimum un fichier `CITATION.cff` parsable par GitHub (qui produit
alors le bouton « Cite this repository »), idéalement un DOI Zenodo, et
en pratique un papier JOSS pour un projet de cette envergure (méthodes
statistiques nouvelles, registre de métriques typées, moteur narratif
factuel anti-hallucination — chacune de ces contributions est citable).

Une bibliothèque nationale n'adoptera pas un outil scientifique non
citable. Une thèse ou un article ne peut pas s'appuyer sur Picarones
si la référence se résume à une URL GitHub mutable.

**Correctif** :
1. Créer `CITATION.cff` (5 min), avec auteurs ORCID et version.
2. Pousser une release GitHub taggée + obtenir un DOI Zenodo (intégration
   automatique : 1 h).
3. Rédiger un `paper.md` (format JOSS, 6 à 8 pages) résumant : philosophie
   « banc d'essai », architecture en 3 cercles, contributions
   méthodologiques (Friedman + Nemenyi, registre typé, moteur narratif).

**Effort** : 1 PJ pour CITATION+Zenodo. Le papier JOSS : ~10 PJ d'écriture
+ 8 à 12 semaines de revue par les pairs.

---

### B-5 — Méthodes statistiques sans citation primaire dans le code

**Fichier** : `picarones/measurements/statistics.py` (1 127 lignes)

**Problème** : le module implémente Friedman, post-hoc Nemenyi, Wilcoxon,
bootstrap, intervalles de confiance, dominance Pareto. Or aucune
référence BibTeX/DOI n'apparaît dans les docstrings. Demšar 2006 (le
papier qui définit Friedman+Nemenyi pour la comparaison de classifieurs,
soit *exactement* ce que fait Picarones) n'est cité nulle part dans le
code. Idem Wilcoxon 1945, Efron 1979 (bootstrap).

Conséquence : un relecteur académique ou un responsable BnF en
évaluation ne peut pas vérifier que l'implémentation correspond aux
définitions canoniques. Le `glossary` (Sprint 21) mentionne les noms
des tests mais ne pointe pas vers les sources primaires.

**Correctif** : ajouter un en-tête de module dans `statistics.py` listant
les références (BibTeX), et un `:references:` dans la docstring de
chaque fonction publique. Ajouter le champ `reference` aux 25 entrées
du glossaire (déjà prévu dans le schéma — voir
`picarones/report/glossary/`, vérifier la complétude).

**Effort** : 1 PJ.

---

### B-6 — Profils de normalisation non tracés à des standards éditoriaux

**Fichiers** :
- `picarones/measurements/normalization.py` (420 lignes)
- `picarones/measurements/mufi.py` (cite « MUFI » sans préciser la version)
- `docs/profiles.md` (présent mais ne pointe ni vers TEI P5 ni vers
  MUFI registry ni vers DEAF)

**Problème** : Picarones revendique des profils `DIPLOMATIC_FR`,
`MUFI`, `EARLY_MODERN`, etc. Pour un médiéviste ou un éditeur critique,
la première question est : « quelle version de MUFI ? quelle
recommandation TEI ? quelle politique pour ſ→s ? ». Sans cette
traçabilité, on ne peut pas comparer un benchmark Picarones à une
édition TEI conforme aux standards de la communauté.

**Correctif** : créer `docs/normalization-specs.md` qui mappe chaque
profil :
- nom du profil
- version exacte de la spec source (MUFI v4.0, TEI P5 Unicode chapter
  3.4, DEAF ortho-2024, …)
- liste exhaustive des transformations appliquées
- DOI/URL stable de la spec
- date de révision

Ajouter un test de non-régression :
`tests/measurements/test_normalization_spec_consistency.py`.

**Effort** : 2 PJ (la connaissance experte est plus rare que le code).

---

### B-7 — Aucun scanner de sécurité dans la CI

**Fichier** : `.github/workflows/ci.yml`

**Manquants** : `bandit` (code Python), `pip-audit` ou `safety` (CVE
des dépendances), `trivy` (scan de l'image Docker), `gitleaks`
(détection de secrets dans l'historique). Pre-commit a `detect-private-key`
seulement.

**Problème** : projet exposant un endpoint public sur HuggingFace Space
avec dépendances cloud (mistralai, anthropic, openai, google-cloud-vision,
azure-ai-formrecognizer). Une CVE non détectée dans une de ces SDK est
une porte d'entrée. Pour BnF, c'est rédhibitoire — les revues
sécurité institutionnelles l'exigent.

**Correctif** : ajouter à `ci.yml` un job `security` parallèle aux tests :

```yaml
security:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.11" }
    - run: pip install bandit pip-audit
    - run: bandit -r picarones/ -ll  # niveau LOW+
    - run: pip-audit --strict
    - uses: aquasecurity/trivy-action@master
      with: { image-ref: 'picarones:latest', exit-code: '1', severity: 'HIGH,CRITICAL' }
```

**Effort** : 1 PJ (intégration + traitement des findings initiaux).

---

### B-8 — Aucun seuil de couverture appliqué (`--cov-fail-under` manquant)

**Fichier** : `.github/workflows/ci.yml:78`

```yaml
pytest tests/ -q --cov=picarones --cov-report=xml --cov-report=term-missing
```

**Problème** : la couverture est calculée et uploadée à Codecov, mais
aucun plancher n'est imposé. Une PR peut faire baisser la couverture de
85 % à 40 % sans qu'aucun signal CI ne se déclenche. Pour un projet
revendiquant 3 359 tests, c'est paradoxal : la rigueur affichée n'est
pas applicable.

**Correctif** : ajouter `--cov-fail-under=85` (mesurer le baseline d'abord
avec `pytest --cov` → fixer le plancher 2 points en dessous). Optionnel
mais recommandé : exporter le delta dans un commentaire de PR via
`coverage-comment-action`.

**Effort** : 0,25 PJ.

---

### B-9 — Graphiques Canvas inaccessibles aux lecteurs d'écran (WCAG 1.1.1 niveau A)

**Fichiers** :
- `picarones/report/templates/_app.js:1062`, `1102`, et autres
  instanciations Chart.js
- `picarones/report/vendor/chart.umd.min.js`

**Problème** : Chart.js produit du `<canvas>`. Sans intervention,
*aucun* contenu n'est exposé à l'AT (assistive technology). Un usager
non-voyant utilisant NVDA/JAWS n'entend qu'« graphique vide ». Cela
viole WCAG 2.1 succès 1.1.1 (Non-text Content) au niveau A — le plus
bas, donc rédhibitoire pour toute déclaration de conformité RGAA en
France.

**Correctif** : pour chaque graphique, ajouter en parallèle :
1. un `<table>` de données équivalentes, marqué
   `aria-describedby` du canvas, masqué visuellement (`visually-hidden`)
   mais lu par les AT ;
2. un `aria-label` descriptif sur le `<canvas>` ;
3. un bouton « Voir les données » qui révèle la table à tous (utile
   aussi pour la copie).

Alternative plus profonde : remplacer Chart.js par des SVG natifs avec
`<title>` et `<desc>` (déjà la pratique dans `pipeline_dag_render.py`,
`taxonomy_cooccurrence_render.py`, etc. — Sprints 64, 75 et al.).
Cohérent avec le reste de la base.

**Effort** : 2 PJ (8 à 12 graphiques Chart.js à doubler).

---

### B-10 — Pas de lien « Aller au contenu » (WCAG 2.4.1)

**Fichier** : `picarones/report/templates/base.html.j2` et `_header.html`

**Problème** : aucune occurrence de `skip`, `main-content` ou équivalent
dans les templates. Un usager-clavier doit traverser toute la
navigation et le panneau latéral avant d'atteindre le rapport. Violation
WCAG 2.1 succès 2.4.1 (Bypass Blocks) au niveau A.

**Correctif** : ajouter dans `_header.html`, premier enfant du `<body>` :

```html
<a href="#main" class="skip-link">{{ i18n.skip_to_content }}</a>
```

+ une classe CSS `.skip-link` qui reste cachée hors `:focus` ; et
ajouter `id="main"` sur le conteneur principal.

**Effort** : 0,25 PJ.

---

### B-11 — Aucune protection CSRF sur les endpoints POST

**Fichier** : `picarones/web/app.py` + 11 routers dans
`picarones/web/routers/`

**Problème** : tous les endpoints POST (`/api/corpus/upload`,
`/api/benchmark/start`, `/api/benchmark/run`, `/api/benchmark/{id}/cancel`,
`/api/config/save`, `/api/htr-united/import`, `/api/huggingface/import`,
`/api/lang/{code}`) acceptent les requêtes sans vérification d'origine
ni token CSRF.

Sur HuggingFace Space en mode public, l'impact est limité (pas de
session utilisateur authentifiée à voler). Mais en déploiement
institutionnel BnF (sur intranet, derrière SSO), un usager logué peut
être victime d'une page tierce qui poste vers `/api/config/save` ou
lance un benchmark coûteux à son insu.

**Correctif** : ajouter le middleware `starlette-csrf` ou équivalent,
piloté par variable d'environnement `PICARONES_CSRF_REQUIRED=1`. En
mode public HuggingFace : laissé désactivé (pas de session). En mode
institutionnel : activé d'office.

**Effort** : 1 PJ + tests.

---

## 3. Problèmes majeurs — à régler dans les 6 prochaines semaines

### M-1 — Pas de fichier de verrouillage des dépendances

**Symptôme** : `pyproject.toml` déclare 11 dépendances cœur et 7 extras
en `>=` sans borne haute. `requirements.txt` à la racine est
divergent et obsolète. Aucun `requirements.lock`, `uv.lock`,
`poetry.lock`.

**Conséquence** : un build Docker du 2 mai 2026 et un build du 2 mai 2027
ne produiront pas le même artefact. Pour un dépôt patrimonial qui doit
pouvoir rejouer un benchmark à 5 ans d'intervalle, c'est inacceptable.

**Correctif** : adopter `uv` ou `pip-tools`, générer `requirements.lock`
et l'épingler dans le `Dockerfile` (`pip install -r requirements.lock`
au lieu de `pip install .`). Régénérer mensuellement via un workflow
dédié + PR automatique.

**Effort** : 1 PJ.

---

### M-2 — Image Docker de base non épinglée

**Fichier** : `Dockerfile:18, 43`

```dockerfile
FROM python:3.11-slim AS builder
FROM python:3.11-slim AS runtime
```

**Correctif** : épingler au digest SHA256 :
`FROM python:3.11.10-slim@sha256:abc123…`. Régénérer trimestriellement.

**Effort** : 0,25 PJ.

---

### M-3 — Endpoint `/health` absent alors que le `HEALTHCHECK` Docker l'appelle

**Fichiers** : `Dockerfile:96` (`curl -f http://localhost:7860/health`)
vs `picarones/web/routers/system.py:13` (qui expose `/api/status`).

**Correctif** : aliaser `/health` → `/api/status` ou créer un endpoint
dédié, plus minimaliste (juste 200 OK + version, sans introspecter
l'état OCR).

**Effort** : 0,25 PJ.

---

### M-4 — Pas de type-checking dans la CI

**État** : `Makefile:100-102` propose un target `typecheck` qui appelle
mypy avec `--ignore-missing-imports --no-strict-optional`, mais n'est
pas appelé par `ci.yml`. Aucune section `[tool.mypy]` dans
`pyproject.toml`. Pas de marqueur `py.typed`.

**Correctif** : configurer mypy dans `pyproject.toml` avec
`strict = true` sur `picarones/core/` (le plus stable), `strict = false`
ailleurs comme état initial. Ajouter un job CI `typecheck` qui devient
bloquant pour `picarones/core/` et avertissant ailleurs. Marquer
`py.typed`.

**Effort** : 2 PJ (premier passage), puis maintenance continue.

---

### M-5 — Pas de pipeline de release vers PyPI

**Symptôme** : `pyproject.toml` épingle `version = "1.0.0"` en dur. Pas
de `setuptools_scm`. Pas de workflow `.github/workflows/release.yml`.
Picarones n'est pas installable via `pip install picarones`.

**Conséquence** : impossible de citer une version exacte (`picarones==1.2.3`)
dans un `requirements.txt` de notebook ou de papier. Toute installation
passe par `pip install git+https://…` (mutable, fragile).

**Correctif** : adopter `setuptools_scm` (version dérivée des tags Git)
+ workflow `release.yml` déclenché sur tag `v*` qui : build sdist+wheel
→ test sur `testpypi` → publie sur PyPI via `pypa/gh-action-pypi-publish`
avec OIDC trust (pas de token long-lived).

**Effort** : 1 PJ.

---

### M-6 — Pas d'image conteneur publiée immutable

**Symptôme** : `Makefile:167` tagge `picarones:latest` et `picarones:1.0.0`
localement mais ne pousse nulle part. HuggingFace Space rebuild à chaque
merge (donc pas un *artefact*, c'est une recompilation). Pas de
publication sur ghcr.io, Docker Hub, ou Quay.

**Correctif** : ajouter un workflow qui pousse vers
`ghcr.io/maribakulj/picarones:1.0.0` et `…:latest` à chaque release.
Avec digest fixe communiqué dans le `CHANGELOG`.

**Effort** : 0,5 PJ.

---

### M-7 — Pas de guide de déploiement institutionnel

**Manquant** : `docs/operations/deployment.md`, `docs/operations/backup.md`,
`docs/operations/data-retention.md`.

**Conséquence** : un DSI BnF qui veut héberger Picarones doit deviner :
- Quelle BD pour `jobs.sqlite` en multi-instance ?
- Comment migrer le schéma de l'historique longitudinal entre versions ?
- Combien de temps les uploads sont-ils conservés ? Politique RGPD ?
- Comment intégrer derrière un proxy SSO (Shibboleth, CAS, OIDC) ?
- Quelle observabilité (logs JSON pour ELK, métriques Prometheus) ?
- Comment sauvegarder/restaurer l'historique ?

INSTALL.md couvre uniquement Docker mono-instance HuggingFace. C'est
insuffisant.

**Correctif** : rédiger les 3 guides ci-dessus. Ajouter une section
RGPD au `SECURITY.md` (rétention des uploads, logs, IP du
rate-limiter).

**Effort** : 3 PJ.

---

### M-8 — Aucune politique de rétention des données ni mention RGPD

**Manquant** : politique explicite pour les uploads ZIP/images,
les logs (qui contiennent IP via le rate-limiter), l'historique
longitudinal SQLite.

**Conséquence** : sur Space public, un visiteur qui upload une image
patrimoniale ne sait pas combien de temps elle est gardée. Sur
déploiement institutionnel BnF, l'absence de politique bloque la
mise en production.

**Correctif** : doc `docs/operations/data-retention.md` + mécanisme
de purge automatique (cron job `purge_uploads_older_than(days=7)`)
+ mention RGPD dans le `README` et la home web.

**Effort** : 1,5 PJ.

---

### M-9 — Pas de déclaration d'accessibilité

**Manquant** : `ACCESSIBILITY.md` (recommandation gouvernementale FR
pour tout service public, RGAA 4.1 art. 47 de la loi 2005-102).

**Correctif** : déclaration explicite après audit RGAA + remédiation
des bloqueurs B-9 et B-10. Pour atteindre WCAG 2.1 niveau AA
(prérequis BnF), prévoir un audit externe après remédiation.

**Effort** : 1 PJ pour la déclaration + remédiation déjà comptée en B-9/B-10
+ audit externe (hors équipe).

---

### M-10 — Pas de divulgation de conflits d'intérêt

**Manquant** : déclaration sur la position de l'outil vis-à-vis des
fournisseurs cloud benchmarkés (OpenAI, Anthropic, Mistral, Google,
Azure). Pricing dans `picarones/data/pricing.yaml` (`last_updated:
2026-04-01`) sans validation indépendante ni veille automatique.

**Conséquence** : un papier qui s'appuie sur l'analyse de Pareto coût
de Picarones doit pouvoir citer une politique d'absence de COI. Sinon
un relecteur peut soupçonner un biais éditorial.

**Correctif** : ajouter une section « Conflicts of interest » dans le
`README` + `paper.md` JOSS, et un en-tête « Pricing as observed on
YYYY-MM-DD ; recompute with your own contracted rates » sur la vue
Pareto.

**Effort** : 0,5 PJ.

---

### M-11 — `CODEOWNERS` et politique de gouvernance absents

**Manquant** : `.github/CODEOWNERS`, `GOVERNANCE.md`, politique de
revue, cadence de release, SLO réponse aux issues.

**Conséquence** : une institution qui évalue la pérennité ne sait pas
s'il y a un mainteneur unique ou plusieurs, ni à quelle cadence elle
peut espérer un correctif.

**Correctif** : créer les deux fichiers. Cadence de release suggérée :
mensuelle pour les versions mineures, trimestrielle pour les majeures.
SLO suggéré (et tenable pour un projet de cette taille) : 5 jours
ouvrés pour un triage initial des issues.

**Effort** : 0,5 PJ + engagement de gouvernance.

---

### M-12 — Reproductibilité des snapshots sous-documentée

**État** : `picarones/report/snapshot.py` (266 lignes) et
`tests/report/test_sprint27_reproducibility_snapshots.py` existent.
Mais ni le `README` ni `docs/user/reading-a-report.md` n'expliquent :
- ce que contient un snapshot (versions OCR, modèles LLM, hash du code,
  hash du corpus, seeds…)
- comment recharger un snapshot pour rejouer un benchmark
- comment documenter un snapshot dans une publication

**Correctif** : créer `docs/reproducibility-snapshots.md`. Inclure
exemples reproductibles. Lier depuis `README` et `paper.md`.

**Effort** : 1 PJ.

---

### M-13 — Tests de concurrence runner / web sous-représentés

**Findings agents tests** :
- `picarones/measurements/runner.py` (1 019 lignes) n'a pas de test
  ciblant : épuisement du pool de processus, échecs partiels,
  processus zombies, `PICARONES_MAX_CONCURRENT_JOBS=32` sous charge.
- `picarones/web/jobs.py` : pas de test pour SSE `Last-Event-ID`
  reconnexion, écritures concurrentes SQLite (`SQLITE_BUSY`), bascule
  `PICARONES_PUBLIC_MODE=1` à chaud, isolation des jobs entre IPs.

**Correctif** : ajouter `tests/integration/test_runner_concurrency.py`
(50+ cas) et `tests/web/test_sse_reconnect.py`.

**Effort** : 3 PJ.

---

### M-14 — Pas de garde-fou anti-régression pour le benchmark lui-même

**Findings agent tests** : un *benchmarking platform* qui ne mesure pas
sa propre dérive de performance est suspect. Le job `regression_check`
dans `ci.yml:207-226` est commenté : « optionnel — activer si vous avez
un corpus de référence ».

**Correctif** : créer un mini-corpus de référence (10 documents libres
de droits couvrant les 3 strates principales : médiéval, imprimé
ancien, moderne) dans `tests/fixtures/reference_corpus/`. Ajouter un
job CI `--fail-if-cer-above 15.0` sur Tesseract+Pero. Exécuter
hebdomadairement (cron), pas à chaque PR (coût).

**Effort** : 2 PJ + sélection corpus.

---

### M-15 — Pas de timeout global pytest

**Fichier** : `.github/workflows/ci.yml:74-78`. Aucun `--timeout`. Un
test bloqué (Tesseract qui freeze, API LLM qui pend) bloque le runner
GH Actions jusqu'au timeout du job (6 h par défaut).

**Correctif** : ajouter `pytest-timeout` aux deps `[dev]`, configurer
`pyproject.toml` :
```toml
[tool.pytest.ini_options]
timeout = 300
timeout_method = "thread"
```

**Effort** : 0,1 PJ.

---

### M-16 — Pas de chargement paresseux pour les rapports volumineux

**Symptôme** : `picarones/report/generator.py` produit un fichier HTML
unique, images en base64. Un corpus de 1 000 documents × 5 moteurs
peut générer un fichier > 200 MB. Le navigateur peine.

**Correctif** : pour la galerie de documents, externaliser les images
dans `report-assets/<doc_id>.png` à côté du HTML, et lazy-loader
(`loading="lazy"`). Optionnel : pagination côté client.

**Effort** : 1 PJ. Garder l'option « monolithique » pour les petits
corpus (par défaut < 50 docs).

---

### M-17 — Documentation déséquilibrée FR/EN

**Constat** : README bilingue ✓. UI/glossaire bilingues ✓. Mais SPECS.md,
CHANGELOG.md, `docs/user/reading-a-report.md`, `docs/case-studies/`,
les guides développeur (4 fichiers) sont en français pur. Un chercheur
britannique ou allemand qui veut contribuer ne peut pas lire les guides
développeur. Un mainteneur qui veut publier le projet sur arXiv doit
réécrire toute la documentation utilisateur en anglais.

**Correctif** : traduire prioritairement (1) `docs/user/reading-a-report.md`,
(2) `docs/developer/index.md` + les 3 sous-guides, (3) `CONTRIBUTING.md`.
Laisser CHANGELOG et SPECS en français pour l'instant — moins critique.

**Effort** : 2 PJ pour les 5 documents prioritaires.

---

### M-18 — Pas de `.dockerignore` ni de `.env.example`

**Symptômes** :
- Pas de `.dockerignore` à la racine → `git`, `docs/`, `tests/` copiés
  inutilement dans l'image (taille +20 %, cache hit dégradé).
- `docker-compose.yml` référence `${OPENAI_API_KEY}`, `${PICARONES_PORT}`
  sans `.env.example` → les utilisateurs doivent deviner.

**Correctif** : 2 fichiers, 30 lignes chacun.

**Effort** : 0,1 PJ.

---

## 4. Problèmes mineurs — à intégrer en backlog

| # | Item | Fichier:ligne | Effort |
|---|------|---------------|--------|
| m-1 | Hardcoded FR `'Données d'ancrage non disponibles.'` bypass i18n | `_app.js:1087` | 0,1 PJ |
| m-2 | Hardcoded FR `'Données Gini non disponibles.'` (fallback) | `_app.js:1049` | 0,1 PJ |
| m-3 | Boutons « Réinitialiser » sans clé i18n | `_header.html:25` | 0,1 PJ |
| m-4 | Tableaux HTML sans `scope="col"` sur `<th>` | templates `view_*.html` | 0,3 PJ |
| m-5 | Palette heatmap non daltonien-friendly | `_styles.css` + `colors.py` | 0,5 PJ |
| m-6 | Nombres dans tableaux non localisés (1234567 vs 1 234 567) | `_app.js` (toLocaleString) | 0,3 PJ |
| m-7 | Pre-commit non rejoué en CI (bypassable via `--no-verify`) | `.github/workflows/ci.yml` | 0,1 PJ |
| m-8 | CI ne teste pas Python 3.13 (alors que `requires-python = ">=3.11"`) | `ci.yml:34` | 0,1 PJ |
| m-9 | API stability tests ne valident pas les `default values` des signatures | `tests/core/test_public_api.py` | 0,3 PJ |
| m-10 | Tests cloud OCR sans cas d'erreur HTTP (429, 401, 503) | `tests/engines/test_engines_cloud.py` | 0,5 PJ |
| m-11 | Versionnement des testdata absent (`tests/.testdata_versions.yaml`) | `tests/` | 0,2 PJ |
| m-12 | Numérotation sprint des fichiers de tests : trous (1, 37, 41, 43…) | `tests/` | 0,1 PJ (audit + nettoyage) |
| m-13 | `requirements.txt` racine partiellement divergent de `pyproject.toml` | `requirements.txt` | 0,1 PJ |
| m-14 | Pas de staleness check automatique sur `pricing.yaml` | générateur | 0,3 PJ |
| m-15 | `picarones.spec` (PyInstaller) avec `hiddenimports` manuels | `picarones.spec:45-98` | 0,5 PJ |
| m-16 | Aucun module `extras/historical/` ni `extras/importers/` séparé en package | `pyproject.toml:84-97` | 1 PJ (refactor planifié déjà documenté) |
| m-17 | `tests/measurements/test_sprint11_i18n_english.py` importe `report.generator` | `tests/measurements/` | 0,2 PJ (déplacer en `tests/integration/`) |

---

## 5. Points forts — à préserver et à valoriser dans la communication

Pour qu'un audit institutionnel soit *crédible*, il doit aussi nommer
explicitement ce qui marche. Les points suivants sont **au-dessus** de
ce qu'on observe dans 90 % des projets de recherche similaires :

1. **Architecture en 3 cercles tenue à 99 %.** Cercle 1 (`picarones/core/`)
   n'a aucune dépendance vers Cercles 2 ou 3. L'API publique
   (`picarones/__init__.py`) ré-exporte uniquement Cercle 1 — surface
   stable, contrat clair (`docs/api-stable.md`). Les 2 violations
   identifiées (B-1, B-2) sont **circonscrites et faciles à corriger**.

2. **Discipline de code rigoureuse.** Lint `ruff` 0 erreur. Logger
   nommé par module systématiquement. 0 `print()` en code métier.
   3 `TODO/FIXME` dans tout le repo (signe rare). 87 `except Exception`
   au total mais **84 sont annotés `# noqa: BLE001` avec contexte
   explicite**, seuls les 3 du B-3 sont des vraies violations.

3. **Sécurité de fond solide.**
   - XML défendu par `defusedxml` partout (XXE / Billion Laughs).
   - Zip-slip prévenu par `Path(member.filename).name` dans
     `web/corpus_utils.py:182-183`.
   - Toutes les requêtes SQLite paramétrées (le `f-string` de
     `jobs.py:235` est un faux positif — voir §6).
   - Aucun `pickle.load()` (vecteur de RCE classique).
   - `subprocess` utilisé une seule fois (`snapshot.py:186`,
     `git rev-parse HEAD` — args hardcodés, `timeout=2`,
     `stderr=DEVNULL`, gestion d'exception explicite).
   - Mode public (`PICARONES_PUBLIC_MODE`) avec gating des moteurs
     cloud, rate-limiting par IP, `Image.verify()` anti-bombe de
     décompression, en-têtes CSP / X-Content-Type-Options /
     X-Frame-Options.

4. **Couverture de tests volumineuse et structurée.** 3 359 tests
   collectés, organisés par cercle (`tests/core`, `tests/measurements`,
   `tests/engines`, `tests/web`, `tests/integration`, etc.). Tests
   d'API publique (`tests/core/test_public_api.py`) garantissant la
   stabilité du contrat externe. Pas de test fantôme `assert True`.

5. **Neutralité éditoriale exemplaire.** La règle « Picarones mesure et
   classe — il ne tranche pas le débat éditorial » est tenue jusque
   dans le moteur narratif (chaque nombre rendu est traçable au
   `payload` du `Fact` correspondant — anti-hallucination *prouvé*
   par tests). Les 5 « leviers d'amélioration » (Sprint 51) sont
   explicitement factuels, pas prescriptifs. Les profils diplomatique
   vs modernisant sont rapportés sans verdict.

6. **Reproductibilité partielle déjà en place.** Snapshot bit-à-bit
   identique sur même entrée (Sprint 27, vérifié par tests). Run
   save/load (Sprint 25). Comparaison de runs (Sprint 26). Manque
   uniquement la doc utilisateur (M-12) pour valoriser.

7. **Documentation interne (CLAUDE.md, CHANGELOG.md, SPECS.md)
   exceptionnellement détaillée.** Le journal des sprints permet à
   un nouveau contributeur ou à un auditeur de comprendre l'évolution
   de chaque décision.

8. **Politique de modules contribués (Sprint 97) déjà formalisée.**
   `core/module_policy.py` + `docs/developer/module-policy.md`. Picarones
   a anticipé le passage à un écosystème de plugins externes — rare
   pour un projet de cette taille.

---

## 6. Faux positifs identifiés et écartés

### F-1 — « SQL injection » dans `picarones/web/jobs.py:235`

L'agent code-quality a flagué cette ligne :

```python
c.execute(
    f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?",
    values,
)
```

**Vérification manuelle** (lecture des lignes 210-238) : la liste
`fields` est construite *exclusivement* à partir de littéraux Python
hardcodés (`"progress = ?"`, `"current_engine = ?"`, etc.) selon des
branches `if X is not None`. À aucun moment un input utilisateur n'y
arrive. Tous les `values` correspondants sont bien paramétrés via `?`.

**Verdict** : pas une vulnérabilité d'injection SQL. Au pire, un *style
fragile* qui pourrait inviter à l'erreur lors d'un futur refactor. À
laisser tel quel ou à refactorer en `m-18` (mineur de polissage).

---

### F-2 — « Pas de `--cov-fail-under` » classé blocker par certains agents

L'agent docs et l'agent CI ont tous deux insisté. C'est bloquant **pour
l'institution** (B-8) mais pas pour la communauté open-source. Je l'ai
gardé en BLOCKER vu la cible BnF.

---

### F-3 — Allégations de couverture de test divergentes (1 072 vs 3 354)

`CLAUDE.md` cite « 1 072 passed » dans la section *État actuel
(Sprint 16)* puis « ~3 354 passed » plus loin (*Contexte développement*).
Le second chiffre est correct (3 359 tests collectés au 2 mai 2026).
La première mention est obsolète depuis le Sprint 16 — à mettre à jour.
Effort : 0,01 PJ (un edit).

---

## 7. Feuille de route synthétique (10 semaines, 1 ETP)

| Semaine | Sprint d'audit | Livrables |
|---------|----------------|-----------|
| 1 | **S-A1 Architecture** | B-1, B-2, B-3 (violations + importers). Tests verts. |
| 1-2 | **S-A2 Sécurité CI** | B-7 (scanners), B-8 (cov threshold), M-15 (timeout pytest). |
| 2-3 | **S-A3 Web/Accessibilité** | B-9 (Chart.js a11y), B-10 (skip-link), B-11 (CSRF), m-1 à m-4 (i18n résiduel + scope). |
| 3-4 | **S-A4 Reproductibilité ops** | M-1 (lock file), M-2 (digest Docker), M-3 (/health), M-12 (doc snapshots), M-18 (.dockerignore + .env.example). |
| 4-5 | **S-A5 Publication scientifique** | B-4 (CITATION + Zenodo), B-5 (refs primaires statistics), B-6 (normalization specs). |
| 5-6 | **S-A6 Distribution** | M-5 (PyPI release), M-6 (image ghcr.io), M-11 (CODEOWNERS + governance). |
| 6-7 | **S-A7 Documentation institutionnelle** | M-7 (deployment guide), M-8 (data retention RGPD), M-9 (ACCESSIBILITY.md), M-10 (COI), M-17 (traduction EN). |
| 7-8 | **S-A8 Robustesse runner+web** | M-13 (tests concurrence), M-14 (anti-régression CER), M-16 (lazy loading reports). |
| 8-9 | **S-A9 Type-checking** | M-4 (mypy strict sur core, gradient ailleurs). |
| 9-10 | **S-A10 Polissage final + audit externe** | Backlog mineur restant + audit externe RGAA + audit externe sécurité. |

En parallèle (n'occupe pas le ETP) : **rédaction du papier JOSS** par
le ou les auteurs académiques (8 à 12 semaines, dont 4 à 6 de revue
par les pairs). Recommandation : démarrer dès la semaine 1.

---

## 8. Synthèse pour la direction

Picarones est un projet de recherche **techniquement solide, méthodologiquement
ambitieux, éditorialement neutre**. Il dispose déjà de la majorité des
briques d'une plateforme institutionnelle :
architecture cohérente, sécurité de fond, tests volumineux, snapshots
reproductibles, anti-hallucination prouvé.

Ce qui manque pour une adoption BnF / Bibliothèque nationale et pour
une citation académique se concentre sur **trois axes orthogonaux** au
code lui-même :

1. **Communication scientifique** (CITATION, JOSS, traçabilité des
   méthodes statistiques et des profils éditoriaux) — sans cela, le
   projet n'est pas citable et donc pas crédible pour un papier ou
   une thèse.
2. **Conformité opérationnelle** (CSRF, accessibilité WCAG niveau A,
   guides de déploiement, RGPD, gouvernance) — sans cela, aucune
   institution publique française ou européenne ne peut le mettre
   en production sur ses infrastructures.
3. **Hygiène CI/CD** (lock file, scanners, seuil de couverture,
   release PyPI, image immutable) — sans cela, la promesse de
   « plateforme reproductible et auditable » n'est pas tenue de bout
   en bout.

Avec 6 à 10 semaines d'investissement par un ingénieur senior + le
calendrier propre du papier JOSS, le projet peut atteindre un état
**publiable et adoptable institutionnellement**. Le code lui-même
nécessite peu de retouches profondes — l'essentiel du travail est
documentation, gouvernance, intégration continue et accessibilité.
