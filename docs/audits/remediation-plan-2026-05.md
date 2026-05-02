# Plan de remédiation — Picarones vers le niveau BnF / British Library

> Réponse opérationnelle à
> [`institutional-readiness-2026-05.md`](institutional-readiness-2026-05.md)
> (13 BLOCKERS, 28 MAJORS, 18 MINORS, 1 faux positif).
> Cible : 0 BLOCKER, 0 MAJOR ouvert ; CITATION valide ; audit RGAA AA passé ;
> `pip install picarones` fonctionnel ; lock file utilisé en prod.
>
> **15 sprints, 8 phases, ~58 PJ, ~12 semaines en 1 ETP.**
> Avec 2 ETP et la parallélisation décrite §2 : ~7–8 semaines.

---

## 1. Principes directeurs

Six principes structurent le séquençage. Chacun est tenu sur **toute** la
durée du plan, pas seulement à un sprint isolé.

1. **Scaffolding avant contenu.** Les *garde-fous CI* (Phase 0)
   sont posés en premier pour que les sprints suivants ne puissent pas
   régresser sur ce qu'on vient de corriger. Sans cela on déboguerait des
   régressions au lieu d'avancer.

2. **Tests avant fixes.** Chaque correctif d'audit s'accompagne d'un test
   de non-régression dans le même PR. Un fix sans test est une dette qui
   se rouvrira au sprint suivant.

3. **DRY entre code et documentation.** Une assertion vérifiable (compteur
   de tests, liste des moteurs, liste des commandes CLI) doit être
   *générée* depuis le code, pas dupliquée à la main. Le sprint A2 pose
   ce principe pour le README et l'audit.

4. **Source unique de vérité.** À chaque divergence entre deux documents
   (CLAUDE.md vs README vs SPECS), on désigne le canon et on déprécie
   le reste. Pas de patch parallèle.

5. **Refonte documentation produit en dernier.** Le README et SPECS
   reflètent un état du code. On ne refait pas ces documents tant que
   les phases 1–6 ne sont pas stabilisées, sinon on les refait deux fois.

6. **Parallélisation contrôlée.** Les phases 3 (accessibilité) et 4
   (reproductibilité opérationnelle) touchent des fichiers disjoints :
   templates HTML / CSS / JS d'un côté, Dockerfile / pyproject /
   workflows GitHub de l'autre. Elles peuvent tourner en parallèle si
   l'équipe a deux ETP. Avec un seul ETP, séquentiel.

---

## 2. Vue d'ensemble — 15 sprints en 8 phases

### Tableau récapitulatif

| Phase | Sprint | Thème | PJ | Sem. (1 ETP) | Items audit |
|---|---|---|---|---|---|
| 0 | A1 | Hardening CI | 4 | 1 | B-7, B-8, M-4, M-15, m-7, m-8, m-9 |
| 0 | A2 | Tests de cohérence documentation | 3 | 2 | (préparation A13, m-12) |
| 1 | A3 | Refactor cercles + importers | 3 | 3 | B-1, B-2, B-3, m-17 |
| 2 | A4 | Sécurité web (CSRF, /health) | 3 | 4 | B-11, M-3 |
| 2 | A5 | Concurrence et performance | 5 | 4–5 | M-13, M-14, M-16, m-10 |
| 3 | A6 | WCAG niveau A (bloquant) | 3 | 5–6 | B-9, B-10, m-3, m-4 |
| 3 | A7 | WCAG AA + i18n résiduel + a11y statement | 3 | 6 | m-1, m-2, m-5, m-6, M-9 |
| 4 | A8 | Reproductibilité opérationnelle | 3 | 5–6 *(parallèle phase 3)* | M-1, M-2, M-12, M-18, m-11, m-13, m-14 |
| 4 | A9 | Distribution PyPI + ghcr.io + releases | 3 | 7 | M-5, M-6, m-15, m-16 |
| 5 | A10 | Politiques de gouvernance | 2 | 8 | M-10, M-11 |
| 5 | A11 | Documentation institutionnelle | 5 | 8–9 | M-7, M-8, M-17 |
| 6 | A12 | Publication scientifique | 5 *(+ peer review externe)* | 9–10 | B-4, B-5, B-6 |
| 7 | A13 | Refonte README | 4 | 10–11 | B-13, M-19 à M-28, m-18, §9.3 |
| 7 | A14 | Refonte SPECS.md | 3 | 11 | B-12 |
| 8 | A15 | Audits externes (RGAA + sécurité) | 1 *(+ cycle externe)* | 11–12 | validation finale |
| 4* | A16 | Build Docker reproductible (digest + lock file) | 1 | *(post-A14)* | M-2 (clôture) |
| **TOTAL** | | | **~59 PJ** | **~12 sem. (1 ETP)** | **60 items** |

> Note : Sprint A16 a été ajouté après l'exécution d'A14 pour clôturer
> M-2 bout-en-bout (digest sha256 sur les deux ARG + lock file
> ``requirements-docker.lock`` consommé par le Dockerfile). La dette
> résiduelle ``apt-get`` non figé est tracée comme nouvel item M-29
> dans `institutional-readiness-2026-05.md` (différé post-v1.2).

### Diagramme de Gantt synthétique

```
Sem. 1   2   3   4   5   6   7   8   9   10  11  12
Phase 0  ████░░  ←  CI hardening + doc consistency tests (gates)
Phase 1      ███   ←  refactor cercles + importers
Phase 2          █████   ←  web sécurité + concurrence/perf
Phase 3              ██████   ←  a11y niveau A puis AA
Phase 4              █████   ←  reproductibilité ops + distribution (parallèle)
Phase 5                      ████   ←  gouvernance + doc institutionnelle
Phase 6                          █████   ←  CITATION + JOSS draft
                                 ════════════════════  ←  peer review externe (calendrier propre)
Phase 7                                  █████   ←  refonte README + SPECS
Phase 8                                       ███   ←  audits externes
                                              ════  ←  audit RGAA externe (calendrier propre)
```

Légende : `█` = sprint actif (1 ETP) ; `═` = calendrier externe en parallèle.

### Dépendances dures

- **A1 doit précéder tout** : sans seuil de couverture, scanners de sécurité
  et timeout pytest, les sprints suivants ne peuvent pas être validés en CI.
- **A2 doit précéder A13/A14** : les tests de cohérence documentation
  posés en A2 deviennent les *gates* qui valident la refonte README/SPECS.
- **A3 doit précéder A12** : les violations Cercle 2→3 doivent être
  réparées avant que SPECS et le papier JOSS décrivent l'architecture.
- **A8 doit précéder A12** : le snapshot de reproductibilité doit être
  documenté avant qu'un papier puisse promettre la reproductibilité.
- **A9 doit précéder A12** : un papier qui pointe vers `pip install
  picarones` exige que ça fonctionne.
- **Phases 1–6 doivent précéder Phase 7** : on ne refait pas le README
  tant que le code n'est pas stable.

### Dépendances souples (parallélisables avec 2 ETP)

- A6+A7 (a11y) ⫽ A8+A9 (ops/distribution) — fichiers disjoints.
- A11 (doc institutionnelle) peut commencer dès la fin de A8.
- A12 démarre dès que A3+A8+A9 sont clos ; sa rédaction peut chevaucher A13/A14.

---

## 3. Phase 0 — Garde-fous (sem. 1–2)

> **Objectif** : aucun sprint ultérieur ne peut faire régresser ce qu'on
> a corrigé, parce que la CI le détecte au PR. Sans cette phase,
> chaque correctif est susceptible de se rouvrir trois sprints plus tard.

### Sprint A1 — Hardening CI (4 PJ)

**Pourquoi à cette position dans la séquence**

Les phases suivantes corrigent des dizaines de fichiers. Sans seuil de
couverture, sans scanners de sécurité, sans type-check, sans timeout
pytest et sans validation pre-commit en CI, chaque correctif est une
prise de risque silencieuse : on apprend la régression à la PR suivante,
voire en production. C'est le sprint qui rend tous les autres exécutables
de manière crédible.

**Items de l'audit résolus**

**B-7** (scanners sécurité CI), **B-8** (cov-fail-under), **M-4**
(mypy en CI), **M-15** (pytest-timeout), **m-7** (pre-commit non rejoué
en CI), **m-8** (Python 3.13 manquant de la matrice), **m-9** (API
stability defaults).

**Livrables concrets**

- `.github/workflows/ci.yml` : ajout d'un job `security` (bandit + pip-audit
  + trivy sur l'image Docker), ajout `--cov-fail-under=85` au job `tests`,
  ajout `pytest-timeout=300` global, ajout d'un job `typecheck` (mypy
  strict sur `picarones/core/`, lax ailleurs), ajout Python 3.13 à la
  matrice (mode warning, pas bloquant 6 mois).
- `.github/workflows/precommit.yml` (nouveau) : rejoue tous les hooks
  pre-commit en CI pour empêcher le bypass `--no-verify`.
- `pyproject.toml` : `[tool.mypy]` avec `strict = true` sur
  `picarones.core.*`, `[tool.pytest.ini_options]` `timeout = 300` et
  `timeout_method = "thread"`. Ajout `pytest-timeout` à `[dev]`.
- `picarones/py.typed` (marqueur PEP 561 pour signaler le typage aux
  consommateurs externes).
- `tests/core/test_public_api_signatures.py` : pour chaque fonction
  exposée par `picarones/__init__.py`, vérifier les valeurs par défaut
  des paramètres via `inspect.signature` (couvre **m-9**).

**Critères d'acceptation**

- [ ] `ruff check picarones/ tests/` passe (régression nulle).
- [ ] `pytest tests/` passe avec `--cov-fail-under=85`.
- [ ] `bandit -r picarones/ -ll` passe en CI.
- [ ] `pip-audit --strict` passe en CI.
- [ ] `mypy picarones/core/ --strict` passe.
- [ ] Un PR qui supprime un default value d'une fonction publique fait
      échouer la CI sur `test_public_api_signatures`.
- [ ] Un PR qui dépasse 5 minutes sur un test individuel fait échouer
      la CI avec un message explicite, pas un hang.

**Risques et mitigation**

- *Risque* : seuil de couverture initialement fixé trop haut, premier PR
  bloqué. *Mitigation* : mesurer le baseline avant de fixer le seuil
  (`pytest --cov` sur `main`), poser le plancher 2 points en dessous.
- *Risque* : `mypy --strict` sur `core/` révèle 50 erreurs cachées.
  *Mitigation* : démarrer par `core/` qui est le plus stable et le mieux
  typé ; les autres cercles passent en mode `--no-strict-optional`
  pendant 1 sprint, durci en A11.

---

### Sprint A2 — Tests de cohérence documentation (3 PJ)

**Pourquoi à cette position dans la séquence**

L'audit §9 montre que README liste un moteur Kraken qui n'existe pas,
documente des variables AWS sans adapter, annonce 1 242 tests au lieu
de 3 356, et liste 9 commandes CLI au lieu de 15. Ces erreurs ne
viennent pas d'incompétence — elles viennent du fait que le README
n'a aucun *gardien* automatique. Si on refait le README en A13 sans
poser ce garde-fou, il dérivera à nouveau dès le sprint suivant.

A2 pose donc des **tests de cohérence** entre la documentation
publiée et le code. Ces tests deviennent ensuite les *gates* qui
valident A13 et A14.

**Items de l'audit résolus**

Préparation des **gates** pour M-19 (compteur tests), M-23
(moteurs annoncés), M-24 (variables env sans adapter), M-25 (CLI),
M-26 (API web), M-27 (métriques), M-28 (sections rapport). **m-12**
(numérotation sprint des tests).

**Livrables concrets**

- `tests/docs/test_readme_consistency.py` (nouveau) :
  - parse les tableaux Markdown du README ;
  - pour chaque moteur listé, vérifie qu'un fichier
    `picarones/engines/{nom_normalisé}.py` existe ;
  - pour chaque commande CLI listée, vérifie qu'elle apparaît dans
    `picarones --help` ;
  - pour chaque endpoint web listé, vérifie qu'il apparaît dans
    `app.openapi()["paths"]` ;
  - pour la phrase « pytest tests/ → N passed », vérifie que N
    correspond au baseline collecté par `pytest --collect-only`.
- `tests/docs/test_specs_consistency.py` (nouveau) : même approche pour
  SPECS.md, avec acceptation explicite des sections marquées « Reporté »
  ou « Abandonné au profit de … » (lecture des balises).
- `tests/docs/test_changelog_links.py` (nouveau) : vérifie que toute
  référence `Sprint N` dans CHANGELOG correspond à une entrée existante.
- `Makefile` : cible `make doc-check` qui lance ces tests seuls (utile
  pour l'auteur du README).
- `docs/developer/doc-consistency.md` (nouveau, ~80 L) : explique le
  contrat (« si vous ajoutez un moteur, ajoutez la ligne dans le tableau
  README ; le test la valide »).
- `tests/__init__.py` ou `conftest.py` : helper qui audite la
  numérotation `test_sprintNN` et signale les trous suspects (couvre
  **m-12**, sortie informative pas bloquante).

**Critères d'acceptation**

- [ ] `pytest tests/docs/` passe sur l'état actuel des docs **après
      avoir corrigé le README et SPECS minimalement** (suppression
      simple des fausses promesses pour ne pas bloquer le présent
      sprint ; refonte complète en A13/A14).
- [ ] Un PR qui ajoute un nouvel adapter OCR sans mettre à jour le
      tableau README échoue à `pytest tests/docs/test_readme_consistency.py`.
- [ ] Un PR qui supprime une commande CLI sans mettre à jour le README
      échoue de la même manière.
- [ ] `make doc-check` produit un rapport lisible en < 5 s.

**Risques et mitigation**

- *Risque* : les tests sont trop stricts et bloquent un PR légitime
  (ex : moteur en cours d'ajout, doc à jour ensuite). *Mitigation* :
  tolérer une « exception déclarée » via une balise HTML invisible
  `<!-- doc-check: skip-engine -->` dans le tableau, à utiliser avec
  modération et auditée à la PR.
- *Risque* : la mise à jour minimale du README/SPECS pour faire passer
  le test interfère avec la refonte A13/A14. *Mitigation* : se limiter
  à *supprimer* les promesses fausses (Kraken ligne, AWS env vars), pas
  à *ajouter* du contenu nouveau — ça reste pour A13.

---

## 4. Phase 1 — Hygiène architecturale (sem. 3)

> **Objectif** : ramener l'architecture à 100 % de conformité au modèle
> 3 cercles avant que la documentation produit (Phase 7) ou un papier
> JOSS (Phase 6) ne décrivent l'architecture.

### Sprint A3 — Refactor cercles + importers (3 PJ)

**Pourquoi à cette position dans la séquence**

L'audit §2 identifie deux violations Cercle 2 → Cercle 3
(`measurements/statistics.py:861` importe `report.diff_utils` ;
`measurements/difficulty.py:195` importe `report.colors`) et trois
`except Exception: pass` qui violent la règle propre du projet.
Ces dettes architecturales sont locales mais doivent être payées
*avant* la Phase 6 : un papier JOSS qui décrit une architecture en
3 cercles ne peut pas se faire mentir par le code. De plus, c'est
**le sprint le moins risqué** des phases suivantes — il rode l'équipe
sur le pattern *fix + test de non-régression* posé par A1.

**Items de l'audit résolus**

**B-1** (statistics.py:861 → core/), **B-2** (difficulty.py:195 →
report/), **B-3** (3 importers `except Exception: pass`), **m-17**
(`tests/measurements/test_sprint11_i18n_english.py` importe Cercle 3 →
déplacement en `tests/integration/`).

**Livrables concrets**

- Création de `picarones/core/diff_utils.py` (déplacement depuis
  `picarones/report/diff_utils.py`). `picarones/report/diff_utils.py`
  devient un ré-export trivial pour rétrocompat. `statistics.py:861`
  importe désormais depuis `core`.
- Création de `picarones/report/difficulty_render.py` qui contient
  `difficulty_color()`. `picarones/measurements/difficulty.py` ne
  contient plus que la logique numérique.
- `picarones/extras/importers/huggingface.py:266, 416` : remplacer les
  deux `except Exception: pass` par
  `logger.warning("[importers/hf] <opération> a échoué (mode dégradé) : %s", e)`.
- `picarones/extras/importers/htr_united.py:448` : idem.
- Émettre un `Fact` `IMPORTER_FALLBACK_TRIGGERED` (priorité MEDIUM,
  template factuel sans chiffres en dur) pour que la synthèse du
  rapport mentionne l'incident à l'utilisateur final.
- Déplacement de `tests/measurements/test_sprint11_i18n_english.py`
  vers `tests/integration/test_sprint11_i18n_english.py` (couvre **m-17**).
- Déplacement de `tests/measurements/test_sprint94_error_absorption.py`
  vers `tests/integration/` (audit §2 MINOR 4).
- Tests de non-régression :
  - `tests/core/test_diff_utils.py` : reproduire les tests de
    `tests/report/test_diff_utils.py` au nouveau chemin (le doublon
    sur `report/` reste pour la rétrocompat).
  - `tests/measurements/test_difficulty_pure.py` : vérifier que
    `picarones.measurements.difficulty` n'importe plus rien depuis
    Cercle 3 (`importlib.util.find_spec` + analyse AST).
  - `tests/extras/test_importer_warnings.py` : vérifier que chaque
    chemin d'erreur des 3 importers loggue un warning explicite
    (capturer via `caplog`).
- Garde-fou architectural : `tests/core/test_circle_dependencies.py`
  (nouveau) qui parse les imports de tous les fichiers
  `picarones/measurements/`, `engines/`, `llm/`, `pipelines/`,
  `modules/` et **échoue** si l'un d'eux importe `picarones.report.*`,
  `picarones.cli.*`, `picarones.web.*`, `picarones.extras.*`.

**Critères d'acceptation**

- [ ] `pytest tests/` passe (régression nulle ; rappel : 3 356 baseline).
- [ ] `tests/core/test_circle_dependencies.py` rapporte 0 violation.
- [ ] `grep -rn "except Exception:$" picarones/extras/importers/` renvoie
      0 ligne (les 3 violations B-3 sont remplacées).
- [ ] Le rapport HTML d'un benchmark où un fallback importer a été
      déclenché contient le `Fact` `IMPORTER_FALLBACK_TRIGGERED` dans la
      synthèse narrative.

**Risques et mitigation**

- *Risque* : la fonction `compute_word_diff` déplacée a des callers
  externes non documentés. *Mitigation* : laisser un ré-export dans
  `picarones/report/diff_utils.py` avec un `DeprecationWarning` au
  premier appel (suppression planifiée 2 versions plus tard).
- *Risque* : le test `test_circle_dependencies` refuse un import
  légitime (par exemple un test qui mocke). *Mitigation* : limiter le
  test à `picarones/`, pas à `tests/` ; ne scanner que les imports
  top-level, pas les imports paresseux dans des fonctions (qui sont
  acceptables pour break dependency cycles).

---

## 5. Phase 2 — Robustesse runtime (sem. 4–5)

> **Objectif** : durcir l'application web et l'orchestrateur de benchmark
> avant de pouvoir promettre une adoption institutionnelle. Une bibliothèque
> nationale ne déploie pas un service qui n'a ni protection CSRF, ni
> endpoint `/health`, ni test de concurrence.

### Sprint A4 — Sécurité web (3 PJ)

**Pourquoi à cette position dans la séquence**

A1 a posé les scanners (bandit + trivy) qui détecteront toute régression
sécurité. A3 a stabilisé l'architecture. Il est maintenant temps de
fermer les deux trous fonctionnels : **B-11** (pas de CSRF) et **M-3**
(le `HEALTHCHECK` Docker pointe vers un `/health` qui n'existe pas).
Ces deux items sont indépendants des autres phases — on les fait avant
A5 (concurrence) parce qu'A5 va ajouter des tests d'intégration qui ont
besoin du middleware CSRF stabilisé.

**Items de l'audit résolus**

**B-11** (CSRF), **M-3** (`/health` absent).

**Livrables concrets**

- `picarones/web/security.py` : ajout du middleware `csrf_middleware`
  basé sur `starlette-csrf` (ou implémentation maison ~80 L : token
  signé HMAC-SHA256 dans cookie `picarones_csrf` + en-tête
  `X-CSRF-Token` exigé sur POST/PUT/DELETE). Activé par variable
  `PICARONES_CSRF_REQUIRED=1`. Désactivé par défaut sur Space public
  (pas de session authentifiée à protéger), activé d'office en mode
  institutionnel.
- `picarones/web/routers/system.py` : ajouter `GET /health` qui retourne
  `{"status": "ok", "version": __version__}` en < 50 ms, sans toucher à
  la BD ni aux engines (vrai healthcheck Kubernetes-ready). Conserver
  `/api/status` qui reste plus riche (pour le frontend).
- `Dockerfile:96` : pointer le `HEALTHCHECK` vers `/health` au lieu de
  `/api/status`.
- `picarones/web/templates/_app.js` : pour chaque appel `fetch` POST,
  injecter automatiquement l'en-tête `X-CSRF-Token` lu depuis le cookie.
- `tests/web/test_csrf.py` (nouveau, ~12 cas) : POST sans token →
  403 ; POST avec token invalide → 403 ; POST avec token valide →
  200 ; en mode public (sans `PICARONES_CSRF_REQUIRED`) → POST passe
  sans token (rétrocompat HF Space).
- `tests/web/test_health.py` (nouveau, ~4 cas) : `GET /health` →
  200 + JSON valide ; latence < 100 ms ; ne déclenche pas de log SQL ;
  fonctionne même si la BD jobs est down.
- Documentation : `SECURITY.md` mise à jour avec un encart « Mode
  institutionnel : exporter `PICARONES_CSRF_REQUIRED=1` derrière votre
  reverse-proxy ».

**Critères d'acceptation**

- [ ] `pytest tests/web/` passe.
- [ ] `curl -X POST -H 'X-CSRF-Token: invalid' http://localhost:7860/api/config/save`
      retourne 403 quand `PICARONES_CSRF_REQUIRED=1`.
- [ ] `docker run … && sleep 35 && docker inspect <id> | grep -c '"Status": "healthy"'`
      retourne 1 (le HEALTHCHECK passe).
- [ ] `bandit -r picarones/web/` ne signale aucune nouvelle issue.

**Risques et mitigation**

- *Risque* : une intégration tierce (jq, script CI maison) qui appelle
  `/api/benchmark/start` sans token casse en mode institutionnel.
  *Mitigation* : documenter explicitement dans `SECURITY.md` la
  procédure « générer un token via `GET /api/csrf/token` puis le passer
  dans toutes les requêtes » + exemple curl.
- *Risque* : l'ajout du middleware ralentit les requêtes. *Mitigation* :
  benchmark p99 sur `/api/status` avant/après ; ajouter au job `tests`
  CI un check `< 200 ms p99 sur 100 requêtes`.

---

### Sprint A5 — Concurrence et performance (5 PJ)

**Pourquoi à cette position dans la séquence**

L'orchestrateur (`measurements/runner.py`, 1 019 lignes) gère
ProcessPool + ThreadPool, et la couche web utilise SSE +
SQLite WAL. L'audit §3 (M-13, M-14, M-16) signale qu'aucun test ne
couvre les cas concurrence sous charge, qu'il n'existe pas de
garde-fou anti-régression de performance, et que les rapports HTML
peuvent dépasser 200 MB sur de gros corpus. Tant que ces trois trous
ne sont pas fermés, la promesse « plateforme robuste » est
invérifiable. A5 vient après A4 parce que les tests de concurrence
appellent l'API web et ont besoin du middleware CSRF stable.

**Items de l'audit résolus**

**M-13** (tests concurrence runner + SSE), **M-14** (anti-régression
CER), **M-16** (lazy loading rapports), **m-10** (tests cloud OCR
sur erreurs HTTP 429/401/503).

**Livrables concrets**

- `tests/integration/test_runner_concurrency.py` (nouveau, ~50 cas) :
  - 32 jobs concurrents avec `PICARONES_MAX_CONCURRENT_JOBS=32` ;
  - épuisement du ProcessPool puis recovery ;
  - mort d'un worker au milieu d'un benchmark ;
  - timeout d'un doc dans un workers (le runner doit isoler) ;
  - écritures SSE concurrentes sur la même file ;
  - réception `Last-Event-ID` après reconnexion → replay correct.
- `tests/web/test_sqlite_concurrent_writes.py` (nouveau, ~10 cas) :
  10 threads écrivent simultanément dans `JobStore` → 0 corruption,
  pas de `SQLITE_BUSY` qui remonte au client.
- `tests/web/test_public_mode_hot_swap.py` (nouveau) : passage à chaud
  `PICARONES_PUBLIC_MODE=0 → 1` au milieu d'un benchmark ne casse pas
  les jobs en cours.
- `tests/engines/test_cloud_http_errors.py` (nouveau, ~12 cas par
  cloud) : Mistral OCR / Google Vision / Azure DI mockés pour
  retourner 429 (rate limit), 401 (clé invalide), 503 (indisponible),
  réponse vide. Vérifier le retry exponentiel + le warning explicite.
- `tests/fixtures/reference_corpus/` (nouveau) : 10 documents libres
  de droits couvrant 3 strates (médiéval, imprimé ancien, moderne) avec
  GT manuelle. Source : Gallica + Wikisource (à vérifier les licences
  doc par doc).
- `.github/workflows/perf_regression.yml` (nouveau) : workflow
  hebdomadaire (cron) qui lance `picarones run` sur le corpus de
  référence avec Tesseract + Pero, échoue si CER > 15 %. **Pas** à
  chaque PR (coût) mais audit hebdo + rapport en GitHub Issue auto.
- `picarones/report/generator.py` : nouveau paramètre
  `lazy_images: bool = False` (défaut conservateur). Si activé,
  externalise les images dans `report-assets/{doc_id}.png` à côté du
  HTML, avec `<img loading="lazy" src="report-assets/…" />`. Le HTML
  reste auto-portant si on copie aussi le dossier.
- `picarones/cli/__init__.py` : option `--lazy-images` sur `picarones
  report` qui propage le paramètre.
- Documentation dans `docs/user/reading-a-report.md` : encart « Pour
  les corpus > 50 documents, activer `--lazy-images` ».

**Critères d'acceptation**

- [ ] `pytest tests/integration/test_runner_concurrency.py` passe en
      < 3 min sur le runner CI.
- [ ] `pytest tests/web/test_sqlite_concurrent_writes.py` passe sans
      flakiness sur 10 runs successifs.
- [ ] Le job `perf_regression` hebdomadaire publie un commentaire
      automatique sur une GitHub Issue dédiée (`#perf-baseline`) avec
      le CER mesuré.
- [ ] Un rapport généré avec `--lazy-images` sur le corpus de référence
      pèse < 5 MB (vs ~50 MB sans).
- [ ] Aucun test n'a un timeout > 60 s individuel (limite douce
      pour parallélisation pytest-xdist plus tard).

**Risques et mitigation**

- *Risque* : le corpus de référence introduit un coût CI permanent.
  *Mitigation* : ne le lancer qu'en hebdo (cron), pas en PR. Le job
  est skippable manuellement via `[skip perf]` dans le commit message
  pour les release urgentes.
- *Risque* : `--lazy-images` casse l'auto-portance promise du rapport.
  *Mitigation* : documenter explicitement (encart en tête du rapport
  généré avec ce flag) et ajouter un sous-flag `--bundle-zip` qui
  produit un `.zip` contenant HTML + dossier d'images.

---

## 6. Phase 3 — Accessibilité (sem. 5–6, parallélisable avec Phase 4)

> **Objectif** : atteindre WCAG 2.1 niveau A bloquant (A6) puis niveau
> AA cible BnF (A7). Cette phase ne touche que les templates HTML/CSS/JS
> et les fichiers i18n — fichiers disjoints de Phase 4. Avec 2 ETP,
> A6+A7 et A8+A9 tournent en parallèle. Avec 1 ETP, A6 → A7 → A8 → A9.

### Sprint A6 — WCAG niveau A bloquant (3 PJ)

**Pourquoi à cette position dans la séquence**

A1 a posé les outils CI mais aucun n'audite l'accessibilité. A4 a
durci les endpoints. Il est maintenant temps de fermer les **deux
violations de niveau A** identifiées par l'audit §3 : graphiques
Chart.js Canvas inaccessibles aux lecteurs d'écran (B-9) et absence de
lien « Aller au contenu » (B-10). Sans ces deux corrections, **aucune
déclaration de conformité RGAA n'est légalement possible**. Le sprint
ouvre aussi la voie pour A11 (declaration d'accessibilité) qui ne peut
pas être rédigée sans audit niveau A passé.

**Items de l'audit résolus**

**B-9** (Canvas charts inaccessibles, ~12 graphiques Chart.js),
**B-10** (skip-to-content), **m-3** (bouton « Réinitialiser » sans clé
i18n), **m-4** (tableaux HTML sans `scope="col"` sur `<th>`).

**Livrables concrets**

- `picarones/report/templates/_app.js` : pour chaque instanciation
  Chart.js (`new Chart(canvas, …)` lignes 1062, 1102, et autres) :
  - ajouter `aria-label` descriptif sur le `<canvas>` ;
  - générer en parallèle un `<table>` masqué visuellement
    (`.visually-hidden`) avec les mêmes données, lié au canvas par
    `aria-describedby` ;
  - ajouter un bouton « Voir les données » qui révèle la table à tous
    (utile aussi pour la copie). Bouton masqué par défaut.
- `picarones/report/templates/_header.html` : en premier enfant du
  `<body>`, ajouter
  `<a href="#main" class="skip-link">{{ i18n.skip_to_content }}</a>`.
  Ajouter `id="main"` sur le `<main>` du `base.html.j2`.
- `picarones/report/templates/_styles.css` : classe `.skip-link` cachée
  hors `:focus` (`position:absolute; left:-9999px;` → revient à
  `top:0; left:0;` au focus, contraste AA).
- Tableaux dans `view_*.html` : ajouter `scope="col"` sur tous les
  `<th>` du tableau classement, du tableau NER, du tableau philologie,
  du tableau levers, etc. (~12 tables totales).
- Bouton « Réinitialiser » (`_header.html:25`) : remplacer le label
  hardcodé par `{{ i18n.reset_all }}`. Ajouter la clé dans
  `picarones/report/i18n/{fr,en}.json`.
- `picarones/report/i18n/fr.json` et `en.json` : nouvelle clé
  `skip_to_content` (« Aller au contenu » / « Skip to content ») et
  `reset_all` (« Réinitialiser » / « Reset all »).
- `tests/report/test_a11y_level_a.py` (nouveau, ~15 cas) :
  - chaque rapport HTML généré (demo + corpus de référence) contient
    `<a class="skip-link">` en premier enfant du `<body>` ;
  - chaque `<canvas>` a un `aria-label` non vide ;
  - chaque `<table>` a au moins un `<th scope="col">` ;
  - chaque `<canvas>` a un `<table>` jumelé via `aria-describedby` ;
  - aucune chaîne hardcodée FR/EN ne traîne dans `_header.html`.
- Optionnel mais recommandé : ajouter `axe-core` (CLI) en CI sur le
  rapport demo (`pytest tests/report/test_a11y_level_a.py` produit le
  HTML, `axe http://file://…` audite). Si `axe` complexe à déployer
  en CI, lancer manuellement à chaque release.

**Critères d'acceptation**

- [ ] `pytest tests/report/test_a11y_level_a.py` passe (15/15).
- [ ] Lecture du rapport demo par NVDA (test manuel, capté en
      vidéo dans `docs/audits/a11y-tests/A6.mp4`) : chaque graphique
      annonce son contenu via `aria-label` + table jumelle accessible.
- [ ] Tab depuis l'URL bar atteint le `<main>` en 1 tabulation
      (skip-link visible et fonctionnel).
- [ ] `axe-core` audit sur le rapport demo signale 0 violation niveau A.

**Risques et mitigation**

- *Risque* : la table jumelle pour chaque chart double le poids HTML.
  *Mitigation* : table générée à la demande via JS (`onclick="showTable(canvas_id)"`),
  pas dans le DOM initial. Pour les utilisateurs AT, table rendue via
  `aria-describedby` qui pointe vers une `<template>` JS-rendered.
- *Risque* : `axe-core` en CI introduit dépendance Node/Chromium.
  *Mitigation* : le test pytest natif est suffisant pour bloquer les
  régressions ; `axe-core` reste audit *manuel* à chaque release.

---

### Sprint A7 — WCAG AA + i18n résiduel + déclaration (3 PJ)

**Pourquoi à cette position dans la séquence**

A6 a fermé les violations de niveau A (donc *éligibilité* à une
conformité). A7 monte au niveau **AA** qui est le standard
institutionnel BnF / Service-Public.fr / European Accessibility Act,
et publie la déclaration d'accessibilité formelle. Sans A7,
l'institution ne peut pas afficher l'engagement légal de conformité
sur la home web. A7 vient juste après A6 parce que la déclaration
agrège les résultats des deux sprints.

**Items de l'audit résolus**

**m-1** (`_app.js:1087` chaîne FR hardcodée), **m-2** (`_app.js:1049`
chaîne FR hardcodée), **m-5** (palette heatmap non-daltonien-friendly),
**m-6** (nombres non localisés dans les tableaux), **M-9** (déclaration
d'accessibilité absente).

**Livrables concrets**

- `picarones/report/templates/_app.js:1087` : remplacer
  `'Données d'ancrage non disponibles.'` par `I18N.no_anchor_data`.
  Ligne 1049 : remplacer le fallback FR par `I18N.no_gini`.
- `picarones/report/i18n/{fr,en}.json` : ajout des clés
  `no_anchor_data`, `no_gini` (la deuxième existe peut-être déjà — à
  vérifier).
- `picarones/report/colors.py` : nouvelle palette daltonien-friendly
  (Okabe-Ito : `#0072B2` bleu / `#E69F00` orange / `#009E73` vert /
  `#CC79A7` rose / `#56B4E9` cyan). Conserver l'ancienne palette
  comme `colors_classic` pour rétrocompat. Variable `report_palette`
  dans `_styles.css` qui pointe vers la nouvelle par défaut, avec
  override possible via `?palette=classic` dans l'URL.
- Toggle dans le panneau « Avancé » du rapport :
  `<label><input type="checkbox" data-key="palette"> Mode daltonien-friendly</label>`
  qui bascule la palette via classe CSS sur `<body>`.
- `picarones/report/templates/_app.js` : utiliser
  `Number(value).toLocaleString(I18N.locale)` partout où un nombre
  s'affiche dans un tableau (chercher `${cer}`, `${pct}`, etc.).
- `picarones/report/i18n/{fr,en}.json` : ajouter le champ
  `locale: "fr-FR"` / `"en-GB"` (utilisé par `toLocaleString`).
- `ACCESSIBILITY.md` (nouveau, ~150 L à la racine) :
  - engagement de conformité WCAG 2.1 AA / RGAA 4.1 ;
  - méthode d'audit (interne A6+A7 + externe en A15) ;
  - dérogations connues (ex : matrice de confusion Unicode reste
    visuellement dense, contournement table jumelle décrit) ;
  - contact référent accessibilité ;
  - calendrier de réaudit (annuel).
- Lien `ACCESSIBILITY.md` ajouté en footer du rapport HTML et de la
  home web.
- `tests/report/test_a11y_level_aa.py` (nouveau, ~12 cas) :
  - palette daltonien-friendly active par défaut ;
  - contraste cellules tableau classement ≥ 4,5:1
    (pour normal text) sur les 4 tiers (excellent/acceptable/médiocre/
    critique) ;
  - nombres dans les tableaux respectent
    `toLocaleString(report_lang)` (test : `1234567` rendu en FR
    contient un espace fine ` ` ou un espace insécable ` `) ;
  - aucune chaîne FR n'apparaît dans le HTML rendu en mode EN.
- `tests/web/test_accessibility_link.py` : la home web et le footer
  du rapport HTML linkent vers `/accessibility` ou
  `ACCESSIBILITY.md`.

**Critères d'acceptation**

- [ ] `pytest tests/report/test_a11y_level_aa.py` passe.
- [ ] Le toggle « Mode daltonien-friendly » dans le panneau Avancé
      bascule la palette en < 100 ms et persiste en URL (`?palette=…`).
- [ ] Sur le rapport demo, `axe-core` signale 0 violation niveau AA
      (audit manuel).
- [ ] `ACCESSIBILITY.md` publié et linké depuis le footer.

**Risques et mitigation**

- *Risque* : la nouvelle palette Okabe-Ito n'est pas appréciée
  esthétiquement par l'équipe. *Mitigation* : laisser le toggle URL
  `?palette=classic` qui revient à l'ancienne palette ; l'équipe
  trouvera son équilibre par usage.
- *Risque* : la déclaration d'accessibilité engage légalement
  (loi 2005-102 art. 47 en France). *Mitigation* : audit externe en
  A15 *avant* publication officielle de la déclaration. La version
  rédigée en A7 reste en mode draft (`ACCESSIBILITY.md` avec encart
  « Audit externe en cours, version provisoire ») jusqu'à validation.

---

## 7. Phase 4 — Reproductibilité opérationnelle (sem. 5–7, parallélisable avec Phase 3)

> **Objectif** : passer d'une plateforme « ça compile chez moi » à une
> plateforme dont *tout* artefact est reproductible bit-à-bit ou à
> minima reconstructible à un commit/digest donné. Préalable absolu à
> la Phase 6 (publication scientifique) qui ne peut pas garantir la
> reproductibilité tant que les builds eux-mêmes ne le sont pas.

### Sprint A8 — Lock file + Docker digest + ops files (3 PJ)

**Pourquoi à cette position dans la séquence**

A1 a hardci la CI mais elle reste vulnérable à une dérive de
dépendances : aucun lock file, image Docker `python:3.11-slim` sans
digest, `requirements.txt` à la racine divergent. Toute promesse
de reproductibilité (Phase 6) tombe si on ne peut pas reconstruire
exactement l'environnement d'un benchmark. A8 vient *avant* A9
(distribution PyPI) parce qu'on ne publie pas un wheel sans avoir
verrouillé ses transitives.

**Items de l'audit résolus**

**M-1** (lock file), **M-2** (Docker digest), **M-12** (snapshots
reproductibilité sous-documentés), **M-18** (.dockerignore +
.env.example), **m-11** (versionnement testdata), **m-13**
(`requirements.txt` divergent), **m-14** (staleness pricing.yaml).

**Livrables concrets**

- Adoption de `uv` (plus rapide que pip-tools, projet Astral) pour la
  génération du lock :
  - `uv pip compile pyproject.toml --extra dev --extra web --extra stats
    --extra llm --extra ner --output-file requirements.lock`
  - `uv pip compile pyproject.toml --output-file requirements-runtime.lock`
    (cœur seulement, utilisé par Docker production).
- `Dockerfile` : remplacer `pip install .` par
  `uv pip sync --system requirements-runtime.lock && pip install .
  --no-deps`. Épingler `FROM python:3.11.10-slim@sha256:…` (digest
  obtenu via `docker pull python:3.11.10-slim && docker inspect`).
- `.dockerignore` (nouveau, ~30 L) : exclure `.git`, `tests/`, `docs/`,
  `*.pyc`, `__pycache__/`, `.venv/`, `.github/`, `examples/`,
  `*.spec`, `node_modules/`, `*.md` sauf `README.md` (utile pour le
  message de welcome).
- `.env.example` (nouveau, ~30 L) : toutes les variables d'env
  utilisées par `docker-compose.yml` et `picarones/web/security.py`
  (PICARONES_PUBLIC_MODE, PICARONES_RATE_LIMIT_PER_HOUR,
  PICARONES_MAX_UPLOAD_MB, PICARONES_BROWSE_ROOTS,
  PICARONES_MAX_CONCURRENT_JOBS, PICARONES_CSRF_REQUIRED,
  MISTRAL_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY,
  GOOGLE_APPLICATION_CREDENTIALS, AZURE_DOC_INTEL_*) avec une ligne
  de commentaire chacune.
- `requirements.txt` (existant) : devient un alias `-r requirements.lock`
  ou supprimé avec un message dans `INSTALL.md` redirigeant vers
  `pip install -e ".[dev,web]"`.
- `tests/.testdata/VERSION.yaml` (nouveau) : pour chaque corpus de test
  versionnable, une entrée `{name, sha256, source_url, commit_picarones,
  date_added}`. Permet de détecter une dérive accidentelle des fixtures.
- `picarones/data/pricing.yaml` : ajouter en tête `last_updated:` et
  `valid_until:` (par défaut +6 mois). Le générateur de rapport
  émet un `Fact` `PRICING_STALENESS_WARNING` (importance MEDIUM) si
  `today > valid_until`.
- `docs/reproducibility-snapshots.md` (nouveau, ~250 L) :
  - ce qu'un snapshot contient (déjà documenté en partie dans
    `picarones/report/snapshot.py` mais éparpillé) ;
  - comment recharger un snapshot pour rejouer un benchmark ;
  - comment versionner un snapshot dans un papier
    (commit picarones + digest Docker + lock file hash) ;
  - exemple complet bout-en-bout avec un mini corpus.
- `.github/workflows/lock_refresh.yml` (nouveau) : workflow mensuel
  (cron 1er du mois) qui régénère les locks et ouvre un PR
  automatique pour validation humaine.

**Critères d'acceptation**

- [ ] `pytest tests/` passe identiquement avec `pip install
      -r requirements.lock` que avec `pip install -e ".[dev,web]"`.
- [ ] `docker build .` fait sans network après le premier pull du
      base image (toutes les deps sont dans le cache).
- [ ] `docker images picarones --format "{{.Size}}"` < 1.5 GB
      (réduction grâce à `.dockerignore`).
- [ ] Un benchmark reproduit à 6 mois d'intervalle avec les mêmes
      lock file + digest Docker + commit picarones produit un rapport
      bit-à-bit identique (test bonus, lourd, fait manuellement avant
      release v1.1.0).
- [ ] Le rapport demo généré 2 jours après `valid_until` du pricing
      contient le `Fact` `PRICING_STALENESS_WARNING`.

**Risques et mitigation**

- *Risque* : `uv pip sync` introduit un comportement subtilement
  différent de `pip` sur des deps avec extras conditionnels.
  *Mitigation* : test parité installation (`pytest -k "test_install_parity"`)
  qui compare `pip freeze` avant/après.
- *Risque* : le PR mensuel auto de refresh des locks crée du bruit.
  *Mitigation* : labels GitHub `auto-refresh` + auto-merge si tous
  les checks passent (zone configurable selon politique de l'institution).

---

### Sprint A9 — Distribution PyPI + ghcr.io + releases (3 PJ)

**Pourquoi à cette position dans la séquence**

A8 a posé la base reproductible. A9 transforme cette base en
**artefacts publiables** : wheel sur PyPI, image conteneur sur ghcr.io,
release GitHub avec changelog auto. Sans A9, le projet reste invitable
en `pip install picarones` ; un papier JOSS qui pointe vers le
projet (Phase 6) doit pouvoir citer une version installable.

**Items de l'audit résolus**

**M-5** (PyPI release pipeline), **M-6** (image conteneur immutable
publiée), **m-15** (`picarones.spec` PyInstaller hidden_imports
manuels), **m-16** (extras placeholder `[historical]` `[importers]`).

**Livrables concrets**

- `pyproject.toml` : adopter `setuptools_scm` pour la version (via tag
  Git). Suppression de la version hardcodée `1.0.0`.
- `.github/workflows/release.yml` (nouveau) : déclenché sur tag `v*` :
  - build sdist + wheel via `python -m build`
  - test parité (`twine check`)
  - publication TestPyPI
  - smoke test (`pip install --index-url testpypi picarones==<version> &&
    picarones --version`)
  - publication PyPI via `pypa/gh-action-pypi-publish` avec OIDC
    trust (pas de token long-lived)
  - build image multi-arch (amd64 + arm64 pour Mac M-series) via
    `docker buildx`
  - push `ghcr.io/maribakulj/picarones:<version>` + `:latest`
  - création GitHub Release avec corps généré depuis le CHANGELOG
    (parser Keep-a-Changelog)
- `picarones.spec` (PyInstaller) : remplacer la liste
  `hiddenimports` manuelle par
  `from PyInstaller.utils.hooks import collect_all` et un
  parcours auto. Tester via un nouveau job CI `build-exe` (non bloquant
  jusqu'à v1.1.0, bloquant ensuite).
- `pyproject.toml` extras : retirer les placeholders
  `historical = []` et `importers = []` (les modules sont *dans* le
  package principal — l'extra n'apporte rien). Documenter dans
  `CHANGELOG.md` que la séparation en packages PyPI séparés
  (`picarones-historical`, `picarones-importers`) est une décision
  architecturale future, pas un placeholder vide aujourd'hui.
- `tests/release/test_pypi_install.py` (nouveau) : sur un job CI
  `release-smoke`, lance dans un container vide
  `pip install picarones && picarones demo --output /tmp/demo.html`
  et vérifie que le HTML est produit.
- `docs/operations/release-process.md` (nouveau, ~80 L) : procédure
  release manuelle (tag → workflow → vérifs → annonce).

**Critères d'acceptation**

- [ ] `pip install picarones==1.1.0-rc1` depuis TestPyPI fonctionne sur
      Linux + macOS + Windows.
- [ ] `docker pull ghcr.io/maribakulj/picarones:1.1.0-rc1` retourne une
      image fonctionnelle qui démarre en < 30 s.
- [ ] La release GitHub `v1.1.0-rc1` est créée automatiquement, son
      corps reflète la section correspondante du CHANGELOG.
- [ ] `picarones.spec` build sans erreur et l'exécutable produit lance
      `picarones demo --output /tmp/demo.html` correctement.

**Risques et mitigation**

- *Risque* : conflit de nom sur PyPI (un autre projet `picarones`
  existerait). *Mitigation* : vérifier en début de sprint via
  `pip search` (déprécié) → `https://pypi.org/project/picarones/`.
  Si conflit, négocier avec le maintainer ou renommer en
  `picarones-bench`.
- *Risque* : OIDC trust setup demande des permissions GitHub Actions
  spécifiques. *Mitigation* : documenter dans `release-process.md` la
  procédure de setup PyPI Trusted Publisher + capture d'écran.

---

## 8. Phase 5 — Gouvernance institutionnelle (sem. 7–8)

> **Objectif** : passer du « projet maintenu par une personne » à un
> **artefact institutionnel** avec politiques publiques (gouvernance,
> conflit d'intérêt), documentation opérationnelle pour DSI BnF
> (déploiement intranet, RGPD, traduction des guides clés), et
> CODEOWNERS qui désigne les responsabilités de revue.

### Sprint A10 — Politiques de gouvernance (2 PJ)

**Pourquoi à cette position dans la séquence**

A8 et A9 ont rendu le projet *distribuable*. Avant qu'une institution
ne l'évalue pour adoption, elle doit pouvoir lire la politique de
maintenance, la divulgation de conflit d'intérêt (le projet benchmarke
des APIs cloud payantes — biais éditorial possible) et l'identification
des mainteneurs. C'est un sprint court mais **bloquant** pour toute
relation avec un service achat / juridique d'institution publique.

**Items de l'audit résolus**

**M-10** (divulgation de conflits d'intérêt), **M-11** (CODEOWNERS,
politique de maintenance, GOVERNANCE).

**Livrables concrets**

- `.github/CODEOWNERS` (nouveau) : pour chaque sous-package, désigner
  le mainteneur de revue. Exemple :
  ```
  /picarones/core/                @maribakulj
  /picarones/measurements/        @maribakulj
  /picarones/engines/             @maribakulj
  /picarones/web/                 @maribakulj
  /picarones/report/              @maribakulj
  /docs/                          @maribakulj
  /docs/case-studies/             @maribakulj  # rotater quand contributeurs domain experts
  ```
  À l'arrivée de contributeurs domain experts (paléographe, archiviste),
  les mettre dans le CODEOWNERS pour les fichiers qui les concernent
  (cas d'études, prompts, glossaire).
- `GOVERNANCE.md` (nouveau, ~120 L à la racine) :
  - rôles : maintenance team, contributeurs occasionnels, reviewers ;
  - cadence release : versions mineures mensuelles ; versions majeures
    trimestrielles ; patches sécurité 72 h ;
  - SLO réponse aux issues : 5 jours ouvrés pour triage initial ;
  - politique de breaking changes : interdits sans tag `v2.0.0` ;
  - procédure de transfert de mainteneur (BDFL → equipe → fondation
    en cas de croissance).
- `CODE_OF_CONDUCT.md` (nouveau si absent ; ~40 L) : adopter
  Contributor Covenant 2.1.
- `README.md` ou `GOVERNANCE.md` : section « Conflicts of interest » :
  - les mainteneurs déclarent leurs affiliations académiques /
    industrielles ;
  - les fournisseurs cloud benchmarkés (OpenAI, Anthropic, Mistral,
    Google, Azure) n'ont aucun lien capitalistique avec le projet
    (à vérifier puis affirmer) ;
  - le `pricing.yaml` reflète les tarifs publics observés à la date
    `last_updated`, sans accord commercial avec les providers.
- `paper.md` (draft JOSS, créé en A12) : reproduira la section COI.
- `tests/docs/test_governance_files_present.py` (nouveau) : vérifie
  que `CODEOWNERS`, `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `LICENSE` existent et ne sont pas vides. Garde-fou
  contre suppression accidentelle.

**Critères d'acceptation**

- [ ] `pytest tests/docs/test_governance_files_present.py` passe.
- [ ] `gh repo view --json codeOfConduct,licenseInfo` retourne les
      bons noms.
- [ ] La page « About » du repo GitHub est complète (description,
      website, topics).
- [ ] Au moins un PR test passe par le mécanisme de review CODEOWNERS
      (assigned reviewers correspondant au path).

**Risques et mitigation**

- *Risque* : le mainteneur unique ne veut pas s'engager sur des SLO
  publics. *Mitigation* : formuler les SLO en mode « best effort
  current » avec date de revue annuelle, pas en engagement contractuel.

---

### Sprint A11 — Documentation institutionnelle (5 PJ)

**Pourquoi à cette position dans la séquence**

A10 a publié les politiques *publiques*. A11 produit la documentation
*opérationnelle* qu'un DSI BnF lit en deuxième : guide de déploiement
intranet (au-delà du seul Space HuggingFace), politique RGPD/rétention,
déclaration d'accessibilité finalisée (issue de A6+A7), traduction
anglaise des guides utilisateur et développeur prioritaires. Sans
A11, l'institution ne peut pas mettre Picarones en production sur
ses propres infrastructures.

**Items de l'audit résolus**

**M-7** (guide déploiement institutionnel), **M-8** (politique RGPD /
rétention des données), **M-17** (traduction EN documentation
prioritaire). Validation finale de **M-9** initiée en A7
(`ACCESSIBILITY.md` passe en mode « audit interne validé, audit externe
en cours »).

**Livrables concrets**

- `docs/operations/deployment-institutional.md` (nouveau, ~250 L) :
  - pré-requis (Python 3.11+, Tesseract, optionnel : SSO Shibboleth /
    CAS / OIDC, BD partagée optionnelle Postgres en remplacement de
    SQLite, reverse-proxy Nginx/Apache) ;
  - architecture cible (mono-instance simple ; multi-instance derrière
    load balancer + BD centralisée) ;
  - intégration SSO via en-tête trusté `X-Remote-User` (déjà géré par
    la plupart des proxies institutionnels) ;
  - configuration des `PICARONES_*` variables d'env pour mode
    institutionnel (CSRF activé, browse roots restreints, rate limit
    aligné sur la politique interne) ;
  - intégration observabilité : format de log JSON pour ELK/Loki,
    métriques Prometheus exposées sur `/metrics` (à implémenter — voir
    note Risque) ;
  - sauvegarde / restauration de l'historique SQLite et des rapports.
- `docs/operations/data-retention-rgpd.md` (nouveau, ~150 L) :
  - quelles données Picarones collecte (uploads, IPs dans le
    rate-limiter, historique benchmarks) ;
  - durées de rétention par défaut et configurables :
    - uploads : 7 jours après dernier accès, purge auto via cron ;
    - logs IP : 24 h ;
    - historique benchmarks : indéfini par défaut, purge sur demande ;
  - procédure d'export / suppression des données d'un usager (droit
    à l'oubli) ;
  - mention RGPD dans le footer web (lien vers cette page).
- `picarones/web/maintenance.py` (nouveau, ~80 L) : tâche de purge
  schedulée (`asyncio.create_task` au démarrage de l'app) qui scanne
  `uploads/` et supprime ce qui dépasse `PICARONES_UPLOAD_RETENTION_DAYS`
  (défaut 7).
- `tests/web/test_upload_retention.py` (nouveau, ~5 cas) : un upload
  daté de 8 jours est supprimé ; un upload daté de 6 jours est
  conservé ; la purge n'efface pas les rapports générés.
- `ACCESSIBILITY.md` : passer en mode « audit interne validé », mettre
  à jour la date de réaudit pour A15.
- Traduction prioritaire en anglais :
  - `docs/user/reading-a-report.md` → `docs/user/reading-a-report.en.md`
  - `docs/developer/index.md` → `docs/developer/index.en.md`
  - `docs/developer/narrative-engine.md` → `.en.md`
  - `docs/developer/extending-glossary.md` → `.en.md`
  - `docs/developer/extending-i18n.md` → `.en.md`
  - `CONTRIBUTING.md` → `CONTRIBUTING.en.md`
  - chaque fichier en français reçoit un en-tête
    `> 🇬🇧 [English version](file.en.md)` et réciproquement.
- `tests/docs/test_translation_parity.py` (nouveau) : vérifie que
  chaque `*.md` a son `.en.md` équivalent et que les sections de
  premier niveau (`##`) correspondent (titres traduits ou
  identiques).
- CHANGELOG.md (cible non-rétroactive) : adopter à partir de v1.2.0
  un format bilingue (sections « Ajouté / Added », « Modifié /
  Changed », « Corrigé / Fixed »).

**Critères d'acceptation**

- [ ] `pytest tests/docs/test_translation_parity.py` passe (5 fichiers
      traduits, 5 paires).
- [ ] Un test manuel de la purge auto sur un upload daté de J-8
      confirme la suppression effective et un log
      `[maintenance] purged upload <id>`.
- [ ] La home web affiche en footer les liens vers `RGPD`,
      `ACCESSIBILITY`, `GOVERNANCE`.
- [ ] `docs/operations/deployment-institutional.md` est revu par au
      moins un DSI partenaire (BnF, BL, KBR ou autre — sollicitation
      à externaliser).

**Risques et mitigation**

- *Risque* : l'intégration Prometheus / observabilité dépasse le scope
  prévu. *Mitigation* : au pire, A11 ne livre que la *documentation*
  de l'intégration (`docs/operations/observability.md`) et l'instrumentation
  effective passe en backlog sprint A16+.
- *Risque* : la traduction EN demande une compétence linguistique pas
  toujours interne. *Mitigation* : version EN passée par DeepL
  + relecture humaine, marquée `<!-- translation: machine + human review -->`
  jusqu'à validation par un anglophone natif.

---

## 9. Phase 6 — Publication scientifique (sem. 8–10)

> **Objectif** : rendre le projet **citable**. Sans CITATION.cff, sans
> DOI Zenodo, sans papier JOSS, et sans citation primaire des méthodes
> statistiques dans le code, Picarones reste un dépôt GitHub mutable
> qu'aucune thèse, aucun papier, aucune institution sérieuse ne peut
> citer comme référence stable.

### Sprint A12 — CITATION + traçabilité méthodes + draft JOSS (5 PJ + cycle externe)

**Pourquoi à cette position dans la séquence**

C'est le sprint qui exige le plus de stabilisation préalable :
- A3 (architecture propre) avant que SPECS et le papier ne décrivent
  les 3 cercles ;
- A8 (lock file) avant que le papier ne promette la reproductibilité ;
- A9 (PyPI release) avant que le papier ne pointe vers `pip install
  picarones` ;
- A10 (COI) avant que le papier ne déclare l'absence de conflit ;
- A11 (a11y validée, RGPD documenté) avant que la soumission ne
  passe les comités d'éthique le cas échéant.

A12 démarre dès que A9 ferme. Le sprint produit la doc et le draft ;
le **cycle de revue par les pairs JOSS (8–12 semaines)** tourne en
parallèle de la suite (Phase 7).

**Items de l'audit résolus**

**B-4** (CITATION.cff + Zenodo + draft JOSS), **B-5** (références
primaires des méthodes statistiques dans le code), **B-6** (traçabilité
des profils de normalisation aux standards éditoriaux MUFI / TEI / DEAF).

**Livrables concrets**

- `CITATION.cff` (nouveau, racine) :
  - format Citation File Format 1.2.0 (parsé automatiquement par
    GitHub) ;
  - champs : `authors` (avec ORCID), `title`, `version`, `date-released`,
    `doi: 10.5281/zenodo.<id>` (assigné par Zenodo après release), `url`,
    `keywords`, `license: Apache-2.0`, `repository-code`,
    `preferred-citation` pointant vers le papier JOSS quand publié.
- Configuration Zenodo : activer l'intégration GitHub-Zenodo dans les
  settings du repo. Une fois la release v1.1.0 publiée (sortie d'A9),
  Zenodo crée un DOI immutable. Mettre à jour `CITATION.cff`.
- `paper.md` (nouveau, racine, format JOSS, ~6–8 pages soit ~600 L) :
  - **Summary** (200 mots) : qu'est-ce que Picarones, à qui ça
    s'adresse, en quoi c'est différent d'ocrevalUAtion / dinglehopper ;
  - **Statement of need** (300 mots) : pourquoi un *banc d'essai* (pas
    un atelier de production) ; pourquoi les métriques philologiques
    et la neutralité éditoriale comptent pour les institutions
    patrimoniales ;
  - **Functionality** (1 page) : architecture en 3 cercles, registre
    typé de métriques (Sprint 34), interface `BaseModule` (Sprint 33),
    moteur narratif factuel anti-hallucination (Sprint 19), pipelines
    composables (Sprints 63–66) ;
  - **Quality control** (½ page) : 3 356 tests, ruff, scanners CI,
    snapshots reproductibles, conformité WCAG AA ;
  - **References** (BibTeX) : Demšar 2006, Wilcoxon 1945, Efron 1979,
    MUFI v4.0, TEI P5, HTR-United, etc.
- `paper.bib` (nouveau, racine) : entries BibTeX correspondantes.
- Soumission JOSS : fork JOSS reviews repo, pre-review issue,
  attente de l'attribution d'un éditeur. Cycle externe non bloquant
  pour les sprints suivants.
- `picarones/measurements/statistics.py` : ajouter en-tête de module
  avec les références primaires (BibTeX en commentaire) :
  ```python
  """Tests statistiques pour la comparaison de moteurs OCR.

  Méthodes implémentées et leurs références primaires :
  - Test de Wilcoxon signé : Wilcoxon, F. (1945). Individual comparisons
    by ranking methods. Biometrics Bulletin, 1(6), 80–83.
  - Test de Friedman + post-hoc Nemenyi : Demšar, J. (2006). Statistical
    comparisons of classifiers over multiple data sets. JMLR, 7, 1–30.
  - Bootstrap intervalles de confiance : Efron, B. (1979). Bootstrap
    methods. Annals of Statistics, 7(1), 1–26.
  - Critical Difference Diagram : Demšar 2006 (cf. ci-dessus).
  """
  ```
  + dans la docstring de chaque fonction publique (`compute_friedman`,
  `compute_nemenyi_posthoc`, `bootstrap_ci`, etc.), ajouter une ligne
  `:references: Demšar 2006 §3.2`.
- `picarones/measurements/normalization.py` : pour chaque profil
  (`DIPLOMATIC_FR`, `MUFI`, `EARLY_MODERN_*`, `MEDIEVAL_*`), ajouter
  en commentaire la spec source (URL stable, version, date d'extraction) :
  ```python
  MEDIEVAL_FRENCH = {
      # Source: TEI P5 §3.4 Unicode (https://tei-c.org/release/doc/...)
      # MUFI v4.0 (https://mufi.info/m.php?p=mufi/specifications)
      # DEAF normalization conventions (Möhren 2017)
      # Date d'extraction: 2026-05-02
      "ſ": "s",  # long s
      ...
  }
  ```
- `docs/normalization-specs.md` (nouveau, ~200 L) : tableau exhaustif
  profil × spec source × date d'extraction × DOI/URL. Liens vers les
  documents source. Politique de mise à jour (à chaque révision MUFI
  ou TEI, ouvrir un PR avec la diff).
- `tests/measurements/test_normalization_traceability.py` (nouveau) :
  pour chaque profil, vérifier qu'il existe une entrée dans
  `docs/normalization-specs.md` (parsing simple).
- `picarones/report/glossary/{fr,en}.yaml` : pour les 25 entrées,
  vérifier que le champ `reference` est rempli avec une citation
  primaire (audit ligne par ligne ; corriger les manquants).

**Critères d'acceptation**

- [ ] `CITATION.cff` est valide selon `cffconvert --validate`.
- [ ] Le bouton « Cite this repository » apparaît sur la page GitHub
      du repo et produit BibTeX + APA correct.
- [ ] DOI Zenodo `10.5281/zenodo.<id>` est attribué et le badge
      apparaît dans le README (mise à jour finale en A13).
- [ ] `paper.md` est validé localement par `whedon` (validateur JOSS,
      `whedon prepare picarones-paper`).
- [ ] `tests/measurements/test_normalization_traceability.py` passe.
- [ ] Une recherche `grep -rn "Demšar 2006" picarones/measurements/` retourne
      au moins 3 occurrences.

**Risques et mitigation**

- *Risque* : revue JOSS demande des changements méthodologiques de fond
  (par ex. ajouter une métrique ou changer une interprétation).
  *Mitigation* : prévoir un sprint A12-bis dédié à ces retours, en
  Phase 8 ; ne pas bloquer A13/A14 sur ce cycle externe.
- *Risque* : Zenodo intégration ne fonctionne qu'avec une release
  GitHub *publique*. *Mitigation* : confirmer que le repo est public
  ou rendre public en début de sprint (cohérent avec Apache-2.0
  affiché).
- *Risque* : MUFI v4.1 sort pendant la rédaction → références
  obsolètes en publication. *Mitigation* : `docs/normalization-specs.md`
  date l'extraction explicitement ; la version implémentée reste
  v4.0 avec tracking ouvert pour v4.1 en backlog.

---

## 10. Phase 7 — Refonte documentation produit (sem. 10–11)

> **Objectif** : maintenant que le code, la CI, l'a11y, l'ops, la
> gouvernance et la communication scientifique sont stabilisées, on
> refait README et SPECS *en dernier* — précisément parce qu'ils
> doivent refléter un état figé. Refaire ces documents avant aurait
> impliqué de les refaire deux fois.

### Sprint A13 — Refonte README (4 PJ)

**Pourquoi à cette position dans la séquence**

Le README est la première impression. L'audit §9.2 a identifié 13
items : markdown cassé, compteurs faux, roadmap arrêtée 75 sprints en
arrière, project structure pré-Sprint 32-34, moteurs annoncés sans
adapter, CLI/API/métriques sous-documentées de moitié à deux tiers.
A13 vient en avant-dernier parce que le contenu à publier dépend de :
- A1+A2 (gates anti-régression activés),
- A3 (architecture stabilisée),
- A6+A7 (a11y validée → on peut afficher le badge AA),
- A8+A9 (`pip install picarones` fonctionne, image ghcr.io disponible),
- A12 (CITATION.cff présent → bouton « Cite this repository » à
  promouvoir dans le README).

**Items de l'audit résolus**

**B-13** (markdown des taglines cassé), **M-19** (compteur de tests),
**M-20** (Roadmap Sprint 22 → 97), **M-21** (Known Issues obsolète),
**M-22** (Project Structure pré-refactor), **M-23** (Kraken /
custom YAML annoncés sans implémentation), **M-24** (variables
`AWS_*` sans adapter), **M-25** (CLI 6/15 documentée), **M-26** (API
web 10/27), **M-27** (métriques 8/28), **M-28** (sections rapport
15/25), **m-18** (copyright, lien SPECS, prompts latin),
**§9.3** (alignement compteurs entre 3 docs).

**Livrables concrets**

- `README.md` réécrit intégralement à partir du squelette de l'existant,
  en respectant les sections suivantes (et seulement celles-ci, pour
  ne pas reproduire l'enflure originelle) :
  1. **Header** (titre, taglines bilingues fermées correctement, badges
     CI / Python / License / DOI Zenodo / PyPI / HF Space).
  2. **What is Picarones** (paragraphe de 100 mots, FR puis EN — pas
     de duplication de section).
  3. **Use case** (paragraphe d'archive/library, 80 mots).
  4. **Quick start** (3 commandes : `pip install picarones`,
     `picarones demo`, `picarones serve`).
  5. **Installation** (renvoie à `INSTALL.md` pour le détail).
  6. **Documentation** (table des renvois vers `docs/user/`,
     `docs/developer/`, `docs/operations/`).
  7. **Citation** (BibTeX généré depuis CITATION.cff + DOI Zenodo).
  8. **License** (Apache 2.0).
- **Tableau « Supported engines »** auto-généré par un script
  `scripts/gen_readme_tables.py` (nouveau) qui lit
  `picarones/engines/__init__.py` et produit la liste à partir du
  registre. Idem pour la liste des commandes CLI (lit
  `picarones --help`) et la liste des endpoints (lit
  `app.openapi()`). Ces tableaux sont insérés dans le README via des
  balises HTML invisibles `<!-- generated:engines -->` /
  `<!-- /generated:engines -->`. Régénération en CI à chaque PR via
  un job qui échoue si le contenu généré diffère du contenu commité.
- **Section « Heritage-specific metrics »** : 3 sous-sections
  (« Métriques classiques OCR/HTR », « Métriques philologiques »,
  « Métriques de comparaison et décision »), chacune avec 3 à 5
  bullets et un lien vers `docs/views.md` pour le détail. Plus de
  liste linéaire de 28 items.
- **Section « Interactive HTML Report »** : screenshot du rapport
  demo (image dans `docs/assets/report-screenshot.png`, taille < 200 KB)
  + liste structurée des 25 sections du rapport regroupées en 5
  familles (synthèse, classement, vues thématiques, analyses
  statistiques, panneaux interactifs).
- **Roadmap** : tableau condensé des sprints **par phase**, pas
  individuellement. Trois colonnes : phase / focus / état. Détail
  technique renvoyé vers `CHANGELOG.md` et `docs/roadmap/`.
- **Known Issues** : suppression intégrale. Cette section devient
  caduque (le présent plan de remédiation et l'audit la remplacent).
  Note de redirection : « Voir
  [`docs/audits/`](docs/audits/) pour les audits en cours ».
- **Project Structure** : régénéré à partir du repo réel via un
  script `scripts/gen_project_structure.py` (nouveau). Insertion via
  `<!-- generated:structure -->`.
- Footer : `Copyright 2024–2026 Picarones contributors`. Lien
  RGPD / Accessibility / Governance / Contributing.
- `tests/docs/test_readme_consistency.py` (déjà créé en A2) doit
  passer **strictement** sur le nouveau README — tous les moteurs,
  toutes les commandes, tous les endpoints, tous les compteurs sont
  vérifiés.
- `tests/docs/test_readme_dual_lang.py` (nouveau) : la version FR et
  la version EN ont des sections de premier niveau qui se
  correspondent.

**Critères d'acceptation**

- [ ] `pytest tests/docs/test_readme_consistency.py` passe (0
      divergence entre tableau README et code).
- [ ] Les badges CI / Codecov / PyPI / DOI / HF Space sont tous
      verts au moment de la PR de refonte.
- [ ] Le nouveau README compte < 400 lignes (vs 786 actuelles, en
      grande partie déléguées à `docs/`).
- [ ] Le markdown des taglines est correct, validé par
      `markdown-link-check` ou équivalent.
- [ ] Un test manuel de rendu sur GitHub ET sur HuggingFace Space
      affiche un README propre, sans `> **` jamais fermé.

**Risques et mitigation**

- *Risque* : la génération auto des tableaux casse à un PR futur si
  un moteur n'a pas la bonne shape. *Mitigation* : le script de
  génération échoue lui-même en CI avant l'étape de comparaison,
  avec un message explicite (« Engine `xyz` manque le champ
  `display_name` »).
- *Risque* : le screenshot du rapport demo prend de la place dans le
  repo. *Mitigation* : 200 KB plafonné par pre-commit (déjà actif
  via `check-added-large-files --maxkb=500`). Image optimisée
  via `pngquant`.

---

### Sprint A14 — Refonte SPECS.md (3 PJ)

**Pourquoi à cette position dans la séquence**

SPECS.md (Mars 2025, addendum Sprints 16-30) est désynchronisé d'~75
sprints. L'audit §9.1 identifie 9 promesses non tenues sans deprecation
et ~25 modules majeurs ajoutés invisibles dans SPECS. A14 vient *après*
A13 parce que :
- le README est la première lecture et doit être impeccable avant
  qu'on touche au document plus technique ;
- A13 a posé le pattern de génération auto des tableaux que SPECS
  va réutiliser pour ses listes de moteurs / métriques ;
- A12 a publié les références primaires des méthodes statistiques —
  SPECS peut maintenant les citer correctement.

**Items de l'audit résolus**

**B-12** (SPECS à refondre intégralement, 9 promesses non tenues +
25 modules ajoutés non documentés).

**Livrables concrets**

- `SPECS.md` réécrit intégralement, version 2.0 datée mai 2026,
  reflétant strictement le code réel, structuré comme suit :
  1. **Vision et positionnement** : philosophie banc d'essai
     (pas atelier), neutralité éditoriale, contribution scientifique
     du projet (registres typés, narrative anti-hallucination,
     interface BaseModule).
  2. **Architecture** : diagramme des 3 cercles à jour
     (Cercle 1 = 7 modules, Cercle 2 = ~70, Cercle 3 = ~50), règle
     de dépendance, registre typé Sprint 34, interface BaseModule
     Sprint 33, GT multi-niveaux Sprint 32.
  3. **Modules fonctionnels** : 6 sous-sections (Corpus, Adaptateurs
     OCR, Pipelines composables, Métriques, Rapport, Interface).
     Chacune décrit ce qui existe *aujourd'hui*, sans projection.
  4. **Métriques** : tableau exhaustif des 28+ métriques avec leur
     statut (« stable »/« expérimental »), spécification exacte,
     citation primaire, jonction de type (TEXT, ALTO, etc.), limites
     connues.
  5. **Modes pipeline** : `zero_shot`, `post_correction_texte`,
     `post_correction_image_texte`, `pipeline_composable_yaml`
     (Sprint 70).
  6. **Sécurité institutionnelle** : récap des garde-fous
     (PICARONES_PUBLIC_MODE, CSRF, zip-slip, defusedxml, rate limit,
     validation Pillow).
  7. **Reproductibilité** : snapshots, lock file, digest Docker.
  8. **Limites assumées et non-fonctionnalités** : liste explicite
     de ce que Picarones **ne fait pas et ne fera pas** dans la
     v1.x — par exemple :
     - pas de recommandation prescriptive (pivot philosophique vs
       SPECS v1) ;
     - pas d'export PDF (CSV + JSON suffisent ; export PDF reporté
       car coût de maintenance disproportionné) ;
     - pas d'adapter Kraken / AWS Textract / Calamari / OCRopus4
       intégré (raisons : maintenance par adapter ~50 PJ ; ouverture
       en plugins externes prévue Sprint 97+) ;
     - pas de moteur custom YAML (refondu en pipelines composables
       Sprint 63) ;
     - pas de k-means clustering automatique des erreurs (taxonomie
       discrète + co-occurrence Jaccard couvrent l'usage) ;
     - pas de dataset curé livré avec le projet (philosophie « banc
       d'essai sur votre golden dataset »).
  9. **Roadmap d'évolution 2026** : pointe vers
     `docs/roadmap/evolution-2026.md`.
- Tableau de migration v1 → v2 SPECS (annexe) : pour chaque promesse
  v1 non tenue, ligne « v1 disait X / v2 documente Y / raison Z ».
  Permet à un lecteur qui avait lu v1 de comprendre ce qui a changé.
- `tests/docs/test_specs_consistency.py` (déjà créé en A2) doit
  passer.
- Lien dans le README (A13) : « Pour la spécification fonctionnelle
  et technique complète, voir [SPECS.md](SPECS.md) ».

**Critères d'acceptation**

- [ ] `pytest tests/docs/test_specs_consistency.py` passe.
- [ ] Aucune section de SPECS ne décrit une fonctionnalité absente
      du code.
- [ ] Toute fonctionnalité majeure citée dans le CHANGELOG depuis
      Sprint 30 est mentionnée dans SPECS v2.
- [ ] La section « Limites assumées » est non vide et liste les 9+
      promesses v1 explicitement abandonnées avec leur raison.

**Risques et mitigation**

- *Risque* : la section « Limites assumées » est lue comme une
  régression par un primo-lecteur. *Mitigation* : reformuler
  positivement (« choix éditoriaux du projet ») et expliquer le
  pivot vers la philosophie banc d'essai en intro.
- *Risque* : SPECS.md devient un duplicata de CLAUDE.md. *Mitigation* :
  garder SPECS *fonctionnel et tourné public* (vocabulaire
  bibliothécaire, exemples patrimoniaux), CLAUDE.md *technique et
  tourné contributeur*. Les deux docs ne se chevauchent pas.

---

## 11. Phase 8 — Validation externe (sem. 11–12 + calendrier externe)

> **Objectif** : valider l'ensemble par des tiers indépendants. C'est
> ce qui transforme l'auto-déclaration en certification.

### Sprint A15 — Audits externes (1 PJ interne + cycle externe)

**Pourquoi à cette position dans la séquence**

Le travail interne est terminé. A15 est volontairement court côté
équipe (1 PJ) parce que la valeur vient des **prestataires externes** :
- audit RGAA externe (cabinet d'a11y, ~3 semaines après contractualisation) ;
- audit sécurité externe (pentest léger, ~2 semaines) ;
- finalisation de la revue JOSS (commencée en A12, ~12 semaines
  cumulées au total).

A15 est purement coordination + intégration des retours.

**Items de l'audit résolus**

Validation finale de **B-9, B-10, M-9** (audit RGAA externe),
**B-7, B-11** (audit sécurité externe), **B-4, B-5** (acceptation JOSS).

**Livrables concrets**

- Sélection et contractualisation d'un cabinet RGAA (par ex.
  Access42, Atalan, Tanaguru en France ; Tetralogical au UK ; ou
  service interne BnF si disponible). Cahier des charges :
  audit WCAG 2.1 AA sur le rapport HTML demo + l'interface web.
  Livrable attendu : rapport d'audit + déclaration de conformité
  signée.
- Sélection et contractualisation d'un audit sécurité (par ex. Synacktiv,
  Ambionics, ou équivalent BL-side). Cahier des charges :
  pentest application web (focus authentification/CSRF, file upload,
  injection, RCE), revue de la chaîne de build (CI scanners + image
  Docker).
- `ACCESSIBILITY.md` : intégrer les retours RGAA, supprimer la
  mention « audit externe en cours ».
- `SECURITY.md` : intégrer les retours pentest, ajouter la date de
  prochain réaudit (annuel).
- Réponse aux reviewers JOSS : un PR par retour majeur, intégré au
  paper. Tour de rév normalement court car le draft a été préparé
  rigoureusement en A12.
- `docs/audits/external-audits-2026/` (nouveau dossier) :
  - `rgaa-audit-2026-MM.pdf` (rapport scanné/PDF du cabinet) ;
  - `pentest-2026-MM.md` (résumé public, le rapport complet reste
    confidentiel) ;
  - `joss-review-correspondence.md` (résumé public des échanges).
- Annonce publique sur la home web et le README v2.1 : badges
  « WCAG 2.1 AA conformité totale » et « JOSS published » avec
  liens.

**Critères d'acceptation**

- [ ] Audit RGAA externe rapporte ≥ 95 % de critères AA conformes
      (le 5 % résiduel listé en dérogations dans `ACCESSIBILITY.md`).
- [ ] Audit sécurité externe ne signale aucune vulnérabilité de
      sévérité HIGH / CRITICAL.
- [ ] Le papier JOSS est accepté (`accepted by JOSS` issue closed).
- [ ] Le DOI JOSS est ajouté au CITATION.cff comme
      `preferred-citation`.

**Risques et mitigation**

- *Risque* : audit RGAA trouve un bloqueur niveau A imprévu (par ex.
  une matrice de confusion Unicode jugée non navigable au clavier).
  *Mitigation* : sprint A15-bis dédié à la remédiation, planifié en
  buffer fin de Phase 8.
- *Risque* : reviewers JOSS demandent une fonctionnalité scientifique
  manquante (par ex. métrique nouvelle). *Mitigation* : sprint
  A12-bis (cf. A12 risques), buffer Phase 8.
- *Risque* : prestataire externe indisponible. *Mitigation* :
  démarcher 2 candidats par audit et choisir le premier qui répond
  dans le délai.

---

## 12. Matrice de couverture

Chacun des **59 items** identifiés par l'audit est mappé à un sprint.
Cette matrice est la garantie d'exhaustivité : si un item n'apparaît
pas ici, le plan a un trou.

### Bloqueurs (13/13 couverts)

| ID | Item | Sprint | Livrable spécifique |
|---|---|---|---|
| B-1 | Violation Cercle 2→3 (`statistics.py:861`) | A3 | `core/diff_utils.py` créé, ré-export rétrocompat |
| B-2 | Violation Cercle 2→3 (`difficulty.py:195`) | A3 | `report/difficulty_render.py` créé |
| B-3 | 3 `except Exception: pass` importers | A3 | `logger.warning` + Fact `IMPORTER_FALLBACK_TRIGGERED` |
| B-4 | Pas de CITATION.cff / JOSS | A12 | `CITATION.cff` + Zenodo DOI + `paper.md` |
| B-5 | Méthodes statistiques non citées | A12 | En-tête module `statistics.py` + `:references:` par fonction |
| B-6 | Profils normalisation non tracés | A12 | `docs/normalization-specs.md` + commentaires inline |
| B-7 | Aucun scanner sécurité CI | A1 | Job `security` (bandit + pip-audit + trivy) |
| B-8 | Pas de `--cov-fail-under` | A1 | `--cov-fail-under=85` dans `ci.yml` |
| B-9 | Canvas Chart.js inaccessibles | A6 | `aria-label` + `<table>` jumelle + bouton |
| B-10 | Pas de skip-to-content | A6 | `<a class="skip-link">` dans `_header.html` |
| B-11 | Pas de CSRF | A4 | Middleware `csrf_middleware` + tests |
| B-12 | SPECS désynchronisé | A14 | Refonte intégrale v2.0 |
| B-13 | Markdown taglines README cassé | A13 | Refonte README intégrale |

### Majors (28/28 couverts)

| ID | Item | Sprint | Livrable spécifique |
|---|---|---|---|
| M-1 | Lock file absent | A8 | `requirements.lock` + `requirements-runtime.lock` via `uv` |
| M-2 | Image Docker non épinglée | A8 | `FROM python:3.11.10-slim@sha256:…` |
| M-3 | `/health` absent | A4 | `GET /health` dans `system.py` |
| M-4 | Pas de mypy en CI | A1 | Job `typecheck` (`core/` strict, ailleurs lax) |
| M-5 | Pas de release PyPI | A9 | `release.yml` + setuptools_scm + OIDC |
| M-6 | Image conteneur non publiée | A9 | Push ghcr.io multi-arch dans `release.yml` |
| M-7 | Pas de guide déploiement institutionnel | A11 | `docs/operations/deployment-institutional.md` |
| M-8 | Pas de politique RGPD | A11 | `docs/operations/data-retention-rgpd.md` + purge auto |
| M-9 | Pas de déclaration a11y | A7 + A11 + A15 | `ACCESSIBILITY.md` (draft A7, draft validé A11, finalisé A15) |
| M-10 | Pas de COI | A10 | Section dans `GOVERNANCE.md` + `paper.md` |
| M-11 | Pas de CODEOWNERS / governance | A10 | `.github/CODEOWNERS` + `GOVERNANCE.md` + `CODE_OF_CONDUCT.md` |
| M-12 | Snapshots reproductibilité sous-doc | A8 | `docs/reproducibility-snapshots.md` |
| M-13 | Tests concurrence runner+SSE | A5 | `test_runner_concurrency.py` + `test_sqlite_concurrent_writes.py` |
| M-14 | Pas d'anti-régression CER | A5 | `perf_regression.yml` cron hebdo + corpus de référence |
| M-15 | Pas de timeout pytest | A1 | `[tool.pytest.ini_options] timeout = 300` |
| M-16 | Pas de lazy loading rapports | A5 | Option `--lazy-images` dans `picarones report` |
| M-17 | Doc déséquilibrée FR/EN | A11 | 5 fichiers traduits + `test_translation_parity.py` |
| M-18 | `.dockerignore` + `.env.example` | A8 | Deux fichiers créés |
| M-19 | Compteur tests faux × 3 | A13 | Génération auto via `gen_readme_tables.py` |
| M-20 | Roadmap arrêtée Sprint 22 | A13 | Roadmap condensée par phase + lien CHANGELOG |
| M-21 | Known Issues obsolète | A13 | Section supprimée + redirection `docs/audits/` |
| M-22 | Project Structure trompeuse | A13 | `gen_project_structure.py` + insertion balisée |
| M-23 | Kraken / customYAML annoncés | A13 | Suppression du tableau (statut documenté en SPECS A14) |
| M-24 | Variables `AWS_*` sans adapter | A13 | Suppression des 3 lignes |
| M-25 | CLI sous-documentée | A13 | Génération auto via `picarones --help` |
| M-26 | API web sous-documentée | A13 | Génération auto via `app.openapi()` |
| M-27 | Métriques sous-vendues | A13 | 3 sous-sections + lien `docs/views.md` |
| M-28 | Sections rapport sous-vendues | A13 | Liste structurée 5 familles + screenshot |

### Mineurs (18/18 couverts)

| ID | Item | Sprint | Livrable spécifique |
|---|---|---|---|
| m-1 | `_app.js:1087` chaîne FR hardcodée | A7 | `I18N.no_anchor_data` |
| m-2 | `_app.js:1049` chaîne FR hardcodée | A7 | `I18N.no_gini` |
| m-3 | Bouton « Réinitialiser » sans i18n | A6 | `i18n.reset_all` |
| m-4 | Tableaux sans `scope="col"` | A6 | Audit + ajout sur ~12 tables |
| m-5 | Palette non daltonien-friendly | A7 | Palette Okabe-Ito + toggle |
| m-6 | Nombres non localisés | A7 | `toLocaleString(I18N.locale)` |
| m-7 | Pre-commit non rejoué en CI | A1 | `.github/workflows/precommit.yml` |
| m-8 | Pas de Python 3.13 | A1 | Matrice `["3.11", "3.12", "3.13"]` |
| m-9 | API stability defaults | A1 | `test_public_api_signatures.py` |
| m-10 | Tests cloud OCR sans HTTP errors | A5 | `test_cloud_http_errors.py` (12×3 cas) |
| m-11 | Versionnement testdata | A8 | `tests/.testdata/VERSION.yaml` |
| m-12 | Numérotation sprint des tests | A2 | Helper `conftest.py` informatif |
| m-13 | `requirements.txt` divergent | A8 | Alias `-r requirements.lock` ou suppression |
| m-14 | `pricing.yaml` staleness | A8 | `valid_until:` + Fact `PRICING_STALENESS_WARNING` |
| m-15 | PyInstaller `hiddenimports` manuels | A9 | `collect_all` auto + job CI `build-exe` |
| m-16 | Extras placeholder vides | A9 | Suppression `historical = []` / `importers = []` |
| m-17 | Test mesures importe Cercle 3 | A3 | Déplacement vers `tests/integration/` |
| m-18 | Petits items README (copyright, etc.) | A13 | Refonte intégrale couvre |
| §9.3 | Compteurs tests divergents 3 docs | A13 | Source unique CLAUDE.md, vérifié par A2 |

**Total : 13 + 28 + 18 + 1 = 60 entrées (les 59 items + §9.3
bookkeeping). Aucune entrée orpheline.**

---

## 13. Risques transverses et stratégies de contingence

### Risque R-1 — JOSS reviewer demande des changements méthodologiques de fond

*Probabilité* : moyenne (JOSS est généralement constructif mais
exigeant sur la rigueur statistique).
*Impact* : peut décaler la publication de 4 à 8 semaines
supplémentaires.
*Mitigation* :
- A12 prépare un draft *défendable* (refs primaires citées, tests
  méthodologiques rigoureux comme `test_friedman_canonical`, etc.) ;
- Sprint A12-bis dédié et planifié en buffer Phase 8 (sem. 12+) ;
- Si refus JOSS, fallback arXiv preprint (sans peer-review formel
  mais citable et DOI-stable).

### Risque R-2 — Audit RGAA externe trouve un bloqueur niveau A imprévu

*Probabilité* : faible si A6 est rigoureux (pytest + axe-core auto).
*Impact* : sprint A15-bis nécessaire, retard ~2 semaines.
*Mitigation* :
- A6 inclut un audit `axe-core` avant la fin du sprint, pas
  seulement à la livraison ;
- A15 prévoit explicitement un buffer A15-bis ;
- Si critique : démarche de **conformité partielle** documentée
  avec dérogations argumentées (acceptable RGAA).

### Risque R-3 — Refactor architecture casse un test caché ou un
consommateur externe

*Probabilité* : faible (lint passe, 3 356 tests, ré-exports
rétrocompat). *Impact* : régression silencieuse découverte tard.
*Mitigation* :
- A3 maintient des ré-exports rétrocompat avec `DeprecationWarning` ;
- `test_circle_dependencies.py` détecte toute nouvelle violation
  immédiatement ;
- Suppression effective des ré-exports planifiée 2 versions plus
  tard (v1.3.0 minimum), pas dans la même release.

### Risque R-4 — Indisponibilité d'une dépendance cloud (HF Datasets,
Gallica, MUFI registry)

*Probabilité* : faible. *Impact* : tests d'importation cassent en
CI.
*Mitigation* :
- Tous les imports de corpus en CI sont mockés (vérifier en A1) ;
- Le test de cohérence `docs/normalization-specs.md` archive les
  versions des spécifications sources dans le repo, pas via lien
  externe live.

### Risque R-5 — Le mainteneur unique manque de bande passante

*Probabilité* : forte sur projet académique mono-personne. *Impact* :
glissement calendrier.
*Mitigation* :
- Le plan est **séquencé pour qu'un sprint soit fermable en isolation**
  — abandon partiel possible sur les phases tardives sans casser
  les phases livrées (par ex. A15 peut s'étaler tant que les
  audits externes tournent) ;
- Documentation A11 + GOVERNANCE A10 préparent l'arrivée de
  contributeurs domain-experts (paléographe, archiviste, dev EN
  natif) — recrutement ouvert dès Phase 5 ;
- En cas de blocage long, annoncer un *« release v1.1 minimal
  institutional »* après A11 (Phase 5 close) — ce serait déjà un
  saut qualitatif majeur sur l'état actuel.

### Risque R-6 — Découverte d'un bug de fond lors d'un audit externe

*Probabilité* : faible (la base de tests est solide). *Impact* :
hotfix sprint nécessaire.
*Mitigation* :
- A1 (scanners + cov-fail-under) attrape la majorité des cas
  silencieux ;
- A8 (snapshots reproductibles) permet de rejouer un benchmark
  pré-bug pour confirmer la régression ;
- Politique de hotfix sécurité 72 h documentée en A10.

---

## 14. Définition de « niveau BnF / British Library atteint »

Le projet est considéré au niveau institutionnel cible quand
**toutes** les conditions suivantes sont satisfaites simultanément.
Cette liste est la *Definition of Done* du présent plan.

### Côté code

- [ ] Audit interne `institutional-readiness-2026-05.md` : 0 BLOCKER
      ouvert, 0 MAJOR ouvert.
- [ ] `pytest tests/` : 100 % vert sur Linux + macOS + Windows
      × Python 3.11 + 3.12 + 3.13.
- [ ] `ruff check` 0 erreur.
- [ ] `mypy picarones/core/ --strict` 0 erreur.
- [ ] `bandit -r picarones/ -ll` 0 issue HIGH/CRITICAL.
- [ ] `pip-audit --strict` 0 vulnérabilité ouverte.
- [ ] Couverture ≥ 85 % (`--cov-fail-under=85` actif et tenu).
- [ ] `tests/core/test_circle_dependencies.py` 0 violation.
- [ ] `tests/docs/test_readme_consistency.py` 0 divergence.
- [ ] `tests/docs/test_specs_consistency.py` 0 divergence.

### Côté distribution

- [ ] `pip install picarones` fonctionne depuis PyPI sur 3 OS.
- [ ] `docker pull ghcr.io/maribakulj/picarones:<version>` retourne
      une image immutable épinglée.
- [ ] `requirements.lock` versionné et utilisé en prod (Dockerfile).
- [ ] Release GitHub `v1.x.y` automatique avec corps depuis CHANGELOG.

### Côté reproductibilité

- [ ] Un benchmark rejoué à 6 mois d'intervalle avec mêmes lock file +
      digest Docker + commit picarones produit un rapport bit-à-bit
      identique (ou avec diff explicable).
- [ ] `docs/reproducibility-snapshots.md` documente la procédure
      end-to-end.

### Côté communication scientifique

- [ ] `CITATION.cff` valide (`cffconvert --validate`), bouton
      « Cite this repository » fonctionnel sur GitHub.
- [ ] DOI Zenodo attribué et présent dans le README.
- [ ] Papier JOSS : statut `accepted` + DOI JOSS dans CITATION.cff
      comme `preferred-citation`. *(ou, en fallback, arXiv preprint
      avec DOI alternatif).*
- [ ] `docs/normalization-specs.md` : chaque profil a sa source citée
      avec date d'extraction.

### Côté gouvernance et conformité

- [ ] `CODEOWNERS`, `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
      `LICENSE`, `ACCESSIBILITY.md`, `CITATION.cff` tous présents et
      non-vides (test CI `test_governance_files_present`).
- [ ] Audit RGAA externe : conformité WCAG 2.1 niveau AA ≥ 95 %, le
      résiduel listé en dérogations argumentées.
- [ ] Audit sécurité externe : 0 finding HIGH/CRITICAL.
- [ ] Politique RGPD documentée + purge auto fonctionnelle.
- [ ] Guide de déploiement institutionnel revu par au moins un DSI
      partenaire.

### Côté documentation produit

- [ ] README < 400 lignes, markdown valide, badges verts, tableaux
      auto-générés depuis le code.
- [ ] SPECS.md v2 reflète strictement le code, contient une section
      « Limites assumées » non vide.
- [ ] Documentation utilisateur et 4 guides développeur traduits en
      anglais avec parité de structure validée par test.
- [ ] CHANGELOG.md à jour, format Keep-a-Changelog respecté.

### Critère méta — sécabilité

- [ ] Le repo peut être récupéré et son rapport demo généré sur une
      machine vierge en moins de 5 minutes
      (`git clone && pip install -e . && picarones demo`), preuve
      par job CI dédié `test_quickstart`.

Quand ces 30 cases sont cochées, l'institution peut adopter Picarones
comme outil de référence interne et les chercheurs peuvent le citer
dans des publications scientifiques avec garantie de stabilité,
reproductibilité, et conformité.

---

*Plan rédigé en réponse à l'audit
[`institutional-readiness-2026-05.md`](institutional-readiness-2026-05.md)
sur la branche `claude/audit-institutional-readiness-8Cw4w`.*
*Période de mise en œuvre suggérée : mai–juillet 2026.*
