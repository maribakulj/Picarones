# Threat model — Picarones

> **Audience** : DSI institutionnelle (BnF, LoC, BL), auditeur
> sécurité, mainteneur.  Ce document complète
> [`/SECURITY.md`](../../SECURITY.md) en formalisant le modèle de
> menace.  Méthodologie : **STRIDE** (Microsoft) + adaptation
> patrimoine numérique.
>
> **Périmètre** : déploiement institutionnel — Picarones tourne sur
> une infrastructure interne (NAS, cluster Kubernetes), un workspace
> partagé entre chercheurs, des clés API cloud côté serveur.
>
> **Hors périmètre** : déploiement public HuggingFace Space (mode
> ouvert anonymisé, sans secrets), CLI mono-utilisateur en local
> (modèle de menace = celui de la machine de l'utilisateur).
>
> **Statut** : v1, 2026-05.  À réviser à chaque release majeure ou
> incident sécurité.

## Acteurs

| Acteur | Confiance | Capacités |
|--------|-----------|-----------|
| **Utilisateur authentifié** (chercheur, archiviste BnF) | Modéré | Upload corpus, lance benchmark, lit rapport, télécharge artefacts |
| **Utilisateur invité** (lecteur d'un rapport publié) | Bas | Lit un rapport HTML produit |
| **Opérateur** (DSI institutionnelle) | Élevé | Déploie, configure, accède aux logs, gère les clés API |
| **Mainteneur** (équipe Picarones) | Élevé sur le code | Push code, release, accès limité aux instances de production |
| **Attaquant externe** | Aucune | Internet public ou utilisateur malveillant |

## Actifs à protéger

| Actif | Sensibilité | Pourquoi |
|-------|-------------|----------|
| **Corpus uploadés** | RGPD (peut contenir PII : registres d'état civil) | Article 4 RGPD — données personnelles si nominatives |
| **Vérités terrain (GT)** | Propriété intellectuelle de l'institution | Investissement humain coûteux ; secret de fait |
| **Clés API cloud** (`OPENAI_API_KEY`, etc.) | Secret crédential | Compromission = facturation arbitraire + exfiltration de données |
| **Résultats de benchmark** | Faible (résultats agrégés) | Sauf si attribués nominativement à un transcripteur |
| **Logs applicatifs** | Modéré (PII collatéral, métadonnées corpus) | Audit trail = preuve juridique mais aussi cible |
| **Code source** | Public (OSS) | Intégrité supply-chain (signed releases, SBOM, SLSA) |
| **Base SQLite des jobs** | Modéré (historique des runs, paramètres) | Permet de reconstituer l'activité d'un utilisateur |

## Surfaces d'attaque

```
┌──────────────────────────────────────────────────────────┐
│  Internet / Intranet                                     │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
   ┌───────────────────────────────────────┐
   │  FastAPI (interfaces/web)             │  ← S1 (HTTP), S2 (auth)
   │  - SecurityHeadersMiddleware          │
   │  - BodySizeLimitMiddleware            │
   │  - RateLimitMiddleware                │
   │  - AuthenticationMiddleware (opt-in)  │
   └────────────────────┬──────────────────┘
                        │
                        ▼
   ┌───────────────────────────────────────┐
   │  RunOrchestrator + JobRunner          │  ← S3 (job exec)
   │  - WorkspaceManager (sandbox)         │
   │  - ZIP extraction (zip-slip safe)     │
   └────────────────────┬──────────────────┘
                        │
        ┌───────────────┼─────────────────┐
        ▼               ▼                 ▼
   ┌──────────┐   ┌───────────┐    ┌─────────────┐
   │ Adapters │   │ Adapters  │    │ Storage     │  ← S4 (cloud)
   │ OCR cloud│   │ LLM cloud │    │ filesystem  │  ← S5 (FS)
   │ (HTTPS)  │   │ (HTTPS)   │    │ + SQLite    │  ← S6 (DB)
   └──────────┘   └───────────┘    └─────────────┘
```

## Menaces — analyse STRIDE

### S — Spoofing (usurpation d'identité)

| ID | Menace | Mitigation |
|----|--------|------------|
| S1 | Un attaquant se fait passer pour un utilisateur authentifié | `AuthenticationMiddleware` opt-in avec `AuthenticationBackend` Protocol — l'institution branche son SSO/LDAP/JWT.  Les endpoints `/health` et `/version` restent publics pour les sondes. |
| S2 | Un client forge `X-Forwarded-For` pour spoofer son IP dans le rate limit | `RateLimitMiddleware.trust_proxy_count: int` (défaut 0 = XFF ignoré).  Lecture du Nème IP en partant de la fin de la chaîne XFF.  Test `tests/interfaces/web/test_rate_limit_xff.py` (7 cas). |
| S3 | Un attaquant publie un faux package `picarones` sur PyPI | Le projet n'est pas encore sur PyPI public.  À la publication : signer les wheels avec Sigstore et publier le SLSA provenance level 3 (cf. backlog). |

### T — Tampering (altération)

| ID | Menace | Mitigation |
|----|--------|------------|
| T1 | Un utilisateur uploade un ZIP avec des chemins zip-slip pour écrire hors workspace | `WorkspaceManager` sandboxe par session, extraction ZIP filtre les chemins absolus et `..`. |
| T2 | Un caller construit `DocumentRef(id="../../etc/passwd")` programmatiquement | `_DOC_ID_RE` regex `^[A-Za-z0-9_.\-/]+$` + validateur Pydantic explicite qui rejette tout segment `..` (S59 #M3). |
| T3 | Un attaquant altère le schéma SQLite `jobs.db` entre deux démarrages | `JobStore.SCHEMA_VERSION` + dispatcher `_MIGRATIONS` qui rejette dur les schémas downgrade.  Pas de mitigation contre une altération en place — c'est au filesystem. |
| T4 | Un cache d'artefact corrompu ferait diverger un run | `ArtifactKey.hash_hex()` multi-paramètres (inputs hash + step + code_version + params + projection_spec) — un cache pollué est rejeté à la lecture parce que la clé ne match plus. |
| T5 | Une fonte / modèle local est remplacé par un fichier malveillant | Picarones ne charge aucun modèle automatiquement.  Les modèles Tesseract et Pero sont pointés explicitement par l'utilisateur ; à charge à lui de vérifier les hashes. |

### R — Repudiation (non-répudiation)

| ID | Menace | Mitigation |
|----|--------|------------|
| R1 | Un utilisateur lance un job coûteux puis nie l'avoir fait | `[audit]` log INFO sur `POST /api/jobs` et `DELETE /api/jobs/{id}` avec IP source (S59 #M2).  Logs structurés à conserver côté ops selon la politique RGPD. |
| R2 | Un attaquant modifie un rapport persisté pour falsifier les chiffres | Le `RunManifest` est byte-déterministe (`model_dump_json` Pydantic ordered).  Le hash SHA-256 du manifest peut être cité dans une publication pour ancrer la version.  Signature cryptographique : non implémentée, à arbitrer (cf. backlog). |
| R3 | Un mainteneur publie une release sans laisser de trace | GitHub Actions `release.yml` enregistre l'identité GitHub du déclencheur ; SLSA provenance (à venir) attestera la chaîne build → wheel. |

### I — Information disclosure

| ID | Menace | Mitigation |
|----|--------|------------|
| I1 | Une clé API cloud (`OPENAI_API_KEY`, etc.) fuit dans un log applicatif | Les adapters ne logent jamais la clé — vérifié par revue de code.  Les exceptions cloud sont catchées et le message reformulé sans inclure de header.  À durcir : un test `bandit` dans la CI sur les patterns `api_key` en variable de log. |
| I2 | Un rapport HTML embarque un CSP permissif et leak via XSS | `CSP: default-src 'self'`, pas de `unsafe-inline`, vérifié par `tests/interfaces/web/test_sprint_a14_s49_security.py`.  Le moteur narratif rend les chiffres via templates YAML (pas de injection HTML). |
| I3 | Le workspace partagé fait fuiter le corpus d'un chercheur à un autre | `WorkspaceManager` sandboxe par `session_id` ; aucun caller ne peut sortir de son workspace via `resolve_output_path`. |
| I4 | Un endpoint `GET /api/jobs/{job_id}` divulgue les paramètres d'un autre utilisateur | Pas d'isolation multi-tenants à ce jour — défaut documenté.  Le déploiement institutionnel doit ajouter une couche d'autorisation par utilisateur (cf. `AuthenticationMiddleware`). |
| I5 | Un attaquant lit `dependencies_lock` du `RunManifest` pour cibler une CVE | Acceptable — `dependencies_lock` est public par design (reproductibilité).  La défense est de patcher rapidement les CVE via `pip-audit` en CI. |

### D — Denial of Service

| ID | Menace | Mitigation |
|----|--------|------------|
| D1 | Upload ZIP géant qui sature le disque | `BodySizeLimitMiddleware` (défaut 100 MiB).  **Limite connue** : ne couvre pas `Transfer-Encoding: chunked` — recommandation = nginx `client_max_body_size` en amont (cf. [`operations/runbook.md`](../operations/runbook.md)). |
| D2 | Flood de requêtes saturant le rate limit en mémoire | `RateLimitMiddleware` avec eviction LRU `max_clients=10000` (S58).  Pas atomique sous très haute concurrence — best-effort assumé. |
| D3 | Job qui hang sur appel cloud (timeout réseau) | `pytest-timeout 5 min` par test ; `urllib.request.urlopen(timeout=)` configurable par adapter ; `call_with_retry` partagé (3 retries 2/4/8s) qui FAIL fast si non-retryable. |
| D4 | DAG cyclique ou infini dans une `PipelineSpec` | Validation statique avec détection de cycle dans `pipeline/validation.py` ; rejet `PipelineSpecError` au load. |
| D5 | XML billion-laughs / XXE sur upload ALTO/PAGE | `defusedxml` exclusif dans `formats/alto/parser.py` et `formats/pagexml/parser.py`. |

### E — Elevation of privilege

| ID | Menace | Mitigation |
|----|--------|------------|
| E1 | Un module contribué tiers s'exécute avec des privilèges qu'il ne devrait pas | `BaseModule` interface stricte ; `module_policy.audit_module` valide qu'un module externe ne dérive que de `BaseModule` et déclare ses `input_types`/`output_types` proprement.  Pas de sandboxing process — un module malicieux peut faire `os.system`. |
| E2 | Un utilisateur web arrive à exécuter du code arbitraire via l'API | `RunSpec` est validé par Pydantic ; `adapter_class` est un dotted-path résolu via `importlib.import_module` mais filtré contre une liste explicite via `RegistryService.bootstrap_defaults()`.  Une release institutionnelle doit verrouiller cette liste. |

## Risques résiduels acceptés

| ID | Risque | Pourquoi accepté |
|----|--------|------------------|
| RR1 | Le rate limit n'est pas atomique sous très haute concurrence | Best-effort suffit pour usage institutionnel ; un Redis-backed rate limiter est l'évolution si besoin |
| RR2 | Un module Python contribué peut faire des `os.system` arbitraires | Le modèle de confiance est *« le mainteneur a revu le code »* — pas de sandbox process.  Pour un usage institutionnel multi-tenant, déployer dans un conteneur isolé par tenant. |
| RR3 | Les clés API cloud sont en variables d'environnement, pas en HSM | Standard de l'industrie ; un Vault-backed secret store est l'évolution si la DSI l'exige. |
| RR4 | Pas d'isolation multi-tenants par user dans le workspace web | Documentée explicitement ; déploiement multi-tenants doit ajouter sa propre couche d'autorisation. |

## Procédure de signalement

Voir [`/SECURITY.md`](../../SECURITY.md) pour le canal de
divulgation responsable.  La version anglaise est dans
[`/SECURITY.en.md`](../../SECURITY.en.md).

## Révisions

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-05 | Création initiale (S60), méthodologie STRIDE |
