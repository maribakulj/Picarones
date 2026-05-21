# SECURITY — Picarones

Picarones est conçu pour être déployé dans trois contextes très différents :

1. **Poste développeur** (Codespaces, laptop) — accès local uniquement, le
   garde-fou est ouvert pour fluidifier l'itération.
2. **Serveur d'institution** (intranet patrimonial, cluster scientifique) —
   accès authentifié interne, mais quelques utilisateurs peuvent lancer des
   benchmarks coûteux ; le serveur doit borner la consommation.
3. **Espace public** (HuggingFace Space, démo en ligne) — n'importe quel
   visiteur peut atteindre l'API ; les clefs serveur (OpenAI, Anthropic,
   Mistral, Azure…) ne doivent **pas** être exposées au DoS-financier.

Ce document décrit les contrôles disponibles et la configuration
recommandée pour chaque cas.

---

## Variables d'environnement de sécurité

| Variable | Défaut | Effet |
|----------|--------|-------|
| `PICARONES_PUBLIC_MODE` | non défini | Si `1`/`true` : refuse OCR cloud + LLM mutualisés et active rate limit. |
| `PICARONES_BROWSE_ROOTS` | (auto) | Liste de chemins (séparateur `:` Unix / `;` Windows) autorisés pour `/api/corpus/browse`. Surcharge le défaut. |
| `PICARONES_MAX_UPLOAD_MB` | `100` | Taille max d'une image uploadée. |
| `PICARONES_MAX_CONCURRENT_JOBS` | `2` | Plafond global de benchmarks simultanés (sémaphore en mémoire). |
| `PICARONES_RATE_LIMIT_PER_HOUR` | `5` (en mode public) | Jobs max par IP et par heure. `0` = désactivé. |
| `PICARONES_CSP` | (politique durcie) | Surcharge la `Content-Security-Policy` envoyée par le middleware. |
| `PICARONES_CSRF_REQUIRED` | non défini | Si `1`/`true`/`yes` : active la protection CSRF (double-submit cookie + signature HMAC) sur tout POST/PUT/PATCH/DELETE. Voir § « CSRF — déploiement institutionnel » ci-dessous. |
| `PICARONES_CSRF_SECRET` | (auto) | Secret HMAC pour signer les tokens CSRF. Si non défini, généré au démarrage avec un warning ; les tokens sont alors invalidés à chaque redémarrage. **À définir en production**. |

---

## CSRF — déploiement institutionnel

L'application embarque un middleware CSRF **désactivé par défaut**
(rétrocompat HuggingFace Space où il n'y a pas de session
authentifiée à protéger). Pour un déploiement BnF / Bibliothèque
nationale derrière SSO :

```bash
export PICARONES_CSRF_REQUIRED=1
export PICARONES_CSRF_SECRET="$(openssl rand -hex 32)"  # 64 chars hex
```

**Comment ça marche** : pattern « double-submit cookie ». Le serveur
pose un cookie `picarones_csrf` (httponly=False, samesite=strict) qui
contient un token `<nonce>.<HMAC-SHA256(secret, nonce)>`. Sur tout
POST/PUT/PATCH/DELETE non exempt, le client doit renvoyer le même
token dans l'en-tête `X-CSRF-Token`. Le serveur compare en temps
constant et vérifie la signature. Une page tierce ne peut pas lire
le cookie (samesite=strict + JS d'origine différente) ni produire
une signature valide (HMAC), donc ne peut pas forger une requête.

**Endpoints exemptés** : `/health`, `/api/csrf/token` (le endpoint
qui *donne* le token).

**Bootstrap d'un client tiers** (curl, scripts CI) :

```bash
# 1. Récupérer un token et persister le cookie dans un jar
curl -c cookies.txt http://picarones.example/api/csrf/token | jq -r .token

# 2. Réutiliser le token dans le header
TOKEN=$(jq -r .token < <(curl -sb cookies.txt http://.../api/csrf/token))
curl -b cookies.txt -H "X-CSRF-Token: $TOKEN" -X POST .../api/lang/fr
```

**Frontend** : le JS embarqué (`web-app.js`) wrappe `fetch()` pour
injecter automatiquement le header sur toute requête mutante
same-origin. Aucun changement requis dans le code applicatif.

---

## Contrôles par contexte

### 🧑‍💻 Développement (défaut, `PICARONES_PUBLIC_MODE` non défini)

```bash
picarones serve --port 8000
```

- Tous les moteurs OCR sont disponibles.
- `/api/corpus/browse` voit `cwd`, `./uploads/`, `/workspaces`, `tempdir`.
- Pas de rate limit.
- CSP appliquée mais permissive (`unsafe-inline` toléré tant que
  des handlers inline subsistent dans les templates).

### 🏛 Serveur d'institution

```bash
export PICARONES_BROWSE_ROOTS="/srv/corpus:/srv/uploads"
export PICARONES_MAX_CONCURRENT_JOBS=4
export PICARONES_MAX_UPLOAD_MB=500
picarones serve --host 0.0.0.0 --port 8000
```

À combiner avec une terminaison TLS et une authentification au niveau
reverse-proxy (nginx + auth basic, ou Keycloak/SAML). Picarones n'embarque
pas son propre système d'authentification — c'est un choix conscient pour
ne pas réinventer un sous-système qui sera mieux servi par l'infra existante.

### 🌐 HuggingFace Space / démo publique

```dockerfile
ENV PICARONES_PUBLIC_MODE=1
ENV PICARONES_RATE_LIMIT_PER_HOUR=5
ENV PICARONES_MAX_CONCURRENT_JOBS=2
ENV PICARONES_MAX_UPLOAD_MB=50
# Optionnel : surcharger les browse roots
# ENV PICARONES_BROWSE_ROOTS=/data/corpus
```

Effets en mode public :

- ❌ Moteurs OCR cloud (`mistral_ocr`, `google_vision`, `azure_doc_intel`)
  refusés en `403`.
- ❌ Pipelines OCR+LLM (`openai`, `anthropic`, `mistral`, `ollama`)
  refusés en `403`.
- ❌ `/api/corpus/browse` se limite à `./uploads/`.
- ⏱ `/api/benchmark/start` et `/api/benchmark/run` rate-limités en `429`.
- 🔒 `Content-Security-Policy` + `X-Frame-Options: DENY` +
  `X-Content-Type-Options: nosniff` + `Referrer-Policy: strict-origin-when-cross-origin`
  sur toutes les réponses.

---

## Contrôles d'upload

### Images
- **Validation Pillow** systématique : `Image.open(...).verify()` dans
  un `try/except` qui capture les `UnidentifiedImageError`,
  `DecompressionBombError`, et l'exception générique.
- **Limite de taille** par fichier (`PICARONES_MAX_UPLOAD_MB`).
- **Basename forcé** : un nom de fichier multipart contenant `..` ou `/`
  est tronqué à son nom de base avant écriture.

### Archives ZIP
- **Bombe ZIP** : taille décompressée bornée à 500 Mo, nombre de fichiers
  borné à 2000.
- **Path traversal** : seuls les noms de base sont conservés (les répertoires
  internes du ZIP sont aplatis).
- **Filtres macOS** : les fichiers `._*` (AppleDouble) sont ignorés.
- **Symlinks** : Python's `zipfile` n'extrait pas les symlinks par défaut ;
  un check explicite (`ZipInfo.external_attr & 0xA000`) est sur la roadmap
  comme défense en profondeur.

---

## Modèle de menace

| Menace | Mitigation |
|--------|-----------|
| Visiteur consomme la clef API mainteneur | `PICARONES_PUBLIC_MODE=1` → 403 sur LLM/OCR cloud. |
| DoS via 50 benchmarks concurrents | `PICARONES_MAX_CONCURRENT_JOBS` (sémaphore) + rate limit par IP. |
| Bombe Pillow (`CVE-2023-50447` & cie) | `Image.verify()` levant `DecompressionBombError`. |
| Path traversal sur browse / image / delete | Validation explicite + résolution + check `is_relative_to`. |
| Exfiltration via browse `/etc` ou `/root` | `PICARONES_BROWSE_ROOTS` restreint, défaut public limité à uploads. |
| XSS via paramètres URL | CSP `default-src 'self'`, `frame-ancestors 'none'`. |
| Clickjacking | `X-Frame-Options: DENY`. |

---

## Reporting de vulnérabilités

Les vulnérabilités potentielles peuvent être ouvertes via une *Security
Advisory* GitHub (privée par défaut) sur
[github.com/maribakulj/Picarones](https://github.com/maribakulj/Picarones).

Merci de **ne pas** divulguer publiquement avant qu'un correctif ne soit
disponible. Les contributeurs prendront en charge la triage en moins de
14 jours.
