# Procédure de release

> Procédure pour publier une release Picarones (PyPI + Docker + GitHub
> Release).  Contexte historique :
> [`docs/archive/2026-audits/remediation-plan.md`](../archive/2026-audits/remediation-plan.md).

## Pré-requis avant tout tag

| Vérif | Commande | Cible |
|---|---|---|
| Tests verts | `pytest tests/ -q` | 0 failed |
| Lint propre | `ruff check picarones/ tests/` | All checks passed |
| Type strict | `python -m mypy picarones/domain/` | 0 erreur |
| Sécurité statique | `bandit -ll -r picarones/` | 0 HIGH |
| CVEs deps | `pip-audit` | aucune CVE non-mitigée dans le runtime |
| CHANGELOG | section `## [X.Y.Z] — YYYY-MM-DD` présente | |
| Compteurs doc | `python scripts/gen_readme_tables.py --check` | exit 0 |

## Mode public vs institutionnel

Pour un déploiement BnF (mode institutionnel), s'assurer que les
variables d'environnement de production sont prêtes **avant** de
tagger.  L'app refuse de démarrer sans.

| Variable | Public (HF Space) | Institutionnel |
|---|---|---|
| `PICARONES_PUBLIC_MODE` | `1` | non set |
| `PICARONES_CSRF_REQUIRED` | non set | `1` |
| `PICARONES_CSRF_SECRET` | non set | **OBLIGATOIRE** |
| `PICARONES_LOG_FORMAT` | non set | `json` (recommandé pour ops) |

### Génération du CSRF secret

```bash
# 32 bytes hex
openssl rand -hex 32

# Persister dans le secret manager institutionnel :
#   - Vault : ``vault kv put secret/picarones csrf=<hex>``
#   - AWS Secrets Manager : ``aws secretsmanager create-secret``
#   - Kubernetes : ``kubectl create secret generic picarones-csrf``
#   - Docker Compose : ``.env`` non versionné
```

**Ne JAMAIS** committer ce secret dans un Dockerfile, un
docker-compose.yml versionné, ou un dépôt git public.

## Vue d'ensemble

Une release Picarones produit **trois artefacts** :

1. Un wheel + sdist sur **PyPI** (`pip install picarones==X.Y.Z`).
2. Une image Docker **multi-arch** sur ghcr.io
   (`docker pull ghcr.io/maribakulj/picarones:X.Y.Z`).
3. Une **GitHub Release** avec le sdist/wheel attachés et les
   release notes extraites du `CHANGELOG.md`.

Le pipeline est entièrement automatisé : il suffit de pousser un
tag `v*.*.*` pour déclencher l'enchaînement complet (workflow
[`.github/workflows/release.yml`](../../.github/workflows/release.yml)).

## Procédure release standard

### Pré-requis (une fois)

1. **PyPI Trusted Publisher** : sur <https://pypi.org/manage/account/publishing/>,
   ajouter ce repo + workflow `release.yml` + environnement `pypi`.
   Idem pour TestPyPI dans l'environnement `testpypi`.
2. **GitHub repo** : créer les environnements `pypi` et `testpypi`
   dans Settings → Environments, et marquer `pypi` comme "required
   reviewers" si vous voulez une validation manuelle finale.
3. **GHCR** : `packages: write` sur `GITHUB_TOKEN` est natif (rien
   à configurer).

### Cycle de release

```bash
# 1. Vérifier que main est vert + à jour
git checkout main
git pull --ff-only

# 2. Mettre à jour le CHANGELOG.md (Keep a Changelog)
#    Ajouter une section ## [1.2.0] — YYYY-MM-DD avec les changes
git add CHANGELOG.md
git commit -m "docs(changelog): release 1.2.0"

# 3. Tag annoté + push
git tag -a v1.2.0 -m "Picarones 1.2.0"
git push origin main
git push origin v1.2.0

# 4. Surveiller le workflow Actions
gh run watch
```

Le workflow déroule **automatiquement** :

1. **build** — sdist + wheel via setuptools_scm (version dérivée du tag),
   `twine check`, smoke test wheel install.
2. **publish-testpypi** — upload TestPyPI via OIDC trust.
3. **testpypi-smoke** — installation depuis TestPyPI dans un container
   vierge + `picarones demo`.
4. **publish-pypi** — upload PyPI via OIDC trust (production).
5. **docker** — build multi-arch (linux/amd64 + linux/arm64) avec
   QEMU, push ghcr.io, attestations SLSA + SBOM.
6. **github-release** — création de la Release GitHub avec corps
   extrait depuis la section CHANGELOG correspondante.

Durée totale : ~15 min (multi-arch + 30s d'indexation TestPyPI).

## Versionnement

Picarones suit **Semantic Versioning 2.0.0** :

- `MAJOR.MINOR.PATCH` — incompatibilité, ajout, fix.
- Suffixes pré-release : `-rc1`, `-beta1`, `-alpha1`. Le workflow
  les détecte et coche `prerelease=true` sur la GitHub Release.

`setuptools_scm` dérive automatiquement la version du tag git :

| Contexte | Version produite |
|---|---|
| Tag `v1.2.0` | `1.2.0` |
| 5 commits après `v1.2.0` | `1.2.1.dev5+g<sha>` (dev seulement) |
| `v1.3.0-rc1` | `1.3.0rc1` (PEP 440) |

## Procédure d'urgence : hotfix sécurité

Pour un fix CVE qui doit sortir en < 72 h (politique GOVERNANCE.md) :

```bash
git checkout -b hotfix-cve-2026-XXXX main
# correctif minimal + test
git commit -m "fix(security): patch CVE-2026-XXXX"
# CHANGELOG bump
git commit -m "docs(changelog): release 1.2.1"
git tag -a v1.2.1 -m "Picarones 1.2.1 (security)"
git push origin hotfix-cve-2026-XXXX v1.2.1
# Le workflow release.yml gère le reste.
# Après merge : annonce sur SECURITY.md + courriel mainteneur.
```

## Yanking d'une release publiée

PyPI permet de retirer (yank) une version compromise sans la
supprimer. À utiliser si une release introduit une régression
critique :

```bash
# Connexion à PyPI → Manage → version concernée → "Yank"
# Justification dans le commentaire (visible publiquement).
```

L'image ghcr.io reste — mais le tag `:latest` ne pointera plus vers
la version yankée si on pousse une nouvelle release.

## Validation post-release

Checklist 30 min après la fin du workflow :

- [ ] `pip install picarones==<version>` fonctionne dans un venv frais.
- [ ] `docker run ghcr.io/maribakulj/picarones:<version>` démarre
      sans erreur et expose `/health`.
- [ ] La GitHub Release affiche bien les release notes attendues.
- [ ] `cffconvert --validate` confirme que `CITATION.cff` cite la
      bonne version.

## Annexe : rollback complet

Si la release est compromise et doit être retirée intégralement :

1. PyPI : yank la version (cf. plus haut).
2. ghcr.io : `docker manifest rm ghcr.io/maribakulj/picarones:<version>`.
3. GitHub Release : passer en draft + ajouter un README explicatif.
4. Tag git : `git push --delete origin v<version>` puis nouveau
   tag `v<version>+1` corrigé (un tag git ne peut pas être réécrit
   sans casser tous les checkouts existants — préférer le bump).

Ne **jamais** force-push un tag déjà publié — les utilisateurs qui
ont fait `git fetch` voient un conflit.
