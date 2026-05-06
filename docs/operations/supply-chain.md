# Supply chain — SBOM, SLSA, signatures

> **Audience** : DSI institutionnelle et conformité réglementaire
> (EU CRA — Cyber Resilience Act, exigible à partir de 2027 pour les
> livraisons à des organismes publics européens).
>
> Décrit comment Picarones documente sa chaîne d'approvisionnement
> logicielle et permet à une institution de vérifier l'intégrité
> d'un wheel ou d'une image Docker avant déploiement.

## SBOM (Software Bill of Materials)

### Format CycloneDX

Picarones produit un SBOM au format **CycloneDX 1.5 JSON** à chaque
release.  Le SBOM liste l'intégralité des paquets Python installés
dans l'environnement de build avec :

- `name`, `version`, `purl` (package URL canonique).
- `licenses` (SPDX expression).
- `hashes` (SHA-256 du wheel).
- `dependencies` (graphe de dépendance complet).

Génération locale :

```bash
pip install cyclonedx-bom
python scripts/gen_sbom.py --output sbom.json
```

Génération automatique dans la CI : voir
[`.github/workflows/release.yml`](../../.github/workflows/release.yml)
qui attache `sbom.json` à chaque GitHub Release.

### Image Docker

L'image Docker `ghcr.io/maribakulj/picarones:<version>` embarque son
propre SBOM (couche métadonnées BuildKit) :

```bash
docker buildx imagetools inspect \
  ghcr.io/maribakulj/picarones:<version> \
  --format '{{ json .SBOM }}'
```

## SLSA Provenance

[SLSA](https://slsa.dev/) (Supply-chain Levels for Software Artifacts)
formalise le niveau de confiance qu'on peut accorder à un artefact
livré.

### État actuel : SLSA Level 2

- **Build** isolé sur GitHub-hosted runners, traçable au commit SHA.
- **Provenance** générée automatiquement par
  [`docker/build-push-action@v5`](https://github.com/docker/build-push-action)
  avec `provenance: true`.

Inspection :

```bash
docker buildx imagetools inspect \
  ghcr.io/maribakulj/picarones:<version> \
  --format '{{ json .Provenance }}'
```

### Trajectoire vers SLSA Level 3

Pour atteindre le niveau 3 (signature non-falsifiable), prochaines
étapes (cf. [`/docs/roadmap/backlog.md`](../roadmap/backlog.md)) :

1. Signer chaque wheel PyPI avec [Sigstore](https://www.sigstore.dev/)
   via `pypi-attestations` (PEP 740).
2. Signer le SBOM avec `cosign sign-blob` lors de la release.
3. Publier les attestations sur Rekor (transparency log).

## Vérification côté institution

Avant déploiement, l'institution peut vérifier qu'un wheel n'a pas
été altéré entre le build CI et le download :

```bash
# 1. Téléchargement.
pip download picarones==<version> --no-deps -d /tmp/audit/

# 2. Vérification du hash contre le SBOM.
sha256sum /tmp/audit/picarones-*.whl
jq -r '.components[] | select(.name == "picarones") | .hashes[0].content' sbom.json
# Les deux valeurs doivent matcher.

# 3. (Future, SLSA L3) Vérification de la signature Sigstore.
# cosign verify-blob --bundle picarones-<version>.whl.sigstore picarones-<version>.whl
```

## Politique de mise à jour des dépendances

- **CVE critique** (CVSS ≥ 9.0) : patch release sous 7 jours.
- **CVE élevée** (7.0 ≤ CVSS < 9.0) : minor release sous 30 jours.
- **CVE moyenne** : prise en compte au prochain cycle de release.

Surveillance :

- `pip-audit` exécuté en CI sur chaque push (cf.
  [`/.github/workflows/precommit.yml`](../../.github/workflows/precommit.yml)).
- Dependabot / Renovate sur `pyproject.toml` pour les minor / patch.

## Conformité EU CRA (anticipation)

L'EU Cyber Resilience Act, applicable à partir de 2027 pour les
produits livrés à des entités publiques de l'UE, exigera :

| Exigence CRA | Statut Picarones |
|--------------|------------------|
| SBOM machine-readable | ✅ CycloneDX 1.5 |
| Vulnerability disclosure policy | ✅ [`/SECURITY.md`](../../SECURITY.md) + RFC 9116 [`/.well-known/security.txt`](../../.well-known/security.txt) |
| Coordinated vulnerability disclosure | ✅ GitHub Security Advisories |
| Cryptographic signing of releases | 🔧 SLSA L2 actuel, L3 prévu |
| Vulnerability handling within reasonable timeframes | ✅ Politique documentée ci-dessus |
| Security updates for at least 5 years | 🔧 Politique LTS à définir avant 1.0 GA |

## Révisions

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-05 | Création initiale (S60) |
