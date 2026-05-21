# Déploiement institutionnel — Picarones

> Ce guide cible les **DSI de bibliothèques nationales et services
> d'archives** qui souhaitent héberger Picarones sur leur propre
> infrastructure (intranet, derrière SSO, avec stockage centralisé)
> plutôt que sur HuggingFace Space public.
>
> Pour le déploiement HuggingFace Space ou un usage local rapide,
> voir [`how-to/install.md`](../how-to/install.md).

## Pré-requis

### Système

- **Linux x86_64 ou ARM64** (Debian 12+, RHEL 9+, Ubuntu 22.04+
  LTS, Rocky 9+).
- **Python 3.11 ou 3.12** (3.13 informationnel).
- **Tesseract OCR ≥ 5.3** (avec packs `fra`, `lat`, `eng` au
  minimum).
- **3 GB RAM par worker** (le ProcessPool spawne un sous-processus
  par moteur ; profil mémoire dominé par Pillow + jiwer).
- **5 GB de disque** pour l'application + 50 GB recommandés pour
  les uploads et la base SQLite des jobs.

### Réseau

- **Sortant** : optionnel (HF Datasets / Gallica / HTR-United
  uniquement si vous activez les imports distants).
- **Entrant** : un seul port HTTP (défaut 7860) à exposer derrière
  votre reverse proxy (Nginx, Apache, Traefik).

### Optionnel

- **PostgreSQL 14+** (en remplacement de SQLite si vous montez en
  multi-instance — voir § Architecture cible).
- **Reverse proxy SSO** (Shibboleth, CAS, OIDC, OAuth2 proxy).
- **Stack observabilité** (Prometheus + Grafana, ou ELK/Loki).

## Architecture cible

### Mono-instance (recommandé pour < 10 utilisateurs simultanés)

```
[Utilisateur] → [Reverse proxy SSO] → [Picarones Docker] → [SQLite jobs.db]
                                              ↓
                                    [Volume persistant uploads/]
                                              ↓
                                    [Volume persistant reports/]
```

Configuration minimale, rétro-compatible avec le déploiement
HuggingFace Space. Le reverse proxy ajoute l'authentification
(SSO institutionnel) et la terminaison TLS.

### Multi-instance (charge > 50 jobs/h)

```
[Utilisateur] → [Load balancer + SSO] → [Picarones × N]
                                              ↓
                                  [PostgreSQL jobs (centralisé)]
                                              ↓
                                  [Volume NFS uploads/ partagé]
```

Notes :
- **PostgreSQL** : `JobStore` utilise SQLite par défaut.
  Pour PostgreSQL, dériver une classe `PostgresJobStore` qui
  implémente la même API (`create_job`, `update_progress`,
  `get_job`, etc.). À défaut, partager la BD SQLite via NFS ne
  fonctionne pas — le mode WAL exige un filesystem local.
- **Volume NFS** pour `uploads/` et `reports/` afin que tous les
  workers voient les mêmes fichiers.
- **Sticky sessions** sur le LB pour SSE (les progress streams
  doivent rester sur le même worker).

## Configuration

Toutes les variables sont documentées dans
[`.env.example`](../../.env.example). Les principales pour un
déploiement institutionnel :

```bash
# Sécurité (Sprints A4 + 24)
PICARONES_PUBLIC_MODE=        # vide ou 0 = mode dev (autorise OCR cloud)
PICARONES_CSRF_REQUIRED=1     # OBLIGATOIRE derrière SSO
PICARONES_CSRF_SECRET="$(openssl rand -hex 32)"

# Restrictions
PICARONES_BROWSE_ROOTS="/var/lib/picarones/uploads:/data/corpus"
PICARONES_MAX_UPLOAD_MB=500
PICARONES_MAX_CONCURRENT_JOBS=8
PICARONES_RATE_LIMIT_PER_HOUR=0   # 0 = illimité (le SSO gère l'identité)

# Persistance
PICARONES_JOBS_DB=/var/lib/picarones/jobs.sqlite

# RGPD
PICARONES_UPLOAD_RETENTION_DAYS=7
PICARONES_LOG_IP_RETENTION_HOURS=24
```

## Intégration SSO

Picarones n'implémente **pas** de mécanisme d'authentification
natif — l'authentification est déléguée à votre reverse proxy.
Pattern recommandé : header trusté `X-Remote-User`.

### Nginx + Shibboleth (université, Renater)

```nginx
location / {
    auth_request /shibauthorizer;
    proxy_set_header X-Remote-User $http_remote_user;
    proxy_set_header X-Remote-Groups $http_remote_groups;
    proxy_pass http://picarones-backend:7860;

    # SSE long-polling
    proxy_buffering off;
    proxy_read_timeout 24h;
}
```

### Apache + CAS (CRU, ESR français)

```apache
<Location />
    AuthType CAS
    Require valid-user
    RequestHeader set X-Remote-User %{REMOTE_USER}s
    ProxyPass http://localhost:7860/
    ProxyPassReverse http://localhost:7860/
</Location>
```

### Traefik + OIDC (déploiements modernes)

```yaml
http:
  middlewares:
    oidc-auth:
      forwardAuth:
        address: "http://oauth2-proxy:4180/auth"
        trustForwardHeader: true
        authResponseHeaders:
          - X-Auth-Request-User
          - X-Auth-Request-Email

  routers:
    picarones:
      rule: "Host(`picarones.institution.fr`)"
      service: picarones
      tls:
        certResolver: letsencrypt
      middlewares:
        - oidc-auth
```

## Sauvegarde et restauration

### Composants à sauvegarder

| Élément | Chemin | Stratégie |
|---|---|---|
| BD jobs | `/var/lib/picarones/jobs.sqlite*` | Snapshot quotidien (mode WAL : sauvegarder `.sqlite`, `.sqlite-wal`, `.sqlite-shm` ensemble) |
| Uploads | `/var/lib/picarones/uploads/` | Snapshot hebdomadaire, rétention 30 jours (cf. RGPD) |
| Rapports | `/var/lib/picarones/reports/` | Snapshot hebdomadaire, rétention illimitée (artefacts citables) |
| Historique longitudinal | `/var/lib/picarones/history.sqlite` | Snapshot quotidien |
| Configuration | `/etc/picarones/`, `.env` | Versionner dans le système de gestion de config (Ansible, Salt) |

### Restauration

```bash
# Arrêt du service
systemctl stop picarones

# Restauration BD
cp backups/jobs.sqlite-2026-05-01.sqlite /var/lib/picarones/jobs.sqlite

# Restauration uploads
rsync -av backups/uploads-2026-05-01/ /var/lib/picarones/uploads/

# Redémarrage
systemctl start picarones

# Vérification : marquage des jobs orphelins
curl http://localhost:7860/api/status
```

Les jobs `running` au moment du snapshot sont automatiquement
marqués `interrupted` au redémarrage. Le tableau de
bord sera donc cohérent.

## Migration de schéma BD

Picarones évolue sa BD via une stratégie **append-only** —
nouvelles colonnes ajoutées avec `ALTER TABLE ADD COLUMN ...
DEFAULT NULL`. Aucune migration destructive entre versions
mineures.

Pour vérifier la compatibilité d'une BD existante avec une
nouvelle version :

```bash
sqlite3 jobs.sqlite "PRAGMA table_info(jobs);" > current_schema.txt
# Comparer avec docs/schema/jobs.sqlite.X.Y.Z.sql versionné
```

Si une migration majeure est nécessaire (changement de moteur SQL,
structure incompatible), elle sera annoncée 2 versions mineures
avant et un script de migration sera fourni dans `scripts/migrate/`.

## Observabilité

### Logs structurés

Picarones logge en **format texte simple** par défaut. Pour ELK /
Loki / Datadog, ajouter un wrapper JSON :

```python
# /etc/picarones/logging.conf
[handler_json]
class = pythonjsonlogger.jsonlogger.JsonFormatter
format = %(asctime)s %(levelname)s %(name)s %(message)s
```

Variable d'env : `PICARONES_LOG_FORMAT=json` (à implémenter dans
un sprint ultérieur — actuellement les logs sont en plain text).

### Métriques Prometheus (recommandé)

L'exposition Prometheus n'est pas livrée par défaut. Pour
l'ajouter, monter un conteneur sidecar `prometheus_client_python`
qui expose :

- `picarones_jobs_total{status="..."}`
- `picarones_jobs_duration_seconds`
- `picarones_uploads_size_bytes_total`
- `picarones_engine_invocations_total{engine="..."}`

Voir `docs/operations/observability.md`.

### Healthcheck

Un endpoint `/health` minimal répond en < 50 ms
sans toucher à la BD ni aux engines. Configurer le LB pour le
cibler avec un timeout court (5 s).

## Sécurité réseau

### Liste blanche réseau

Si vos engines cloud sont activés (`PICARONES_PUBLIC_MODE` non
défini), autoriser en sortie uniquement :

| Domaine | Usage |
|---|---|
| `api.openai.com` | OpenAI / GPT-4o |
| `api.anthropic.com` | Claude |
| `api.mistral.ai` | Mistral OCR + LLM |
| `vision.googleapis.com` | Google Vision |
| `*.cognitiveservices.azure.com` | Azure Doc Intelligence |
| `huggingface.co` | Imports HF Datasets (optionnel) |
| `gallica.bnf.fr` | Imports Gallica (optionnel) |

### Politique de mots de passe

Les clés API LLM/OCR sont passées **en variables d'environnement
uniquement** (jamais sur le filesystem en clair). Voir [`SECURITY.md`](../../SECURITY.md).

## Mise à l'échelle

| Charge | Configuration |
|---|---|
| < 5 jobs/h, < 5 utilisateurs | Mono-instance, SQLite, 2 vCPU / 4 GB RAM |
| 5–50 jobs/h, < 20 utilisateurs | Mono-instance, SQLite, 4 vCPU / 8 GB RAM, ProcessPool 8 workers |
| > 50 jobs/h | Multi-instance derrière LB, PostgreSQL centralisé, NFS uploads |
| > 500 jobs/h | Considérer un orchestrateur de tâches dédié (Celery + Redis), hors scope Picarones |

## Checklist déploiement

- [ ] Tesseract installé avec packs de langues nécessaires.
- [ ] Variables d'env configurées (mode dev/public, CSRF, browse roots).
- [ ] Volume persistant pour `uploads/`, `reports/`, BD.
- [ ] Reverse proxy SSO en place (CSRF activé !).
- [ ] Backup automatique configuré (cron quotidien minimum).
- [ ] Healthcheck `/health` configuré dans le LB.
- [ ] TLS terminé au reverse proxy.
- [ ] Liste blanche réseau si engines cloud actifs.
- [ ] Rétention RGPD configurée
      (cf. [`data-retention-rgpd.md`](data-retention-rgpd.md)).
- [ ] Audit RGAA externe planifié si prestation publique
      (cf. [`accessibility.md`](accessibility.md)).
- [ ] Issue tracking institutionnel (Mantis, JIRA, Redmine)
      synchronisé avec les issues GitHub si pertinent.

## Aide et support

- Issues GitHub étiquetées `deployment` :
  <https://github.com/maribakulj/Picarones/issues?q=label%3Adeployment>
- Pour un support contractualisé (SLO renforcés, intégration
  spécifique), contractualiser une prestation séparément (modalités
  hors-projet, à définir au cas par cas).

---

*Dernière mise à jour : 2 mai 2026.*
