# Observabilité — Picarones

> **Audience** : opérateur (DSI institutionnelle, SRE).  Décrit
> comment instrumenter Picarones pour qu'il soit observable depuis
> Prometheus, Grafana, Loki, Datadog, etc.
>
> Pour la réponse aux incidents, voir [`runbook.md`](runbook.md).
> Pour le déploiement, voir [`deployment-institutional.md`](deployment-institutional.md).

## Principes

Picarones expose trois types de signaux :

1. **Logs structurés** (stdlib `logging`).  Tous les modules
   utilisent `logger = logging.getLogger(__name__)`.  Niveaux
   conventionnels : DEBUG, INFO, WARNING, ERROR.  Aucun `print` en
   production.
2. **Audit trail** spécifique : `[audit] <event> <key=value>`
   (par convention).  Émis par les endpoints sensibles
   (`POST/DELETE /api/jobs`).
3. **Endpoints de santé** : `GET /health`, `GET /version`.

L'export vers une plateforme observabilité (Prometheus, Datadog, ELK)
est laissé au déploiement institutionnel — Picarones ne pousse rien
de lui-même.

## Logs structurés

### Format recommandé

Configurer le root logger en JSON pour l'ingestion automatique :

```python
# /etc/picarones/logging.yaml
version: 1
disable_existing_loggers: false
formatters:
  json:
    format: '{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
handlers:
  stdout:
    class: logging.StreamHandler
    stream: ext://sys.stdout
    formatter: json
loggers:
  picarones:
    level: INFO
    handlers: [stdout]
    propagate: false
root:
  level: WARNING
  handlers: [stdout]
```

Activer au démarrage :

```bash
PICARONES_LOG_CONFIG=/etc/picarones/logging.yaml \
  uvicorn picarones.interfaces.web:create_app --factory ...
```

### Niveaux par module

| Module | Niveau prod recommandé |
|--------|------------------------|
| `picarones.adapters.*` | INFO |
| `picarones.app.services.*` | INFO |
| `picarones.interfaces.web.*` | INFO |
| `picarones.pipeline.*` | INFO (DEBUG si chasse à un bug d'orchestration) |
| `picarones.evaluation.*` | WARNING (très verbeux en INFO) |
| `picarones.adapters._retry` | WARNING (déjà bavard sur les retries) |

### Exemples de lignes utiles à monitorer

| Pattern | Signification | Alerte |
|---------|---------------|--------|
| `[adapter] erreur retryable.*` | Cloud API instable | > 10/min sur 5 min → page |
| `OCRAdapterError` | Échec définitif d'OCR | > 5/min → warning |
| `[job_runner] job .* en échec` | Job s'est terminé en error | track per-IP |
| `[audit] job_submitted` | Soumission de job | tracker pour audit RGPD |
| `[audit] job_cancelled` | Annulation de job | tracker pour audit RGPD |
| `WinError 87` | Filename Windows invalide | DEVRAIT être 0 (corrigé S59) — sinon régression |
| `database is locked` | SQLite contention | > 1/min → page |

## Audit trail

Les opérations sensibles produisent un log INFO normalisé :

```
INFO [audit] job_submitted job_id=abc123 corpus=bnf_xviii from=10.0.0.42
INFO [audit] job_cancelled job_id=abc123 from=10.0.0.42
```

Ces lignes sont **destinées à être conservées** selon la politique
RGPD de l'institution (cf. [`data-retention-rgpd.md`](data-retention-rgpd.md)).
Stockage minimum recommandé : 90 jours (audit interne) ; 5 ans si
soumis aux Archives nationales.

Pour ingestion SIEM :

```
filter '[audit] '
extract job_id, corpus, from
forward to siem.bnf.fr:514 (syslog)
```

## Endpoints de santé

### `GET /health`

Réponse `200 OK` si le process est en mesure de servir.  Vérifie :

- `JobStore` accessible (lecture)
- `WorkspaceManager` accessible (écriture sandbox)
- Pas de check sur les API cloud (un cloud down ne doit pas planter
  les health probes locales)

```json
{
  "status": "ok",
  "version": "1.3.0-dev",
  "job_store": "ok",
  "workspace": "ok"
}
```

À utiliser comme **liveness probe** (Kubernetes) ou **healthcheck**
(Docker).  Recommandation : every 30s, fail after 3 consecutive.

### `GET /version`

Réponse :

```json
{
  "version": "1.3.0-dev",
  "code_version": "git-sha-abc1234",
  "python": "3.11.15"
}
```

Utile pour déterminer la version déployée sans accès au filesystem.

## Métriques (à venir)

Picarones n'expose pas encore d'endpoint Prometheus `/metrics`.
Recommandation immédiate : monitorer les logs.

**Backlog** (cf. [`/docs/roadmap/backlog.md`](../roadmap/backlog.md)) :

- Compteur `picarones_jobs_total{status="complete|error|cancelled"}`
- Histogramme `picarones_job_duration_seconds`
- Compteur `picarones_adapter_calls_total{adapter, status}`
- Histogramme `picarones_adapter_latency_seconds{adapter}`
- Gauge `picarones_jobs_running` (instantané)

Implémentation visée : `prometheus_client` middleware FastAPI optionnel.

## Tracing distribué

Pour les institutions qui orchestrent Picarones avec d'autres services
(ETL, cataloguing), le tracing OpenTelemetry est recommandé.

État actuel : pas d'instrumentation native.  Une instrumentation
opportuniste via `opentelemetry-instrumentation-fastapi` peut être
activée par le déploiement sans modifier Picarones :

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from picarones.interfaces.web import create_app

app = create_app(state=...)
FastAPIInstrumentor.instrument_app(app)
```

## Dashboards Grafana — squelette

Les panels recommandés pour un dashboard Picarones :

1. **Jobs throughput** — courbes par status (complete/error/cancelled),
   stack area, 24 h.
2. **Adapter latency p50/p95/p99** par adapter (Tesseract, Pero,
   Mistral OCR, Google Vision, Azure DI, OpenAI, Anthropic, Mistral
   chat, Ollama).
3. **Error rate par adapter** — % d'erreurs sur la dernière heure.
4. **Concurrence** — `picarones_jobs_running` actuel, comparé à
   `PICARONES_MAX_CONCURRENT_JOBS`.
5. **Workspace size** — `du -sh /var/lib/picarones/workspaces` via
   exporter node.
6. **Heap RSS** du process Picarones (via node_exporter ou
   process_exporter).

## SLOs suggérés

Pour un déploiement institutionnel ouvert aux chercheurs :

| Métrique | SLO 30j | Action si dépassé |
|----------|---------|-------------------|
| Disponibilité `/health` | 99.5 % | Investiguer infra |
| Job completion rate | > 95 % | Examiner taux d'erreurs adapter |
| API p95 latency (CRUD jobs) | < 500 ms | Profiler le `JobStore` |
| Cloud adapter retry rate | < 5 % | Demander quota plus haut |

## Révisions

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-05 | Création initiale (S60) |
