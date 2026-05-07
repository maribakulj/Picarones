# Runbook — réponse aux incidents Picarones

> **Audience** : opérateur (DSI institutionnelle, SRE) en garde
> active.  Ce document liste les incidents prévisibles et les
> procédures de mitigation.  Pour le déploiement initial, voir
> [`deployment-institutional.md`](deployment-institutional.md) ;
> pour l'observabilité, voir [`observability.md`](observability.md).
>
> **Convention** : chaque scénario suit le format
> `Symptôme → Diagnostic → Mitigation → Suivi`.

## Index des scénarios

| ID | Scénario | Sévérité | Page |
|----|----------|----------|------|
| INC-01 | Job stuck en `running` | MAJOR | [§INC-01](#inc-01--job-stuck-en-running) |
| INC-02 | Disk full sur le workspace | BLOCKER | [§INC-02](#inc-02--disk-full-sur-le-workspace) |
| INC-03 | Cloud API rate limit / quota dépassé | MAJOR | [§INC-03](#inc-03--cloud-api-rate-limit) |
| INC-04 | SQLite `database is locked` | MAJOR | [§INC-04](#inc-04--sqlite-database-is-locked) |
| INC-05 | Memory leak (RSS qui croît continûment) | MAJOR | [§INC-05](#inc-05--memory-leak) |
| INC-06 | Compromission d'une clé API cloud | BLOCKER | [§INC-06](#inc-06--compromission-de-cl%C3%A9-api) |
| INC-07 | Rapport HTML corrompu / non-déterministe | MEDIUM | [§INC-07](#inc-07--rapport-html-corrompu) |
| INC-08 | CI bloquée > 30 min (déjà vu) | MEDIUM | [§INC-08](#inc-08--ci-bloqu%C3%A9e) |
| INC-09 | Upgrade qui casse les jobs en cours | MAJOR | [§INC-09](#inc-09--upgrade-casse-jobs) |
| INC-10 | Restauration depuis backup | MEDIUM | [§INC-10](#inc-10--restauration-backup) |

---

## INC-01 — Job stuck en `running`

**Symptôme**.  `GET /api/jobs/{job_id}` retourne `status=running`
depuis > 1 heure alors que le corpus tient en quelques minutes.

**Diagnostic**.

```bash
# 1. Le thread daemon existe-t-il encore ?
curl -s http://localhost:7860/api/jobs/{job_id} | jq '.status, .progress'

# 2. Les logs montrent-ils une activité récente ?
journalctl -u picarones -n 200 | grep "{job_id}"

# 3. Y a-t-il un appel cloud bloqué ?
ss -tnp | grep :443  # connexions TLS sortantes
```

Causes typiques :

- Appel cloud qui hang sans timeout (anciens adapters).
- Workspace en read-only (impossible d'écrire le résultat).
- Process daemon mort sans avoir mis à jour le statut.

**Mitigation**.

```bash
# Forcer l'annulation (dégrade en cancelled, pas en error).
curl -X DELETE http://localhost:7860/api/jobs/{job_id}

# Si le service ne répond plus :
systemctl restart picarones
# Au boot, le lifespan hook ``mark_orphaned_jobs_interrupted`` bascule
# automatiquement les jobs ``running`` en ``interrupted``.
```

**Suivi**.  Vérifier que le `JobRunner` n'a pas d'autres threads
zombies via `len(runner._threads)` (devrait redescendre).  Si
récurrent, instrumenter avec un timeout de soft-cap par job.

---

## INC-02 — Disk full sur le workspace

**Symptôme**.  Les jobs échouent en `error` avec
`OSError: [Errno 28] No space left on device`.  L'API web peut
elle-même planter au boot (`JobStore` ne peut plus persister).

**Diagnostic**.

```bash
df -h /var/lib/picarones/workspaces  # ou le path configuré
du -sh /var/lib/picarones/workspaces/*
```

Coupable typique : caches d'artefacts non purgés (`InMemoryArtifactStore`
n'a pas de TTL ; `FilesystemArtifactStore` non plus).

**Mitigation**.

```bash
# 1. Identifier les workspaces les plus gros.
du -sh /var/lib/picarones/workspaces/* | sort -rh | head -10

# 2. Purger les workspaces dont aucun job actif ne dépend (lookup
#    via JobStore).
sqlite3 /var/lib/picarones/jobs.db \
  "SELECT job_id, status, payload FROM jobs WHERE status NOT IN ('pending', 'running');" \
  | jq -r '.payload | fromjson | .output_dir'

# 3. Pour chaque output_dir terminé, archiver puis supprimer.
tar czf /backup/picarones-archive-$(date +%F).tar.gz <output_dirs>
rm -rf <output_dirs>
```

**Suivi**.  Établir une politique de rétention dans
[`data-retention-rgpd.md`](data-retention-rgpd.md).  Recommandation :
purger les workspaces > 30 jours sans accès.

---

## INC-03 — Cloud API rate limit

**Symptôme**.  Logs WARN : `[adapter] erreur retryable (tentative 3/4,
attente 8s) : 429 Too Many Requests`.  Job se termine en error après
épuisement des retries.

**Diagnostic**.

```bash
# Compter les 429 dans la dernière heure.
journalctl -u picarones --since "1 hour ago" \
  | grep "429" | wc -l

# Identifier les jobs concernés.
journalctl -u picarones --since "1 hour ago" \
  | grep -B2 "429" | grep "job_runner"
```

Causes typiques : un benchmark de 5000 documents lance 5000 appels
en parallèle, dépasse la quota de l'organisation cloud.

**Mitigation immédiate**.

```bash
# 1. Réduire le parallélisme du runner (env var).
sed -i 's/PICARONES_RUNNER_MAX_WORKERS=8/PICARONES_RUNNER_MAX_WORKERS=2/' /etc/picarones/.env
systemctl restart picarones

# 2. Re-soumettre les jobs en error qui se sont arrêtés au milieu.
# (Picarones ne fait pas de resume automatique sur erreur cloud — le
# cache d'artefacts du PipelineExecutor évite de re-exécuter les
# steps déjà terminés au prochain run.)
```

**Mitigation long terme**.  Demander une quota plus haute au
fournisseur cloud, ou ajouter un throttle au niveau adapter (token
bucket par adapter).

---

## INC-04 — SQLite `database is locked`

**Symptôme**.  Logs ERROR : `sqlite3.OperationalError: database is
locked`.  Touche typiquement le `JobStore`.

**Diagnostic**.

```bash
# 1. Compter les processes qui ont la DB ouverte.
lsof /var/lib/picarones/jobs.db

# 2. Vérifier le mode WAL.
sqlite3 /var/lib/picarones/jobs.db "PRAGMA journal_mode;"
# Devrait répondre "wal".  Si "delete" ou "rollback", le WAL n'a pas
# pris.
```

Causes : un process autre que Picarones a ouvert la DB (backup
maladroit), ou le filesystem ne supporte pas WAL (FAT32, NFS sans
verrous).

**Mitigation**.

```bash
# 1. Stopper l'autre process si identifié.
# 2. Si NFS : remonter avec ``-o nolock`` côté serveur ne marche PAS
#    (WAL exige des verrous).  Solution : déplacer ``jobs.db`` sur un
#    filesystem local et exporter le résultat via NFS read-only.
# 3. Si filesystem ne supporte vraiment pas WAL, le code retombe sur
#    ``rollback journal`` (cf. job_store.py:185-189) — fonctionnel
#    mais bloquant en lecture pendant les écritures.

# Test de santé.
sqlite3 /var/lib/picarones/jobs.db "PRAGMA integrity_check;"
```

**Suivi**.  Configurer un monitoring du `journal_mode` au boot.

---

## INC-05 — Memory leak

**Symptôme**.  RSS du process Picarones croît continûment au-delà
de 2 GB après plusieurs heures.

**Diagnostic**.

```bash
# Profiling minimal sans installer d'outil.
ps -o pid,rss,cmd -p $(pgrep picarones) | tail -1

# Si py-spy disponible :
py-spy dump --pid $(pgrep picarones)
```

Causes connues :

- `JobRunner._threads` non nettoyé (FIXÉ en S58).
- `RateLimitMiddleware._buckets` non borné (FIXÉ en S58 — eviction LRU).
- Caches d'artefacts in-memory accumulés (cf. INC-02).

**Mitigation**.

```bash
systemctl restart picarones
# Le lifespan hook nettoie les jobs orphelins ; les caches in-memory
# sont vidés par redémarrage.
```

**Suivi**.  Si récurrent, exporter `picarones._mem_audit` (à
implémenter — backlog) et corréler avec les jobs actifs.

---

## INC-06 — Compromission de clé API

**Symptôme**.  Facturation cloud anormale, ou notification du
fournisseur (« nous avons détecté une utilisation suspecte de votre
clé »).

**Mitigation immédiate** (dans l'ordre).

```bash
# 1. Révoquer la clé chez le fournisseur (console cloud).
# 2. Stopper Picarones pour éviter qu'il ne tente de relancer avec
#    la clé invalidée.
systemctl stop picarones
# 3. Rotater la clé dans le secret store.
vault kv put secret/picarones OPENAI_API_KEY=sk-NEW...
# 4. Reload + redémarrage.
systemctl start picarones
# 5. Audit des jobs récents pour identifier les exfiltrations.
sqlite3 /var/lib/picarones/jobs.db \
  "SELECT job_id, payload, created_at FROM jobs ORDER BY created_at DESC LIMIT 100;"
```

**Suivi**.  Notifier le DPO institutionnel sous 24 h si des
documents avec PII (registres, état civil) ont été envoyés à l'API
compromise.  Voir [`data-retention-rgpd.md`](data-retention-rgpd.md).

---

## INC-07 — Rapport HTML corrompu

**Symptôme**.  Deux runs identiques produisent des rapports HTML
différents byte-for-byte.

**Diagnostic**.

```bash
# Comparer les hashes de manifests.
sha256sum run-A/run_manifest.json run-B/run_manifest.json

# Si différents : un des paramètres canoniques a divergé.
diff <(jq -S . run-A/run_manifest.json) <(jq -S . run-B/run_manifest.json)
```

Causes typiques : un adapter cloud (gpt-4o, claude) qui a une
température > 0 → non-déterminisme natif.  Vérifier les
`adapter_kwargs` dans le manifest.

**Mitigation**.  Forcer `temperature: 0.0` dans la `RunSpec` YAML.
Pour les benchmarks de reproductibilité, exclure les adapters
non-déterministes.

---

## INC-08 — CI bloquée

**Symptôme**.  Un job GitHub Actions reste en `queued` ou
`in_progress` > 30 minutes pour ce qui devrait être un test < 5 min.

**Diagnostic**.  Vérifier dans cet ordre :

1. **Codecov upload hang** (déjà vu — 50+ min) → couvert par
   `timeout-minutes: 5` sur l'étape Codecov + `fail_ci_if_error: false`
   depuis le S59.
2. **Live tests qui s'exécutent** au lieu d'être deselected → le
   marker `live` doit être dans `addopts` de `pyproject.toml`
   (vérifié par les tests dual-lang).
3. **Codespaces / runner épuisé** → annuler manuellement le job,
   relancer.

**Mitigation**.  Annuler le workflow run (UI GitHub Actions),
relancer.  Si récurrent, élever un incident infra GitHub.

---

## INC-09 — Upgrade casse jobs

**Symptôme**.  Après `git pull && pip install -e .`, les jobs
soumis avant l'upgrade échouent en `error`.

**Diagnostic**.  Le `JobStore` utilise une table `schema_version` ;
une bump de SCHEMA_VERSION sans migration livre `JobStoreError` au
boot.

**Mitigation**.

```bash
# 1. Stopper le service AVANT l'upgrade.
systemctl stop picarones

# 2. Backup du JobStore.
cp /var/lib/picarones/jobs.db /var/lib/picarones/jobs.db.bak

# 3. Upgrade.
git pull && pip install -e ".[dev,web]"

# 4. Vérifier le schéma.
sqlite3 /var/lib/picarones/jobs.db "SELECT version FROM schema_version;"

# 5. Démarrer.  Le dispatcher applique automatiquement les
#    migrations enregistrées dans ``_MIGRATIONS``.
systemctl start picarones
```

**Suivi**.  Tester chaque upgrade en staging avant prod.

---

## INC-10 — Restauration depuis backup

**Symptôme**.  Corruption ou perte du workspace ou de la DB jobs.

**Pré-requis**.  Backup récent (recommandé : snapshot quotidien du
volume `/var/lib/picarones/`).

**Mitigation**.

```bash
# 1. Stopper le service.
systemctl stop picarones

# 2. Restaurer.
rsync -av /backup/picarones-2026-05-XX/ /var/lib/picarones/

# 3. Vérifier l'intégrité SQLite.
sqlite3 /var/lib/picarones/jobs.db "PRAGMA integrity_check;"

# 4. Démarrer.  Les jobs ``running`` au moment du backup seront
#    automatiquement marqués ``interrupted`` par le lifespan hook.
systemctl start picarones
```

**Suivi**.  Communiquer aux utilisateurs que les jobs en cours au
moment du backup sont à relancer.

---

## Escalade

Si un incident dépasse les procédures ci-dessus :

1. Documenter l'observation dans un fichier `incidents/<date>.md`
   (snapshot du symptôme + commandes lancées + résultat).
2. Ouvrir une issue GitHub avec le label `incident`.
3. Pour une vulnérabilité de sécurité, suivre la procédure de
   [`/SECURITY.md`](../../SECURITY.md) (canal privé).

## Révisions

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-05 | Création initiale (S60), 10 scénarios |
