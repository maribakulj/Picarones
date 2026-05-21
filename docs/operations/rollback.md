# Procédure de rollback

> Guide opérationnel pour rétrograder une version Picarones en
> production institutionnelle.

Audience : équipe ops BnF / autres institutions.

## Vue d'ensemble

Un rollback Picarones touche **trois couches** indépendantes :

1. **Code applicatif** — l'image Docker / le wheel installé.
2. **Schéma SQLite des jobs** — versionné dans la table
   `schema_version`.
3. **Configuration** — variables d'environnement.

La règle d'or : **rollback dans l'ordre inverse de la migration**.
Code → schéma → config.

## 1. Rollback du code applicatif

### Cas A : déploiement Docker

```bash
# Identifier la version actuelle
docker inspect picarones | grep Version

# Rollback vers la version précédente (ex : v2.0.0 → v1.5.3)
docker pull ghcr.io/maribakulj/picarones:v1.5.3
docker stop picarones
docker rm picarones
docker run --name picarones \
    -p 7860:7860 \
    --env-file /etc/picarones/.env \
    -v /var/lib/picarones/jobs.db:/app/jobs.db \
    ghcr.io/maribakulj/picarones:v1.5.3

# Vérifier
curl -fsS http://localhost:7860/health
```

### Cas B : install pip

```bash
# Lister les versions disponibles
pip index versions picarones

# Rollback
pip install --force-reinstall picarones==1.5.3

# Vérifier
picarones --version
```

### Cas C : Kubernetes

```bash
# Rollback la dernière révision du deployment
kubectl rollout undo deployment/picarones -n picarones-prod

# Ou cibler une révision spécifique
kubectl rollout history deployment/picarones -n picarones-prod
kubectl rollout undo deployment/picarones --to-revision=42

# Vérifier
kubectl rollout status deployment/picarones
kubectl get pods -n picarones-prod
```

## 2. Rollback du schéma SQLite

`JobStore` (`picarones/adapters/storage/job_store.py`) versionne
son schéma dans la table `schema_version`.  Si le rollback du code
nécessite de revenir à un schéma antérieur :

### Cas simple : downgrade non destructif

Si la migration N→N+1 a uniquement **ajouté** des colonnes / tables
(jamais supprimé), le code v1.5.3 lit un schéma v2 sans plantage —
les nouvelles colonnes sont simplement ignorées.

Aucune action requise.

### Cas complexe : downgrade destructif

Si la migration a renommé/supprimé des colonnes, le code v1.5.3
peut lever des `sqlite3.OperationalError`.

**Procédure** :

1. Stopper Picarones.
2. **Sauvegarder la DB** :

   ```bash
   cp /var/lib/picarones/jobs.db \
      /var/lib/picarones/jobs.db.before-rollback-$(date +%Y%m%d-%H%M%S)
   ```

3. Restaurer un dump pré-migration depuis la sauvegarde quotidienne :

   ```bash
   # Localisation typique de la backup
   ls /var/backups/picarones/

   # Restaurer la veille de la migration
   sqlite3 /var/lib/picarones/jobs.db ".restore /var/backups/picarones/jobs.db.YYYY-MM-DD"
   ```

4. **Forcer le schema_version à la valeur attendue par le code
   v1.5.3** (si nécessaire) :

   ```sql
   sqlite3 /var/lib/picarones/jobs.db
   sqlite> UPDATE schema_version SET version = 1;
   sqlite> .quit
   ```

5. Redémarrer Picarones v1.5.3.
6. Vérifier que les jobs anciens sont lisibles via `GET /api/jobs`.

### Pas de backup ?

Le code v2.0+ a un mode "lecture seule défensive" : si le
`schema_version` est plus récent que celui attendu, le code log
un warning explicite et tente de continuer.  En cas d'incident,
les jobs *nouveaux* fonctionnent ; les jobs *anciens* (avec des
champs nouveaux) peuvent apparaître tronqués mais ne crashent
pas.

## 3. Rollback de la configuration

Les variables d'environnement sont versionnées **hors du code**
(secret manager + `.env` non committé).  Pour rollback :

```bash
# Si la config a été modifiée en même temps que la release :
# remettre le snapshot précédent du .env

cp /etc/picarones/.env.before-deploy /etc/picarones/.env

# Recharger
docker compose restart  # ou systemctl restart picarones
```

## 4. Vérifications post-rollback

Checklist obligatoire :

- [ ] `curl /health` retourne 200.
- [ ] `curl /version` retourne la version cible.
- [ ] `GET /api/jobs` liste les jobs sans erreur.
- [ ] Les logs ne montrent pas de `sqlite3.OperationalError`
  ou de `KeyError` sur des colonnes manquantes.
- [ ] Soumission d'un nouveau benchmark via UI fonctionne.
- [ ] CSRF token (si mode institutionnel) : `GET /api/csrf/token`
  retourne 200, et un POST avec ce token passe le middleware.
- [ ] Logs en JSON (si `PICARONES_LOG_FORMAT=json`) parsables
  par l'ingester.

## 5. Cas d'urgence : rollback total

Si le déploiement est compromis (sécurité, données corrompues) :

```bash
# 1. Couper le trafic au reverse-proxy
nginx -s stop  # ou : kubectl scale deployment/picarones --replicas=0

# 2. Snapshot disque complet
tar czf /var/backups/picarones-emergency-$(date +%s).tar.gz \
    /var/lib/picarones/

# 3. Restaurer la dernière version stable connue + DB de la veille
# (suivre §1 Cas A/B/C + §2 ci-dessus)

# 4. Notifier les utilisateurs (SSO bandeau, email)
```

## 6. Pas de rollback automatique

Picarones **n'inclut pas** de rollback automatique sur erreur
post-déploiement.  La discipline est :

- Déployer en horaire de faible trafic.
- Surveiller les logs pendant 30 min après déploiement.
- Avoir la procédure ci-dessus prête (testée régulièrement en
  préprod).
- En cas de doute, rollback rapide vaut mieux que diagnostic
  prolongé.

## Voir aussi

- [`runbook.md`](runbook.md) — incidents courants en production.
- [`release-process.md`](release-process.md) — la procédure de
  forward-déploiement.
- [`data-retention-rgpd.md`](data-retention-rgpd.md) — gestion des
  données utilisateur lors des restaurations.
