# Rétention des données et conformité RGPD

> Ce document décrit **quelles données Picarones collecte**, **où
> elles sont stockées**, **combien de temps**, et **comment un
> usager peut demander leur suppression**.
>
> Cible : DPO et services juridiques institutionnels qui doivent
> intégrer Picarones à un registre de traitements RGPD.

## Données collectées

| Donnée | Origine | Stockage | Caractère personnel |
|---|---|---|---|
| Images uploadées (`uploads/`) | Upload utilisateur | Disque local | **Oui** si l'image identifie une personne (signature, registre paroissial nominatif, etc.) |
| Vérité terrain (.gt.txt) | Upload utilisateur | Disque local | Idem |
| Configuration de benchmark | Saisie utilisateur | BD `jobs.sqlite` | Indirect (préférences) |
| Job ID + statut + progress | Généré | BD `jobs.sqlite` | Non |
| Adresse IP du requérant | HTTP request | Mémoire `RateLimiter` | **Oui** (donnée personnelle au sens RGPD) |
| Logs applicatifs | Generated | stdout / fichier | Variable (peuvent contenir IP) |
| Rapports HTML générés | Sortie | Disque local `reports/` | Hérite des images du corpus |
| Historique longitudinal | Auto | BD `history.sqlite` | Non (statistiques agrégées) |

## Durées de rétention

### Par défaut

| Donnée | Durée | Variable d'env |
|---|---|---|
| Uploads (`uploads/<corpus_id>/`) | **7 jours** après dernier accès | `PICARONES_UPLOAD_RETENTION_DAYS=7` |
| IP dans rate-limiter | **24 heures** glissantes | `PICARONES_LOG_IP_RETENTION_HOURS=24` |
| Logs applicatifs | **30 jours** (rotation) | géré par syslog/journald |
| Jobs terminés | **90 jours** dans la BD | `PICARONES_JOBS_RETENTION_DAYS=90` |
| Rapports HTML | **Indéfini** par défaut (artefacts citables) | `PICARONES_REPORTS_RETENTION_DAYS=0` (0 = jamais) |
| Historique longitudinal | **Indéfini** (objectif analytique long terme) | — |

### Adapter à votre politique

Pour un déploiement avec exigences RGPD strictes (par exemple :
établissement scolaire, archive avec mineurs identifiables), ces
durées doivent être réduites et documentées dans votre registre
de traitements.

```bash
# Exemple : déploiement strict
export PICARONES_UPLOAD_RETENTION_DAYS=3
export PICARONES_LOG_IP_RETENTION_HOURS=4
export PICARONES_JOBS_RETENTION_DAYS=30
export PICARONES_REPORTS_RETENTION_DAYS=180
```

## Mécanismes de purge automatique

### Uploads anciens

Le module `picarones.interfaces.web.maintenance` exécute une tâche
asyncio en arrière-plan qui scanne `uploads/` toutes les 6 heures
et supprime les sous-dossiers dont :

- `mtime` > `PICARONES_UPLOAD_RETENTION_DAYS` jours, **ET**
- aucun job actif (status ∈ {running, queued}) ne référence le
  corpus.

Le mtime est rafraîchi à chaque nouvel accès (lecture, lancement
d'un benchmark dessus), donc un corpus utilisé reste tant qu'il
sert.

Les **logs** signalent chaque purge : `[maintenance] purged
upload <id> (last access: <date>)`.

### IP du rate-limiter

Le `RateLimiter` garde les compteurs en mémoire, fenêtre
glissante. Aucune persistance disque — les IP disparaissent au
redémarrage et au-delà de la fenêtre de rate limit.

### Jobs terminés en BD

Suppression manuelle via la commande CLI :

```bash
picarones jobs purge --older-than 90d
```

(À ajouter dans un sprint ultérieur si demande utilisateur — le
volume de la BD jobs reste modeste, < 1 MB pour 10 000 jobs.)

## Droit d'accès et de suppression

### Export RGPD

Un utilisateur peut demander l'**export de toutes les données le
concernant** :

```bash
# Pour un identifiant SSO connu (ex: x-remote-user)
picarones jobs export --user <identifiant> --output user_export.json
```

Le JSON exporté contient :

- la liste de ses jobs (création, statut, durée, métriques agrégées) ;
- les chemins (mais PAS le contenu) des uploads qu'il a soumis ;
- les rapports HTML qu'il a générés (chemins).

### Suppression / droit à l'oubli

Sur demande d'un utilisateur identifié :

```bash
picarones jobs delete --user <identifiant> --confirm
```

Effets :

- supprime tous les uploads qu'il a soumis ;
- anonymise ses jobs en BD (champ `created_by` mis à `null`,
  conservation des métriques agrégées qui ne l'identifient pas) ;
- supprime les rapports nominatifs.

L'**historique longitudinal** (statistiques agrégées sur
plusieurs runs) **n'est pas affecté** car il ne contient pas
d'identifiant personnel — seulement des moyennes et des deltas
de CER. Cette base est compatible RGPD car les données y sont
anonymisées par construction.

### Délai de réponse

Conformément à l'article 12 du RGPD, Picarones (via son
mainteneur institutionnel) répondra à toute demande d'accès,
rectification ou suppression dans un délai de **un mois maximum**.

## Sous-traitance : engines cloud

Si vous activez les engines cloud (Mistral OCR, Google Vision,
Azure DI) ou les LLMs (OpenAI, Anthropic, Mistral), **les images
uploadées sont transmises aux serveurs de ces fournisseurs** pour
traitement.

| Fournisseur | Localisation des serveurs | DPA disponible |
|---|---|---|
| OpenAI | États-Unis (zone de transfert atypique au sens UE) | Oui (Data Processing Addendum) |
| Anthropic | États-Unis | Oui |
| Mistral AI | France / UE | Oui (siège social France) |
| Google Cloud Vision | Multi-zones (paramétrable) | Oui (UE possible via Cloud Region EU) |
| Azure Document Intelligence | Multi-zones (paramétrable) | Oui (UE possible) |

**Implications** :

- Pour un corpus contenant des données personnelles, **privilégier
  les engines locaux** (Tesseract, Pero) qui ne sortent rien du
  serveur.
- Si vous devez utiliser un engine cloud, contractualiser le DPA
  avec le fournisseur **avant** d'envoyer le corpus.
- Mettre à jour votre registre de traitements en mentionnant
  explicitement le sous-traitant.

Le **mode public** (`PICARONES_PUBLIC_MODE=1`) refuse
automatiquement les engines cloud — utile pour un déploiement
public où les uploads ne sont pas filtrés.

## Notification de violation de données

En cas de fuite ou compromission affectant des données personnelles
hébergées par Picarones :

1. **T+0 à T+72 h** : notification à la CNIL (formulaire en ligne :
   <https://notifications.cnil.fr/notifications/index>).
2. **T+72 h à T+1 mois** : notification individuelle aux personnes
   concernées si « risque élevé » (RGPD art. 34).
3. **Audit forensique** : `SECURITY.md` documente la procédure de
   conservation des logs pour analyse post-incident.

Coordonnées du DPO institutionnel à inscrire dans votre déploiement
(variable d'env ou page web `/legal`) :

```bash
export PICARONES_DPO_CONTACT="dpo@institution.fr"
export PICARONES_DPO_PHONE="+33 X XX XX XX XX"
```

(Exposition dans une page legales prévue dans un sprint UX
dédié — actuellement à mettre dans le footer du reverse proxy.)

## Mention légale recommandée pour la home

À ajouter en footer de votre déploiement :

> Picarones traite les corpus que vous lui confiez pour les besoins
> du benchmarking d'outils OCR/HTR. Les uploads sont conservés
> [N] jours après dernier accès puis supprimés automatiquement.
> Les rapports générés sont conservés indéfiniment comme artefacts
> de référence. Vos données ne sont **jamais** transmises à des
> tiers sauf si vous activez explicitement un engine cloud — voir
> [politique de rétention](/legal/rgpd) et [contact DPO](/legal/dpo).

## Voies de recours

En cas de difficulté ou de désaccord :

- Contacter le DPO institutionnel (en priorité).
- Saisir la [CNIL](https://www.cnil.fr/fr/plaintes) (autorité
  française).
- Saisir l'autorité de contrôle de votre pays de l'UE.

## Tests automatisés

Le test `tests/web/test_upload_retention.py` valide
que :

- la purge auto efface les uploads > N jours ;
- la purge ne touche pas les uploads d'un job en cours ;
- la BD jobs reste cohérente après purge.

---

*Dernière mise à jour : 2 mai 2026.*
