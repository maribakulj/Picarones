# Modèle d'Accord de Sous-Traitance (DPA)

> **Audience** : Délégué à la Protection des Données (DPO) de
> l'institution déployant Picarones, équipe juridique de cette même
> institution, mainteneur du projet.
>
> **Statut** : modèle de référence — à adapter et à signer entre
> l'institution (responsable de traitement) et chaque sous-traitant
> activé via les adapters cloud.  Ce document **n'est pas un contrat
> en lui-même** ; il définit les clauses minimales à inclure.
>
> **Référence légale** : Article 28 du Règlement (UE) 2016/679 (RGPD),
> [version consolidée](https://eur-lex.europa.eu/eli/reg/2016/679/oj).

## Pourquoi un DPA ?

Lorsqu'une institution patrimoniale (BnF, LoC, BL) déploie Picarones
en activant des adapters cloud (Mistral OCR, OpenAI, Anthropic,
Google Vision, Azure Document Intelligence), elle envoie des
documents qui peuvent contenir des **données à caractère personnel**
(PII) — typiquement :

- Registres d'état civil (naissances, mariages, décès).
- Recensements (noms, adresses, professions).
- Correspondance personnelle (lettres privées, journaux).
- Notes manuscrites avec mentions nominatives.

L'envoi de ces données à un tiers (le fournisseur cloud) constitue
une **sous-traitance** au sens RGPD §28 ; un accord écrit (DPA) est
**obligatoire** entre l'institution (responsable de traitement) et
chaque sous-traitant.

## Périmètre

Ce modèle couvre la sous-traitance des opérations de transcription
OCR/HTR effectuées par des services cloud activés par l'institution
via Picarones.  **Il ne couvre pas** :

- Le déploiement Picarones lui-même (l'institution est seule
  responsable de l'instance).
- Les adapters locaux (Tesseract, Pero OCR, Ollama) qui n'envoient
  rien à l'extérieur.

## Clauses minimales (RGPD §28.3)

### 1. Objet et durée du traitement

Transcription automatique de documents numérisés via OCR, HTR ou VLM
cloud, pour la durée du marché entre l'institution et le fournisseur.

### 2. Nature et finalité du traitement

- **Nature** : envoi d'images de documents et/ou de fragments de
  texte ; réception de transcriptions textuelles ou de descriptions
  structurées (ALTO, JSON canonique).
- **Finalité** : fournir à l'institution un benchmark comparatif de
  pipelines OCR/HTR sur son corpus, dans le cadre d'une évaluation
  technique préalable à un déploiement de production.

### 3. Type de données à caractère personnel

Selon le corpus envoyé.  L'institution **doit identifier en amont**
si le corpus contient :

- Données nominatives (noms, prénoms, dates de naissance/décès…).
- Données sensibles au sens RGPD §9 (origine raciale ou ethnique,
  opinions politiques, convictions religieuses, données de santé,
  orientation sexuelle…).

Pour les corpus sensibles, l'institution **doit privilégier les
adapters locaux** (Tesseract, Pero OCR, Ollama) ou anonymiser le
corpus avant envoi.

### 4. Catégories de personnes concernées

- Personnes citées dans les documents historiques (typiquement
  défuntes, sauf mention contraire).
- Auteurs ou correspondants des documents.

### 5. Obligations du sous-traitant

Le sous-traitant cloud s'engage à :

a) ne traiter les données que sur **instruction documentée** du
   responsable (l'institution).  Pas de réutilisation pour
   entraînement de modèles, sauf consentement explicite (cf. §10).

b) garantir que les **personnes autorisées** à traiter les données
   sont soumises à une obligation de confidentialité.

c) mettre en œuvre les **mesures de sécurité** énumérées au RGPD
   §32 (chiffrement en transit, contrôle d'accès, journalisation,
   tests réguliers).

d) ne pas recourir à un **autre sous-traitant** sans autorisation
   écrite préalable et spécifique du responsable.

e) **assister** le responsable dans la réponse aux demandes
   d'exercice de droits (accès, rectification, effacement…) et dans
   les obligations de notification de violations.

f) **supprimer ou retourner** les données à la fin de la prestation,
   sauf obligation légale de conservation.

g) mettre à disposition du responsable toutes les **informations
   nécessaires** pour démontrer la conformité au §28.

### 6. Localisation des traitements

L'institution **doit privilégier** les fournisseurs offrant un
hébergement et un traitement strictement dans l'Espace économique
européen (EEE).

| Adapter | Localisation par défaut | Disponibilité EEE |
|---------|------------------------|-------------------|
| Mistral OCR / chat | France (cf. [Mistral Trust](https://mistral.ai/security/)) | Oui |
| OpenAI | États-Unis | EU residency dispo via Enterprise |
| Anthropic Claude | États-Unis | EU residency limitée |
| Google Vision | Multi-régions | EEE configurable |
| Azure Document Intelligence | Multi-régions | EEE configurable |

Pour un transfert hors EEE, **clauses contractuelles types** (CCT)
2021/914/UE applicables OBLIGATOIRES.

### 7. Sécurité

Mesures minimales :

- Chiffrement TLS 1.2+ en transit.
- Pas d'enregistrement des prompts/réponses pour entraînement
  (option à activer côté fournisseur, cf. §10).
- Logs d'accès conservés < 30 jours sauf incident de sécurité.
- Tests de pénétration au moins annuels (à charge du sous-traitant).

### 8. Sous-sous-traitance

Liste des sous-sous-traitants autorisés à fournir au démarrage et à
chaque modification.  L'institution dispose d'un droit d'objection
à toute nouvelle sous-sous-traitance.

### 9. Audit

L'institution se réserve le droit, à ses frais et avec préavis
raisonnable (30 jours), de conduire un audit du sous-traitant ou de
mandater un tiers indépendant pour vérifier la conformité des
mesures techniques et organisationnelles.

### 10. Réutilisation pour entraînement de modèles

**Disposition critique** pour le patrimoine numérique : les
documents envoyés sont la propriété intellectuelle de l'institution
(et parfois du domaine public) ; les fournisseurs ne doivent **PAS**
les utiliser pour entraîner leurs modèles sans accord écrit.

Configuration recommandée par fournisseur :

| Fournisseur | Comment opt-out |
|-------------|------------------|
| OpenAI | Compte Enterprise ou via API avec `data_retention=zero` |
| Anthropic | Compte Enterprise ; pas d'option opt-out sur API standard |
| Mistral | API Enterprise tier ; opt-out par défaut sur certains plans |
| Google Vision | Activer Workspace Data Loss Prevention |
| Azure | Activer "Customer-Managed Keys" + opt-out training |

### 11. Notification de violation

Le sous-traitant s'engage à notifier l'institution **dans les 24
heures** de la connaissance d'une violation de données à caractère
personnel les concernant, par e-mail ET courrier signé.

### 12. Effacement à fin de prestation

À la fin du marché ou à la résiliation, le sous-traitant restitue
ou supprime toutes les données dans un délai de 30 jours, et
fournit une **attestation de destruction**.

## Annexes

### Annexe 1 — Description du traitement

À compléter par l'institution :

- [ ] Nom du corpus traité
- [ ] Volume estimé (nombre de documents, taille en GB)
- [ ] Période de traitement (du / au)
- [ ] Liste des adapters cloud activés
- [ ] Volume de PII estimé dans le corpus

### Annexe 2 — Mesures de sécurité

À compléter par le sous-traitant — référence :
[ANSSI Référentiel Général de Sécurité](https://www.ssi.gouv.fr/).

### Annexe 3 — Liste des sous-sous-traitants autorisés

À compléter par le sous-traitant.

## Procédure de signature

1. L'institution remplit les annexes en fonction du corpus prévu.
2. Le DPO de l'institution valide la liste des adapters cloud
   activés (`AdapterRegistry`).
3. Le contrat est signé par les deux parties (institution +
   fournisseur cloud) AVANT activation de l'adapter en production.
4. Une copie est conservée dans le dossier de conformité du
   traitement (durée minimale : 5 ans après la fin du traitement).

## Référence légale

- [Règlement (UE) 2016/679 — RGPD](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
- [Lignes directrices CEPD sur les sous-traitants](https://edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-072020-concepts-controller-and-processor-gdpr_fr)
- [Décision d'adéquation EU-US Data Privacy Framework (2023)](https://commission.europa.eu/document/fa09cbad-dd7d-4684-ace5-c1e932f3eda7_en)

## Révisions

| Version | Date | Changements |
|---------|------|-------------|
| 1.0 | 2026-05 | Création initiale (S60), modèle aligné RGPD §28 |
