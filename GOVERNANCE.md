# Gouvernance — Picarones

> Sprint A10 du plan de remédiation institutionnelle
> ([`docs/audits/remediation-plan-2026-05.md`](docs/audits/remediation-plan-2026-05.md)).
>
> Ce document explicite **comment Picarones est maintenu et fait
> évoluer** : qui décide, à quelle cadence, avec quels engagements de
> service. Il complète :
>
> - [`CONTRIBUTING.md`](CONTRIBUTING.md) — comment contribuer
>   techniquement (branches, tests, style) ;
> - [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — comportement attendu
>   dans les espaces du projet ;
> - [`SECURITY.md`](SECURITY.md) — signalement de vulnérabilités ;
> - [`.github/CODEOWNERS`](.github/CODEOWNERS) — qui est reviewer par
>   défaut sur quel chemin.

## Rôles

### BDFL / Maintainer principal

À ce stade du projet (mai 2026, ~3 600 tests, 1.x), **Picarones
est maintenu en BDFL** par
[@maribakulj](https://github.com/maribakulj). Toute décision finale
sur les contrats d'API publique, les choix éditoriaux (palette,
règle de neutralité du moteur narratif, conventions de
normalisation) et les releases lui incombe.

### Reviewers

Tout collaborateur qui a contribué au moins **3 PR mergées** en
6 mois peut être ajouté à `.github/CODEOWNERS` comme reviewer sur
les paths qu'il connaît bien. La promotion est sur invitation par
le BDFL après accord du contributeur.

### Domain experts

Pour les sujets qui exigent une expertise non-tech :

- **Paléographie / archivistique** : cas d'études, prompts médiévaux,
  glossaire historique, profils de normalisation diplomatique.
- **Accessibilité** : déclarations RGAA, audits manuels NVDA/VoiceOver,
  retours WCAG.
- **Linguistique computationnelle** : métriques NER, taxonomies
  d'erreurs, recherchabilité fuzzy.

Ces personnes sont nommées dans `CODEOWNERS` sur les chemins qui
les concernent et leur revue est **systématique** — pas pour bloquer
le merge mais pour garantir la qualité éditoriale.

### Évolution vers un comité de pilotage

Si Picarones atteint **> 5 mainteneurs actifs** ou **> 5 institutions
adoptantes documentées**, la gouvernance bascule en comité de pilotage
(steering committee) avec un vote à la majorité simple sur :

- ajout/retrait d'un mainteneur ;
- ruptures d'API publique (breaking changes au-delà d'un tag majeur) ;
- changement de licence ;
- évolution majeure de la philosophie éditoriale (neutralité du
  rapport, ouverture aux modules contribués externes).

Cette évolution sera entérinée par une PR sur ce fichier signée par
tous les mainteneurs actifs.

## Cadence de release

| Type | Cadence cible | Déclencheur |
|---|---|---|
| **Patch** (1.2.x) | À la demande | Bug fix, correctif sécurité, doc |
| **Mineure** (1.x.0) | Mensuelle | Sortie d'au moins un sprint d'audit ou d'un nouveau feature group |
| **Majeure** (x.0.0) | Trimestrielle | Breaking changes API publique cumulés |
| **Hotfix sécurité** | < 72 h | CVE critique, problème de confidentialité |

La cadence est un objectif, pas un engagement contractuel. Une
release mineure peut être reportée si le travail planifié n'est pas
prêt — la stabilité prime sur le calendrier.

Procédure release détaillée :
[`docs/operations/release-process.md`](docs/operations/release-process.md).

## SLO de réponse

| Type d'événement | SLO en jours ouvrés |
|---|---|
| Triage initial d'une issue | 5 |
| Réponse à une PR (1ère revue) | 7 |
| Triage initial d'une faille de sécurité (cf. SECURITY.md) | 2 |
| Correctif d'une faille HIGH/CRITICAL | 5 |
| Release patch après correctif sécurité | 3 |

Ces SLO sont **best-effort** : Picarones est maintenu sur du temps
disponible, sans garantie commerciale. Pour des SLO contractualisés,
contractualiser un support institutionnel via une convention de
prestation (cf. modalités à définir au cas par cas).

## Politique de breaking changes

L'API publique de Picarones est définie par
[`docs/api-stable.md`](docs/api-stable.md). Elle inclut :

- les symboles ré-exportés depuis `picarones/__init__.py` ;
- les commandes CLI `picarones X` documentées dans le README ;
- les endpoints HTTP listés dans le README ;
- la structure du JSON de résultats (`BenchmarkResult.as_dict()`).

**Garanties** :

- À l'intérieur d'un tag majeur (`1.x`), aucun de ces contrats ne
  change de manière incompatible.
- Une dépréciation est annoncée au moins **2 versions mineures**
  avant suppression, avec `DeprecationWarning` à l'appel et entrée
  dans le `CHANGELOG.md` en section *Deprecated*.
- Un breaking change exige un bump majeur (`1.x → 2.0`) et une
  section *BREAKING CHANGES* dans la release notes.

Hors API publique (modules internes, helpers privés `_*`) : aucune
garantie de stabilité — utiliser à vos risques.

## Procédure de transfert de mainteneur

Si le BDFL doit transférer la maintenance (départ, indisponibilité
prolongée, conflit d'intérêt apparu) :

1. **Annonce** : issue publique sur le repo + entrée CHANGELOG.
2. **Période de transition** (≥ 3 mois) : nouveau mainteneur en
   shadow review, le BDFL valide.
3. **Bascule** : transfert des droits GitHub (admin), update de
   `CODEOWNERS`, update du `CITATION.cff` (auteurs).
4. **Communication** : annonce à la communauté HuggingFace Space,
   PyPI, et institutions adoptantes connues.

Si aucun successeur n'est trouvé en 6 mois, le projet entre en
**mode archive** : tag `archived: 2026-XX`, dernière release de
sécurité, README enrichi d'un encart « projet en cherche d'un
nouveau mainteneur — fork bienvenu ».

## Conflicts of interest (Sprint A10, M-10)

Picarones benchmarke des fournisseurs cloud commerciaux (OpenAI,
Anthropic, Mistral, Google, Microsoft Azure). Pour qu'un papier ou
une étude qui s'appuie sur ses analyses Pareto coût soit citable
sans suspicion, nous déclarons :

### Affiliations des mainteneurs

Les mainteneurs déclarent leurs affiliations académiques et
industrielles dans la liste suivante (mise à jour à chaque ajout
de mainteneur dans `CODEOWNERS`) :

- **@maribakulj** : aucune affiliation salariée chez un fournisseur
  OCR/LLM/cloud benchmarké par Picarones. Aucune participation
  capitalistique dans une entreprise concurrente ou cliente d'un
  des fournisseurs.

### Indépendance des données de pricing

Les valeurs dans `picarones/data/pricing.yaml` :

- proviennent **exclusivement** des tarifs publics affichés sur les
  sites des fournisseurs (sans NDA, sans accord commercial) ;
- sont datées explicitement (`meta.last_updated`, `meta.valid_until`
  depuis Sprint A8) ;
- peuvent être surchargées par l'utilisateur via
  `ReportGenerator(..., pricing=...)` pour refléter ses propres
  tarifs négociés.

Aucun fournisseur ne finance, sponsorise ou influence le contenu
de cette table. Si un fournisseur souhaite corriger une valeur, il
doit ouvrir une PR publique avec source vérifiable.

### Indépendance éditoriale du moteur narratif

Le moteur narratif (Sprint 19+) émet des `Fact` traçables au JSON
d'entrée. Aucune logique privilégie un fournisseur sur un autre :

- les seuils des détecteurs sont éditoriaux (publics dans
  `core/facts.py`) et symétriques entre moteurs ;
- la philosophie « Picarones mesure et classe — il ne tranche pas »
  garantit l'absence de recommandation prescriptive.

Toute évolution de cette politique exige un PR signé par la
totalité des mainteneurs actifs.

### Sponsoring et financement

Si Picarones reçoit un jour un financement institutionnel ou
commercial (subvention, prestation, sponsorship), le donateur sera
listé en pied du `README.md` avec mention explicite « Picarones
n'est pas obligé envers ses sponsors et conserve son indépendance
éditoriale ». Toute restriction d'usage demandée par un sponsor
sera publiquement refusée et déclenchera l'annulation du sponsoring.

## Ressources

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — guide technique pour PR
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1
- [`SECURITY.md`](SECURITY.md) — signalement de vulnérabilités
- [`ACCESSIBILITY.md`](ACCESSIBILITY.md) — engagement WCAG 2.1 AA
- [`docs/operations/release-process.md`](docs/operations/release-process.md)
  — cycle de release

---

*Dernière mise à jour : 2 mai 2026 (Sprint A10).*
