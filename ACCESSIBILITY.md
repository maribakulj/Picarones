# Déclaration d'accessibilité — Picarones

> Sprint A7 du plan de remédiation institutionnelle
> ([`docs/audits/remediation-plan-2026-05.md`](docs/audits/remediation-plan-2026-05.md)).
>
> **Statut au 2 mai 2026** : audit interne Sprints A6 + A7 validé,
> audit externe RGAA / WCAG **en cours** (planifié au Sprint A15).
> Cette déclaration est **provisoire** jusqu'à validation par un
> cabinet d'audit indépendant.

## Engagement

Picarones s'engage à rendre son interface web et le rapport HTML
qu'elle génère accessibles conformément à :

- **WCAG 2.1** (Web Content Accessibility Guidelines, W3C) niveau **AA**,
- **RGAA 4.1** (Référentiel général d'amélioration de l'accessibilité,
  France, art. 47 de la loi 2005-102),
- **EN 301 549** (norme européenne d'accessibilité des produits TIC).

## Périmètre couvert

- L'application web FastAPI servie sur ``picarones serve``
  (configuration, lancement de benchmarks, suivi en temps réel).
- Le rapport HTML interactif généré par ``picarones report``
  (tableau de classement, galerie, vue document, vues d'analyses,
  panneau Avancé, glossaire contextuel).

## État de conformité

### Critères WCAG 2.1 niveau A — **conforme** (Sprint A6)

| Critère | Statut | Mécanisme |
|---|---|---|
| 1.1.1 Non-text Content | ✓ | Tous les graphiques Chart.js ont `aria-label` + `<table>` jumelle accessible via bouton « Voir les données ». |
| 1.3.1 Info and Relationships | ✓ | `scope="col"` sur tous les `<th>` rendus. `<main>` avec `id` et `role`. |
| 2.1.1 Keyboard | ✓ | Toute interaction utilise `<button>`, `<a>`, `<input>` natifs — accessible au clavier. |
| 2.4.1 Bypass Blocks | ✓ | Skip-link `<a class="skip-link" href="#main">` premier enfant focusable du `<body>`. |
| 3.1.1 Language of Page | ✓ | `<html lang="fr|en">` dynamique selon le rendu. |
| 4.1.2 Name, Role, Value | ✓ | Contrôles natifs, labels ARIA appropriés. |

### Critères WCAG 2.1 niveau AA — **conforme** (Sprint A7)

| Critère | Statut | Mécanisme |
|---|---|---|
| 1.4.3 Contrast (Minimum) | ✓ | Palette par défaut Okabe-Ito, contraste ≥ 4,5:1 sur fond blanc et fond de tableaux. |
| 1.4.5 Images of Text | ✓ | Aucun texte rendu en image — tout est HTML/SVG sélectionnable. |
| 1.4.10 Reflow | ✓ | Layout responsive ; pas de scroll horizontal forcé jusqu'à 320 px de large. |
| 1.4.11 Non-text Contrast | ✓ | Bordures de boutons, focus rings ≥ 3:1 contre l'environnement. |
| 2.4.6 Headings and Labels | ✓ | Hiérarchie h1/h2/h3 respectée, labels descriptifs. |
| 3.1.2 Language of Parts | ✓ | Termes techniques marqués si la langue diffère du rapport. |
| 4.1.3 Status Messages | ✓ | Messages d'état (« Chargement… », « Aucune donnée ») dans des régions `role="status"` ou `aria-live`. |

### Critères WCAG 2.1 niveau AAA — **partiellement conforme**

Le niveau AAA est un objectif d'amélioration continue, non une exigence
légale. Sont actuellement non conformes par décision éditoriale :

| Critère | Décision | Justification |
|---|---|---|
| 1.4.6 Contrast (Enhanced) — 7:1 | Reporté | La palette Okabe-Ito tient 4,5:1 mais pas systématiquement 7:1. Un thème AAA dédié pourra être ajouté à la demande institutionnelle. |
| 2.3.2 Three Flashes | N/A | Aucun contenu clignotant produit par le rapport. |

## Mécanismes d'accessibilité notables

### Palette daltonien-friendly (par défaut)

Depuis Sprint A7, la palette par défaut est **Okabe-Ito 2008**
(palette qualitative recommandée pour la déficience de la vision
des couleurs : deutéranopie, protanopie, tritanopie). L'ancienne
palette rouge/vert/jaune reste accessible via :

- la case « Mode daltonien-friendly » du panneau « ⚙ Avancé »,
- le paramètre URL ``?palette=classic`` (état partageable).

### Tableau de données pour graphiques

Chaque graphique Chart.js du rapport propose un bouton « Voir les
données » qui révèle un `<table>` avec les valeurs sources. Cette
table est rendue à la demande (lazy) pour ne pas alourdir le DOM
initial mais reste annoncée par les lecteurs d'écran via
`aria-describedby`.

### Skip-to-content

Un lien « Aller au contenu » apparaît au focus en haut de la page
(Tab depuis la URL bar y mène en 1 tabulation) et permet d'éviter
la navigation principale et les bandeaux contextuels.

### Bilinguisme intégral

Le rapport est livré en français ou en anglais (paramètre
``--lang``). Les libellés a11y (skip-link, boutons, ARIA labels,
captions de tableaux jumeaux) sont localisés dans les deux langues.

## Dérogations connues et plan de remédiation

| Item | Statut | Échéance |
|---|---|---|
| Matrice de confusion Unicode (vue Caractères) | Densité visuelle élevée — la table jumelle accessible reste l'alternative principale. | Refonte UX prévue Sprint post-A14. |
| Génération PDF du rapport | Non livrée (cf. SPECS.md « Promesses non tenues »). | Pas de plan d'ajout, le HTML couvre les usages observés. |
| Audit RGAA externe | En cours, prestataire à contractualiser. | Sprint A15 (~sem. 11–12). |

## Tests automatisés

Le contrat d'accessibilité est verrouillé par deux suites de tests :

- ``tests/report/test_a11y_level_a.py`` — 13 cas, valident le niveau A
  bloquant (skip-link, canvas a11y, scope=col, i18n du Reset).
- ``tests/report/test_a11y_level_aa.py`` — 12+ cas, valident le niveau
  AA (palette par défaut, locale FR/EN, toggle daltonien, fallback
  i18n des messages d'erreur charts).

Toute régression sur ces tests bloque le merge en CI.

## Agents utilisateurs supportés

L'accessibilité est validée sur :

- **Lecteurs d'écran** : NVDA 2024.x (Windows), VoiceOver (macOS),
  TalkBack (Android). Test JAWS prévu lors de l'audit externe.
- **Navigateurs** : Firefox ESR, Chrome stable, Safari récent, Edge
  stable. Aucune dépendance à une fonctionnalité non standard.
- **Agrandissements** : confort visuel jusqu'à zoom 200 %.

## Voies de recours

Si une difficulté d'accès vous empêche d'utiliser Picarones, vous
pouvez :

1. ouvrir une issue GitHub étiquetée ``a11y`` :
   <https://github.com/maribakulj/Picarones/issues/new?labels=a11y>,
2. contacter le mainteneur référent accessibilité (canal défini dans
   ``GOVERNANCE.md`` après Sprint A10).

En cas de discrimination caractérisée, le défenseur des droits
(France) peut être saisi : <https://www.defenseurdesdroits.fr/>.

## Calendrier de réaudit

- Audit interne automatisé : à chaque PR via la CI.
- Audit interne manuel (NVDA / VoiceOver) : à chaque release majeure.
- **Audit externe RGAA / WCAG complet** : annuel, premier audit prévu
  Sprint A15 (mai 2026).

## Remerciements

La palette Okabe-Ito (Okabe & Ito, 2008,
<https://jfly.uni-koeln.de/color/>) est utilisée avec leur
permission de diffusion ouverte.

---

*Dernière mise à jour : 2 mai 2026 (Sprint A7).*
