# Politique de modules contribués

> Sprint 97 — B.6 du plan d'évolution 2026.

Ce document décrit le **cadre de qualité** applicable à tout module
qu'un utilisateur tiers veut faire évaluer dans une pipeline composée
Picarones (axe B). Picarones étant un **banc d'essai**, le code des
modules reste chez l'utilisateur — la plateforme se contente de les
charger, les exécuter, et les évaluer. La politique ci-dessous décrit
les **garanties d'interface** et les **métadonnées obligatoires** pour
qu'un module soit acceptable.

## TL;DR

Pour qu'un module soit acceptable :

1. Il **hérite** de `picarones.domain.module_protocol.BaseModule` (Sprint 33).
2. Il déclare ses `input_types` et `output_types` (parmi
   `ArtifactType.{IMAGE, TEXT, ALTO, PAGE, ENTITIES, READING_ORDER}`).
3. Il fournit un `ModuleManifest` avec **5 champs obligatoires** :
   `name`, `version`, `author`, `license`, `description`.
4. Il passe `audit_module(MyClass, manifest)` (cf.
   `picarones.evaluation.metrics.module_policy`).

Un module qui ne passe pas l'audit n'est **pas exécutable**. Pas
exécutable = pas dans la pipeline, pas dans le rapport.

## Pourquoi cette politique

Avec l'arrivée de l'axe B (banc d'essai de pipelines composées),
Picarones évalue des modules tiers que l'institution amène : LLM
correcteurs, reconstructeurs ALTO, classifieurs d'entités, mappeurs
divers. Sans cadre, ce mode d'usage est ingérable :

- on ne sait pas qui a écrit quoi ;
- on ne sait pas sous quelle licence le module est distribué ;
- on ne peut pas reproduire un benchmark si la version a changé ;
- on ne peut pas citer le module dans un papier scientifique ;
- l'interface peut casser silencieusement et personne ne s'en aperçoit.

La politique de modules adresse ces 5 points avec un manifest
structuré et un audit automatique au chargement.

## Manifest — champs obligatoires

```python
from picarones.evaluation.metrics.module_policy import ModuleManifest

manifest = ModuleManifest(
    name="my-llm-correcteur",
    version="1.2.0",
    author="alice@institution.fr",
    license="MIT",
    description="Correcteur LLM pour textes médiévaux français.",
    input_types=["text"],
    output_types=["text"],
    citation=(
        "Alice Dupont, 2025. A Domain-Adapted LLM Corrector for "
        "Medieval French. arXiv:2511.12345."
    ),
    homepage="https://github.com/alice/my-llm-correcteur",
    picarones_min_version="1.2.0",
)
```

| Champ | Obligatoire | Description |
|---|---|---|
| `name` | oui | Identifiant unique (slug). |
| `version` | oui | Version sémantique (`1.2.0`). |
| `author` | oui | Auteur ou institution responsable. |
| `license` | oui | SPDX recommandé (`MIT`, `Apache-2.0`, `GPL-3.0-or-later`…), non validé techniquement. |
| `description` | oui | Description courte (≤ 1 phrase). |
| `input_types` | oui | Liste des types d'entrée acceptés. |
| `output_types` | oui | Liste des types de sortie produits. |
| `citation` | non | Citation académique (BibTeX, DOI, ou texte libre). |
| `homepage` | non | URL du dépôt ou de la page projet. |
| `picarones_min_version` | non | Version minimale de Picarones requise. |
| `extra` | non | Métadonnées libres. |

## Contrat `BaseModule`

Tout module exécutable hérite de
`picarones.domain.module_protocol.BaseModule` (Sprint 33). Le contrat minimal
est :

```python
from picarones.domain.artifacts import ArtifactType
from picarones.domain.module_protocol import BaseModule

class MyLlmCorrecteur(BaseModule):
    name = "my-llm-correcteur"
    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.TEXT,)
    execution_mode = "io"  # "io" pour API, "cpu" pour local

    def process(self, inputs):
        text = inputs[ArtifactType.TEXT]
        # ... votre logique ...
        return {ArtifactType.TEXT: corrected_text}
```

## Audit automatique

```python
from picarones.evaluation.metrics.module_policy import audit_module

result = audit_module(MyLlmCorrecteur, manifest)
if not result.passed:
    for check in result.checks:
        if not check.passed:
            print(f"FAIL: {check.name} | {check.detail}")
    raise SystemExit("module rejeté par l'audit")
```

L'audit vérifie :

1. **Manifest valide** — 5 champs obligatoires non vides + `input_types`/`output_types` non vides.
2. **Héritage** — la classe hérite de `BaseModule`.
3. **Cohérence I/O** — `module.input_types` et `module.output_types`
   correspondent (case-insensitive) à ceux du manifest.
4. **Méthode `process`** — callable sur la classe.

## Stratégie d'ouverture en deux temps

Picarones est aujourd'hui en **phase fermée** : seuls les modules
officiels (ceux dans `picarones/adapters/ocr/`, `picarones/adapters/llm/`,
`picarones/pipeline/`) sont garantis stables. Les modules contribués
externes sont acceptés via PR sur le repo principal après audit.

La **phase ouverte** sera déclenchée quand 5–6 modules officiels
stables et un guide de contribution éprouvé seront disponibles.
L'ouverture passera alors par des plugins PyPI (`picarones-module-X`)
avec mécanisme `entry_points`. **Le manifest et l'audit décrits ici
sont la fondation de cette phase ouverte** — tout plugin externe devra
les fournir.

## Vue HTML « Modules audités »

Quand une pipeline composée est exécutée, le rapport HTML expose un
bloc **« Modules audités »** (cf.
`picarones.reports.html.module_audit_render.build_module_audit_html`) qui
liste pour chaque module :

- statut d'audit (✓ vert ou ✗ rouge avec compte des checks échoués) ;
- métadonnées (version, auteur, licence) ;
- types d'entrée/sortie ;
- citation académique (si fournie) ;
- page projet (si fournie).

Le module ne juge pas la qualité scientifique du module — il **expose
les métadonnées que le contributeur a déclarées**, pour permettre la
reproductibilité et la citation correcte dans les publications
dérivées.
