# Écrire un module pour le banc d'essai

Ce tutoriel montre **par l'exemple** comment écrire un module
Picarones qui peut être chargé dans une pipeline composée, audité,
et inclus dans un rapport. Pour la **politique normative complète**
(contrat d'interface, métadonnées obligatoires, règles d'audit),
voir [`developer/module-policy.md`](../developer/module-policy.md).

---

## Cas d'usage

Vous avez écrit un script qui post-corrige du texte OCR avec une
heuristique métier (par exemple : règles de normalisation propres
à un fonds d'archives donné). Vous voulez le brancher dans
Picarones pour mesurer son apport vs un baseline.

C'est exactement le cas que cible l'axe B (banc d'essai de
pipelines composées).

---

## Module minimal

Un module Picarones est une **classe Python** qui hérite de
`BaseModule` et implémente `run(...)`.

```python
# my_corrector.py
from picarones.domain.module_protocol import BaseModule
from picarones.domain.artifacts import ArtifactType, Artifact


class MyCorrector(BaseModule):
    """Post-corrige le texte OCR avec une règle métier."""

    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.TEXT,)

    def run(self, artifact: Artifact) -> Artifact:
        text = artifact.payload
        # Votre logique métier ici.
        corrected = text.replace(" l'", " l'").replace("  ", " ")
        return Artifact(
            type=ArtifactType.TEXT,
            payload=corrected,
        )
```

Quatre points à retenir :

1. `input_types` et `output_types` doivent être déclarés au niveau
   classe (le planner les lit avant exécution).
2. `run` prend un `Artifact` et en retourne un. Pas d'effet de
   bord, pas de mutation.
3. Le type de sortie peut différer du type d'entrée (par exemple
   `IMAGE → TEXT` pour un OCR).
4. La classe ne doit rien savoir de Picarones au-delà de
   `BaseModule` — c'est du Python ordinaire.

---

## Manifeste

Pour être chargé, le module doit déclarer un manifeste avec
**5 champs obligatoires** :

```python
from picarones.domain.module_protocol import ModuleManifest

MANIFEST = ModuleManifest(
    name="my-corrector",
    version="0.1.0",
    author="Vous <vous@institution.fr>",
    license="MIT",
    description="Post-correction par règles métier.",
)
```

Le manifeste sert à tracer **qui** est responsable du module dans
le rapport et à versionner les comparaisons longitudinales.

---

## Audit

Avant exécution, le module passe un audit statique :

```python
from picarones.evaluation.metrics.module_policy import audit_module

issues = audit_module(MyCorrector, MANIFEST)
assert not issues, f"Module non conforme : {issues}"
```

Si l'audit échoue, le module n'est **pas chargé** dans la pipeline
— pas d'exception silencieuse en production. Les règles d'audit
sont énumérées dans
[`developer/module-policy.md`](../developer/module-policy.md).

---

## Brancher dans une pipeline

Une pipeline est décrite par un `PipelineSpec`. Le module est
référencé par son chemin Python :

```python
from picarones.domain.pipeline_spec import PipelineSpec, PipelineStep

spec = PipelineSpec(
    name="ocr-puis-correction",
    steps=[
        PipelineStep(
            name="ocr",
            module="picarones.adapters.ocr.tesseract:TesseractAdapter",
        ),
        PipelineStep(
            name="post-correction",
            module="my_corrector:MyCorrector",
        ),
    ],
)
```

Lancez le benchmark avec ce pipeline :

```bash
picarones run \
  --corpus mon_corpus/ \
  --pipeline ocr-puis-correction.yaml \
  --output rapport.html
```

Le rapport présente alors **la pipeline complète** comme un
« moteur » à part entière, comparable aux autres dans le tableau
récapitulatif et le diagramme CD.

---

## Étapes suivantes

- Politique normative et règles d'audit :
  [`developer/module-policy.md`](../developer/module-policy.md)
- Étendre le moteur narratif pour commenter votre module :
  [`developer/extending-i18n.md`](../developer/extending-i18n.md)
- Reproductibilité de la comparaison :
  [`reference/reproducibility-snapshots.md`](../reference/reproducibility-snapshots.md)
- Architecture en cercles (où se branche un module) :
  [`explanation/architecture.md`](../explanation/architecture.md)
