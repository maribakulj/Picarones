# API Reference (auto-générée)

> **Audience** : développeur tiers, contributeur, mainteneur.  Cette
> référence est **générée automatiquement** depuis les docstrings du
> code par [mkdocstrings](https://mkdocstrings.github.io/), au build
> du site de documentation.
>
> Pour la **politique de stabilité** de l'API publique (semver,
> deprecation periods, symboles cibles), voir
> [`../reference/api-stable.md`](../reference/api-stable.md).
>
> Pour l'**architecture** et le **pourquoi** des choix de design,
> voir [`../explanation/architecture.md`](../explanation/architecture.md).

## Build local

```bash
pip install -e ".[docs]"
mkdocs serve  # hot-reload sur http://localhost:8000
```

ou

```bash
mkdocs build  # site statique dans site/
```

## Structure

L'API publique est groupée par cercle architectural :

| Cercle | Référence |
|--------|-----------|
| Domain (types purs) | [`domain.md`](domain.md) |
| Pipeline (orchestration) | [`pipeline.md`](pipeline.md) |
| Evaluation (métriques + vues) | [`evaluation.md`](evaluation.md) |
| Adapters (OCR/LLM/VLM) | [`adapters.md`](adapters.md) |
| App services (orchestrateur, jobs) | [`app.md`](app.md) |

## Stabilité

Tous les symboles documentés ici sont de l'**API publique** ce qui
signifie :

- Suivent semver — un retrait nécessite une release majeure et une
  deprecation period d'au moins une release mineure (`DeprecationWarning`
  émis depuis la version N, suppression en N+2 majeure).
- Sont vérifiés par `tests/core/test_public_api_signatures.py`.

Les symboles **privés** (préfixe `_` ou non listés dans `__all__`)
peuvent changer sans préavis.
