"""``SearchView`` — vue canonique 3, Sprint A14-S16.

Troisième vue d'évaluation canonique : "quel pipeline maximise la
**recherchabilité plein-texte** ?".

Distinct de TextView et AltoView
--------------------------------
| Vue | Question | Métriques |
|---|---|---|
| TextView (S14) | meilleur texte final ? | CER, WER, MER, WIL |
| AltoView (S15) | meilleur ALTO exploitable ? | validity, line_count, word_box |
| SearchView (S16) | meilleur pour la recherche plein-texte ? | searchability_recall, numerical_seq |

Un même pipeline peut avoir un excellent CER (TextView) tout en
étant mauvais pour la recherche fuzzy (SearchView), si ses erreurs
se concentrent sur des noms propres ou des dates.  Et inversement,
un pipeline avec un CER médiocre peut donner une excellente
recherchabilité si les erreurs sont sur des caractères non-significatifs.

Cette divergence est précisément ce que le rapport BnF doit rendre
visible — c'est l'objet du document
``docs/views/comparing-views.md``.

Types acceptés
--------------
Comme TextView : RAW_TEXT, CORRECTED_TEXT, ALTO_XML, PAGE_XML,
CANONICAL_DOCUMENT.  La projection vers RAW_TEXT est appliquée
automatiquement par ``projections_by_source_type``.

Métriques par défaut
--------------------
- ``searchability_recall`` — fraction des tokens GT retrouvés à
  distance de Levenshtein ≤ 2 (proxy Elastic).
- ``numerical_sequence_preservation`` — fraction des années 4
  chiffres de la GT préservées strictement.

Toutes ∈ [0, 1] avec ``higher_is_better=True``.

higher_is_better
----------------
**Critique** : les métriques de cette vue sont des recall
(``higher_is_better=True``), à l'inverse de TextView dont les
métriques sont des erreurs (``higher_is_better=False``).  Le
rapport doit colorier les chiffres de SearchView dans le sens
opposé de ceux de TextView.
"""

from __future__ import annotations

from picarones.domain.artifacts import ArtifactType
from picarones.domain.evaluation_spec import EvaluationView
from picarones.domain.projection_spec import ProjectionSpec


#: Métriques calculées par défaut.
DEFAULT_SEARCH_METRICS: tuple[str, ...] = (
    "searchability_recall",
    "numerical_sequence_preservation",
)


#: Types acceptés.  Identique à TextView : tout ce qui peut être
#: projeté vers RAW_TEXT est éligible.
DEFAULT_SEARCH_CANDIDATE_TYPES: frozenset[ArtifactType] = frozenset({
    ArtifactType.RAW_TEXT,
    ArtifactType.CORRECTED_TEXT,
    ArtifactType.ALTO_XML,
    ArtifactType.PAGE_XML,
    ArtifactType.CANONICAL_DOCUMENT,
})


#: Mapping ``source_type → ProjectionSpec`` (identique à TextView).
DEFAULT_SEARCH_PROJECTIONS: dict[ArtifactType, ProjectionSpec] = {
    ArtifactType.ALTO_XML: ProjectionSpec(
        source_type=ArtifactType.ALTO_XML,
        target_type=ArtifactType.RAW_TEXT,
        projector_name="alto_to_text",
    ),
    ArtifactType.PAGE_XML: ProjectionSpec(
        source_type=ArtifactType.PAGE_XML,
        target_type=ArtifactType.RAW_TEXT,
        projector_name="page_to_text",
    ),
    ArtifactType.CANONICAL_DOCUMENT: ProjectionSpec(
        source_type=ArtifactType.CANONICAL_DOCUMENT,
        target_type=ArtifactType.RAW_TEXT,
        projector_name="canonical_to_text",
    ),
}


#: Dimensions explicitement non évaluées.
DEFAULT_SEARCH_IGNORED_DIMENSIONS: tuple[str, ...] = (
    # Qualité caractère par caractère : c'est TextView (S14).
    "char_level_accuracy",
    # Structure documentaire : c'est AltoView (S15).
    "geometry",
    "block_structure",
    "reading_order",
    # Sémantique (synonymes, paraphrases) : non évaluée par cette
    # vue, qui reste lexicale.
    "semantic_equivalence",
)


#: Avertissement par défaut.
DEFAULT_SEARCH_WARNINGS: tuple[str, ...] = (
    "Cette vue mesure la recherchabilité PLEIN-TEXTE (rappel "
    "fuzzy à distance de Levenshtein ≤ 2, années préservées).  "
    "Un pipeline avec un excellent CER peut être moyen ici si "
    "ses erreurs se concentrent sur les noms propres ou les "
    "dates.  Et inversement.  Lire ensemble TextView et SearchView "
    "pour juger un pipeline.",
    "Métriques higher_is_better=True (rappel) — le sens de "
    "coloration est OPPOSÉ à celui de TextView (qui mesure des "
    "erreurs, lower_is_better).",
)


def build_search_view(
    *,
    name: str = "searchability",
    description: str = (
        "Mesure la recherchabilité plein-texte d'un pipeline "
        "(rappel fuzzy + années préservées)."
    ),
    candidate_types: frozenset[ArtifactType] | None = None,
    metric_names: tuple[str, ...] | None = None,
    normalization_profile: str | None = None,
    extra_warnings: tuple[str, ...] = (),
    extra_ignored_dimensions: tuple[str, ...] = (),
) -> EvaluationView:
    """Construit la vue canonique SearchView."""
    return EvaluationView(
        name=name,
        description=description,
        candidate_types=(
            candidate_types if candidate_types is not None
            else DEFAULT_SEARCH_CANDIDATE_TYPES
        ),
        projection=None,
        projections_by_source_type=DEFAULT_SEARCH_PROJECTIONS,
        normalization_profile=normalization_profile,
        metric_names=(
            metric_names if metric_names is not None
            else DEFAULT_SEARCH_METRICS
        ),
        warnings=DEFAULT_SEARCH_WARNINGS + extra_warnings,
        ignored_dimensions=DEFAULT_SEARCH_IGNORED_DIMENSIONS + extra_ignored_dimensions,
    )


__all__ = [
    "build_search_view",
    "DEFAULT_SEARCH_METRICS",
    "DEFAULT_SEARCH_CANDIDATE_TYPES",
    "DEFAULT_SEARCH_PROJECTIONS",
    "DEFAULT_SEARCH_IGNORED_DIMENSIONS",
    "DEFAULT_SEARCH_WARNINGS",
]
