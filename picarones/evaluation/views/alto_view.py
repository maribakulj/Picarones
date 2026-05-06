"""``AltoView`` — vue canonique 2, Sprint A14-S15.

Deuxième vue d'évaluation canonique : "quel pipeline produit le
meilleur ALTO exploitable ?".

Distinct de ``TextView`` (S14)
------------------------------
``TextView`` projette tout vers texte plat et ignore la structure
documentaire.  ``AltoView`` fait l'inverse : exige un ``ALTO_XML``
en entrée et mesure la **fidélité structurelle** (validité,
nombre de lignes, présence des bbox de mots, etc.).

Un même pipeline peut être évalué dans les deux vues.  Le rapport
HTML (S22) présentera les deux côte-à-côte pour qu'un lecteur
comprenne *pourquoi* deux pipelines avec le même CER produisent
des ALTO de qualités différentes.

Pipelines omis explicitement
----------------------------
Un pipeline qui ne produit pas d'``ALTO_XML`` (exemple : Tesseract
texte brut sans ALTO) ne peut pas être évalué dans ``AltoView``.
Le caller (typiquement un service applicatif au S19) doit
**omettre** ce pipeline du résultat ``AltoView`` plutôt que de lui
attribuer un score factice à 0.

Le pattern est démontré dans le test
``tests/evaluation/views/test_sprint_a14_s15_alto_view.py`` :
le caller boucle sur ``[TextView, AltoView]`` et pour chaque vue
filtre les pipelines dont l'artefact n'est pas dans
``view.candidate_types``.

Métriques par défaut
--------------------
- ``alto_validity`` — l'hypothèse est-elle structurellement
  cohérente ? (≥ 1 page, ≥ 1 bloc, ≥ 1 ligne).
- ``alto_line_count_ratio`` — ratio min/max du nombre de lignes.
- ``alto_word_box_coverage`` — fraction des mots qui ont une bbox.

Toutes ∈ [0, 1] avec ``higher_is_better=True``.

Reportées à un sprint suivant
-----------------------------
- ``textline_alignment`` (IoU des bbox de lignes).
- ``reading_order_consistency`` (Kendall tau sur les IDs).
- ``layout_f1`` (ICDAR 2015) via wrapper de
  ``evaluation/metrics/layout.py``.
"""

from __future__ import annotations

from picarones.domain.artifacts import ArtifactType
from picarones.domain.evaluation_spec import EvaluationView


#: Métriques calculées par défaut.  Toutes typées
#: ``(ALTO_XML, ALTO_XML)``.
DEFAULT_ALTO_METRICS: tuple[str, ...] = (
    "alto_validity",
    "alto_line_count_ratio",
    "alto_word_box_coverage",
)


#: Types acceptés.  Volontairement strict : seul ``ALTO_XML``
#: passe.  PAGE_XML pourrait être ajouté via une projection
#: ``page_to_alto`` (post-livraison) si le besoin se présente.
DEFAULT_ALTO_CANDIDATE_TYPES: frozenset[ArtifactType] = frozenset({
    ArtifactType.ALTO_XML,
})


#: Dimensions explicitement non évaluées.
DEFAULT_ALTO_IGNORED_DIMENSIONS: tuple[str, ...] = (
    # Qualité linguistique pure : c'est TextView (S14) qui la mesure.
    "linguistic_quality",
    # Recherchabilité fuzzy : c'est SearchView (S16).
    "search_recall",
    # Hallucinations contenu : c'est HallucinationView (post-S18).
    "content_hallucination",
)


#: Avertissement par défaut affiché en tête du bloc AltoView.
DEFAULT_ALTO_WARNINGS: tuple[str, ...] = (
    "Cette vue mesure la fidélité STRUCTURELLE de l'ALTO produit "
    "(validité, nombre de lignes, bbox).  La qualité TEXTUELLE de "
    "ce qui est dans cet ALTO est mesurée par TextView ; les deux "
    "doivent être lues ensemble pour juger un pipeline.",
    "Les pipelines qui ne produisent pas d'ALTO sont OMIS de cette "
    "vue.  Aucun score factice n'est attribué à un pipeline absent.",
)


def build_alto_view(
    *,
    name: str = "alto_documentary",
    description: str = (
        "Mesure la fidélité structurelle de l'ALTO produit par un "
        "pipeline (validité, lignes, bbox)."
    ),
    metric_names: tuple[str, ...] | None = None,
    extra_warnings: tuple[str, ...] = (),
    extra_ignored_dimensions: tuple[str, ...] = (),
) -> EvaluationView:
    """Construit la vue canonique AltoView.

    Pas de ``candidate_types`` paramétrable (la vue exige par
    nature ALTO_XML uniquement) ni de ``projection``
    (l'évaluation se fait sur l'ALTO tel quel, pas sur sa
    projection).

    Le caller qui veut une vue plus stricte (par exemple "exiger
    aussi le mapping vers une GT_ALTO précise") peut composer
    plusieurs ``AltoView`` paramétrées.
    """
    return EvaluationView(
        name=name,
        description=description,
        candidate_types=DEFAULT_ALTO_CANDIDATE_TYPES,
        projection=None,
        projections_by_source_type={},
        normalization_profile=None,
        metric_names=(
            metric_names if metric_names is not None
            else DEFAULT_ALTO_METRICS
        ),
        warnings=DEFAULT_ALTO_WARNINGS + extra_warnings,
        ignored_dimensions=DEFAULT_ALTO_IGNORED_DIMENSIONS + extra_ignored_dimensions,
    )


__all__ = [
    "build_alto_view",
    "DEFAULT_ALTO_METRICS",
    "DEFAULT_ALTO_CANDIDATE_TYPES",
    "DEFAULT_ALTO_IGNORED_DIMENSIONS",
    "DEFAULT_ALTO_WARNINGS",
]
