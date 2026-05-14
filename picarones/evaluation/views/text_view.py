"""``TextView`` — vue canonique 1, Sprint A14-S14.

Première vue d'évaluation cible BnF : "quel pipeline produit le
meilleur texte final ?"

Cette vue répond à un cas d'usage central : comparer librement
plusieurs pipelines hétérogènes (Tesseract texte brut, OCR+LLM
texte corrigé, OCR+LLM+ALTO remappé, VLM avec reconstruction
ALTO, etc.) en projetant **toutes** leurs sorties vers du texte
plat avant de calculer CER/WER.

Garde-fou méthodologique
------------------------
Comparer un texte brut OCR et un ALTO reconstruit serait
trompeur si on regardait juste les chiffres : l'ALTO porte une
structure que le texte plat n'a pas.  ``TextView`` documente
explicitement cette projection dans le ``ProjectionReport`` du
``ViewResult`` : pour chaque artefact non-RAW_TEXT, le rapport
listera les ``ignored_dimensions`` (``geometry``, ``blocks``,
``reading_order``, ``ids``...) et les ``warnings`` du projecteur
correspondant.

Types acceptés
--------------
- ``RAW_TEXT`` : pas de projection (identité).
- ``CORRECTED_TEXT`` : pas de projection (identité).
- ``ALTO_XML`` : projeté via ``AltoToText``.
- ``PAGE_XML`` : projeté via ``PageToText``.
- ``CANONICAL_DOCUMENT`` : projeté via ``CanonicalToText``.

Métriques par défaut
--------------------
``cer``, ``wer``, ``mer``, ``wil``.  Le caller peut surcharger
via le paramètre ``metric_names`` du builder.

Limites assumées
----------------
- Pas de comparaison fuzzy / search recall — c'est ``SearchView``
  (S16).
- Pas d'évaluation structurelle ALTO — c'est ``AltoView`` (S15).
- ``CANONICAL_DOCUMENT`` peut perdre beaucoup de structure ; le
  warning du ``ProjectionReport`` le signale.
"""

from __future__ import annotations

from picarones.domain.artifacts import ArtifactType
from picarones.domain.evaluation_spec import EvaluationView
from picarones.domain.projection_spec import ProjectionSpec


#: Métriques calculées par défaut quand on construit une ``TextView``
#: sans surcharge.  Toutes typées ``(RAW_TEXT, RAW_TEXT)`` (la
#: comparaison se fait toujours après projection vers texte).
DEFAULT_TEXT_METRICS: tuple[str, ...] = ("cer", "wer", "mer", "wil")


#: Types acceptés par défaut.  Le caller peut restreindre
#: (par exemple en construisant une ``TextView`` "OCR seul" qui
#: n'accepte que ``RAW_TEXT``).
DEFAULT_TEXT_CANDIDATE_TYPES: frozenset[ArtifactType] = frozenset({
    ArtifactType.RAW_TEXT,
    ArtifactType.CORRECTED_TEXT,
    ArtifactType.ALTO_XML,
    ArtifactType.PAGE_XML,
    ArtifactType.CANONICAL_DOCUMENT,
})


#: Mapping ``source_type → ProjectionSpec`` pour la projection
#: automatique vers RAW_TEXT.  Aucune projection pour RAW_TEXT et
#: CORRECTED_TEXT (déjà du texte).
DEFAULT_TEXT_PROJECTIONS: dict[ArtifactType, ProjectionSpec] = {
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


#: ``ignored_dimensions`` par défaut.  Listées explicitement dans
#: le rapport pour qu'un lecteur sache **ce que la vue ne dit
#: PAS** sur les pipelines comparés.
DEFAULT_TEXT_IGNORED_DIMENSIONS: tuple[str, ...] = (
    "geometry",
    "block_structure",
    "reading_order",
    "ids",
    "confidence",
    "formatting",
)


#: ``warnings`` par défaut.  Affichés en tête du bloc TextView
#: dans le rapport pour signaler la portée de la comparaison.
DEFAULT_TEXT_WARNINGS: tuple[str, ...] = (
    "Cette vue compare les sorties textuelles finales après "
    "projection éventuelle.  Les pipelines qui produisent ALTO/PAGE/"
    "markdown sont projetés vers du texte plat — leurs structures "
    "spatiale et documentaire ne sont PAS évaluées ici.  Pour "
    "évaluer la qualité ALTO, voir AltoView (S15).",
)


def build_text_view(
    *,
    name: str = "text_final",
    description: str = (
        "Compare les sorties textuelles finales après projection "
        "éventuelle (ALTO/PAGE/markdown → texte plat)."
    ),
    candidate_types: frozenset[ArtifactType] | None = None,
    metric_names: tuple[str, ...] | None = None,
    normalization_profile: str | None = None,
    char_exclude: str | None = None,
    extra_warnings: tuple[str, ...] = (),
    extra_ignored_dimensions: tuple[str, ...] = (),
) -> EvaluationView:
    """Construit la vue canonique TextView.

    Parameters
    ----------
    name:
        Identifiant lisible de la vue (``"text_final"`` par défaut).
    description:
        Phrase courte affichée dans le rapport.
    candidate_types:
        Set des types acceptés.  Défaut : tous les 5 types texte
        ou projetables vers texte.
    metric_names:
        Métriques calculées.  Défaut : ``("cer", "wer", "mer", "wil")``.
    normalization_profile:
        Profil de normalisation texte appliqué après projection
        (cf. ``picarones.formats.text.normalization``).  ``None``
        par défaut (NFC implicite).  Exemples utiles :
        ``"medieval_french"``, ``"caseless"``, ``"sans_apostrophes"``.
    extra_warnings:
        Avertissements additionnels à propager dans le rapport en
        plus des warnings par défaut.
    extra_ignored_dimensions:
        Dimensions additionnelles à signaler comme ignorées.
    """
    return EvaluationView(
        name=name,
        description=description,
        candidate_types=(
            candidate_types if candidate_types is not None
            else DEFAULT_TEXT_CANDIDATE_TYPES
        ),
        projection=None,
        projections_by_source_type=DEFAULT_TEXT_PROJECTIONS,
        normalization_profile=normalization_profile,
        char_exclude=char_exclude,
        metric_names=(
            metric_names if metric_names is not None
            else DEFAULT_TEXT_METRICS
        ),
        warnings=DEFAULT_TEXT_WARNINGS + extra_warnings,
        ignored_dimensions=DEFAULT_TEXT_IGNORED_DIMENSIONS + extra_ignored_dimensions,
    )


__all__ = [
    "build_text_view",
    "DEFAULT_TEXT_METRICS",
    "DEFAULT_TEXT_CANDIDATE_TYPES",
    "DEFAULT_TEXT_PROJECTIONS",
    "DEFAULT_TEXT_IGNORED_DIMENSIONS",
    "DEFAULT_TEXT_WARNINGS",
]
