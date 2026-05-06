"""Métriques structurelles ALTO — Sprint A14-S15.

Métriques typées ``(ALTO_XML, ALTO_XML)`` qui mesurent la fidélité
**documentaire** d'un ALTO produit par un pipeline (par exemple un
reconstructeur post-correction LLM ou un VLM avec module
ALTO_reconstruction) face à la GT ALTO du corpus.

Distinct de ``picarones/measurements/alto_metrics.py`` (legacy)
qui calcule CER/WER sur le **texte extrait** des deux ALTO.  Ici
on mesure la **structure** : nombre de lignes, présence de bbox,
ordre de lecture cohérent.

Métriques livrées au S15
------------------------
- ``compute_alto_validity(ref, hyp)`` — 1.0 si l'hypothèse a une
  structure cohérente (≥ 1 page, ≥ 1 bloc, ≥ 1 ligne).  Détecte
  les ALTO vides ou tronqués.
- ``compute_line_count_ratio(ref, hyp)`` — ``min(n_hyp, n_ref) /
  max(n_hyp, n_ref)`` ∈ [0, 1].  1.0 = même nombre de lignes.
- ``compute_word_box_coverage(ref, hyp)`` — fraction des
  ``AltoString`` de l'hypothèse qui ont une ``bbox``.  1.0 = tous
  les mots ont une boîte (cas idéal pour un reconstructeur ALTO).

Reportées à des sprints suivants (post-livraison)
-------------------------------------------------
- ``textline_alignment`` (IoU des bbox de lignes) — exige un
  algorithme d'alignement bipartite par bbox.
- ``reading_order_consistency`` (Kendall tau sur les IDs de
  lignes) — exige un mapping ID → position.
- ``layout_f1`` (ICDAR 2015) — déjà implémenté dans
  ``evaluation/metrics/layout.py`` (migré au S10) sur des
  ``Region`` génériques ; un wrapper ALTO peut être ajouté plus
  tard.

Convention de signature
-----------------------
Les payloads attendus sont des ``AltoDocument`` parsés (par le
``payload_loader`` du service applicatif).  Si le caller passe
des bytes XML brut, il doit appeler ``parse_alto`` lui-même
en amont.

higher_is_better
----------------
Toutes les métriques de ce module ∈ [0, 1] avec ``higher_is_better=True``
(1.0 = parfait, 0.0 = pire).  Cohérent avec le schéma ICDAR pour
les métriques de fidélité documentaire.
"""

from __future__ import annotations

from picarones.formats.alto.types import AltoDocument


def _count_lines(doc: AltoDocument) -> int:
    """Compte le nombre total de ``AltoLine`` dans un document."""
    return sum(
        len(block.lines)
        for page in doc.pages
        for block in page.blocks
    )


def _count_strings(doc: AltoDocument) -> int:
    """Compte le nombre total de ``AltoString`` dans un document."""
    return sum(
        len(line.strings)
        for page in doc.pages
        for block in page.blocks
        for line in block.lines
    )


def compute_alto_validity(
    reference: AltoDocument,
    hypothesis: AltoDocument,
) -> float:
    """Vérifie que l'hypothèse a une structure ALTO cohérente.

    Cohérence = au moins 1 page ET au moins 1 bloc ET au moins
    1 ligne dans l'hypothèse.  Détecte les ALTO vides, tronqués,
    ou produits par un reconstructeur défaillant.

    Returns
    -------
    float
        1.0 si l'hypothèse est structurellement cohérente,
        0.0 sinon.

    Notes
    -----
    On ne compare PAS la cohérence à la référence ici — la
    référence est juste passée pour homogénéité d'API avec les
    autres métriques.  Un ALTO de référence vide (cas dégénéré)
    n'invalide pas l'hypothèse.
    """
    if not hypothesis.pages:
        return 0.0
    has_block = any(page.blocks for page in hypothesis.pages)
    if not has_block:
        return 0.0
    has_line = any(
        block.lines
        for page in hypothesis.pages
        for block in page.blocks
    )
    if not has_line:
        return 0.0
    return 1.0


def compute_line_count_ratio(
    reference: AltoDocument,
    hypothesis: AltoDocument,
) -> float:
    """Ratio min/max du nombre de lignes des deux ALTO.

    Returns
    -------
    float
        ``min(n_hyp, n_ref) / max(n_hyp, n_ref)`` ∈ [0, 1].
        1.0 = même nombre de lignes.  0.0 si l'un des deux n'a
        aucune ligne (cas dégénéré).

    Permet de détecter un reconstructeur qui invente ou perd des
    lignes vs la GT.  Ne dit RIEN sur l'alignement spatial —
    c'est ``textline_alignment`` (post-livraison) qui mesurera
    cette dimension.
    """
    n_ref = _count_lines(reference)
    n_hyp = _count_lines(hypothesis)
    if n_ref == 0 and n_hyp == 0:
        return 1.0  # convention : deux vides identiques
    if n_ref == 0 or n_hyp == 0:
        return 0.0
    return min(n_ref, n_hyp) / max(n_ref, n_hyp)


def compute_word_box_coverage(
    reference: AltoDocument,
    hypothesis: AltoDocument,
) -> float:
    """Fraction des ``AltoString`` de l'hypothèse qui ont une ``bbox``.

    Returns
    -------
    float
        ``n_strings_with_bbox / n_strings_total`` ∈ [0, 1].
        1.0 = tous les mots ont une boîte (cas idéal pour un
        reconstructeur ALTO).  0.0 si l'hypothèse n'a aucun mot.

    La référence n'est pas utilisée dans le calcul, mais elle est
    passée pour homogénéité d'API.  Un caller qui veut comparer
    "candidat a-t-il autant de bbox que la GT" peut mesurer
    ``compute_word_box_coverage(gt, hyp) / compute_word_box_coverage(hyp, gt)``
    ou utiliser un calcul dédié.
    """
    total = _count_strings(hypothesis)
    if total == 0:
        return 0.0
    with_bbox = sum(
        1
        for page in hypothesis.pages
        for block in page.blocks
        for line in block.lines
        for s in line.strings
        if s.bbox is not None
    )
    return with_bbox / total


__all__ = [
    "compute_alto_validity",
    "compute_line_count_ratio",
    "compute_word_box_coverage",
]
