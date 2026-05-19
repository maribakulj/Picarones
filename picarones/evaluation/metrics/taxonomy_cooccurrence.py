"""Co-occurrence des classes taxonomiques d'erreur — Sprint 75 (A.I.4 chantier 1).

A.I.4 chantier 1 du plan d'évolution 2026.

Pourquoi ce module
------------------
La taxonomie d'erreurs (10 classes, ``picarones/core/taxonomy.py``)
est calculée par document mais le rapport actuel ne montre qu'un
seul histogramme global.  La roadmap A.I.4 demande trois lectures
plus fines de cette taxonomie ; ce sprint livre la première :
**co-occurrence**.

Si ``ligature_error`` et ``abbreviation_error`` co-occurrent
toujours dans les mêmes documents, c'est un signal de scribe
particulier — utile pour stratifier le corpus *a posteriori*
(qu'est-ce qui caractérise les documents difficiles ?).

Mesure
------
Indice de **Jaccard** entre paires de classes au niveau
**document** :

.. math::

   J(A, B) = \\frac{|D_A \\cap D_B|}{|D_A \\cup D_B|}

où ``D_X`` est l'ensemble des documents qui contiennent au moins
une erreur de classe ``X``.

- ``J(A, B) = 1`` : A et B apparaissent toujours ensemble (et
  jamais l'un sans l'autre).
- ``J(A, B) = 0`` : A et B ne co-occurrent jamais.
- ``J(A, B) = 0,5`` : A et B partagent la moitié de leur union.

Stratégie de découpage
----------------------
Couche de calcul pure d'abord (pattern Sprint 35, 38, 52-58).
Le rendu HTML (heatmap SVG) est livré dans le même sprint pour
boucler la dimension ; les chantiers 2 et 3 d'A.I.4 (évolution
intra-document, taxonomie comparative) suivent.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def compute_taxonomy_cooccurrence(
    per_doc_classes: Iterable[Iterable[str]],
    *,
    min_doc_count: int = 1,
    top_n_pairs: int = 10,
) -> Optional[dict]:
    """Calcule la matrice de Jaccard inter-classes au niveau document.

    Parameters
    ----------
    per_doc_classes:
        Itérable de docs, chaque doc étant un itérable de noms de
        classes taxonomiques détectées (set, list, tuple…).
        Les doublons à l'intérieur d'un doc sont ignorés (présence
        binaire au niveau doc).
    min_doc_count:
        Nombre minimum de documents dans lesquels une classe doit
        apparaître pour figurer dans la matrice (défaut 1).
        Permet d'écarter les classes anecdotiques.
    top_n_pairs:
        Nombre de paires retournées dans ``top_pairs`` (triées par
        Jaccard décroissant).  Défaut 10.

    Returns
    -------
    Optional[dict]
        ``{
            "classes": list[str],          # triées alpha
            "n_documents": int,
            "doc_count": dict[str, int],   # nb docs par classe
            "cooccurrence_matrix": dict[str, dict[str, float]],
                # symétrique, diagonale = 1.0 (sauf classe vide)
            "top_pairs": list[tuple[str, str, float]],
                # paires les plus co-occurrentes (Jaccard désc.)
        }``
        ou ``None`` si aucune classe ne dépasse ``min_doc_count``
        ou si l'itérable est vide.
    """
    docs: list[frozenset[str]] = []
    for doc_classes in per_doc_classes:
        if doc_classes is None:
            continue
        cleaned = frozenset(c for c in doc_classes if c)
        docs.append(cleaned)
    if not docs:
        return None

    # Comptage par classe
    doc_count: dict[str, int] = {}
    for doc in docs:
        for cls in doc:
            doc_count[cls] = doc_count.get(cls, 0) + 1

    # Filtrage min_doc_count
    classes = sorted(
        c for c, n in doc_count.items() if n >= min_doc_count
    )
    if not classes:
        return None

    # Matrice de Jaccard
    matrix: dict[str, dict[str, float]] = {
        c: {} for c in classes
    }
    for i, ca in enumerate(classes):
        docs_a = {idx for idx, d in enumerate(docs) if ca in d}
        for cb in classes[i:]:
            if ca == cb:
                # Diagonale : Jaccard(X, X) = 1 si X est présent
                matrix[ca][cb] = 1.0 if docs_a else 0.0
                continue
            docs_b = {idx for idx, d in enumerate(docs) if cb in d}
            inter = len(docs_a & docs_b)
            union = len(docs_a | docs_b)
            jaccard = inter / union if union > 0 else 0.0
            matrix[ca][cb] = jaccard
            matrix[cb][ca] = jaccard  # symétrique

    # Top paires (hors diagonale)
    pairs: list[tuple[str, str, float]] = []
    for i, ca in enumerate(classes):
        for cb in classes[i + 1:]:
            j = matrix[ca][cb]
            if j > 0:
                pairs.append((ca, cb, j))
    pairs.sort(key=lambda p: (-p[2], p[0], p[1]))
    top_pairs = pairs[:top_n_pairs]

    return {
        "classes": classes,
        "n_documents": len(docs),
        "doc_count": doc_count,
        "cooccurrence_matrix": matrix,
        "top_pairs": top_pairs,
    }


__all__ = [
    "compute_taxonomy_cooccurrence",
]
