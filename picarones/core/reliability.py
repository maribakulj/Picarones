"""Métriques de fiabilité — Sprint 83 (A.II.4).

Sprint 83 — A.II.4 du plan d'évolution 2026 (Étape 4).

Pourquoi ce module
------------------
Une publication scientifique qui rapporte un CER LLM sans
stabilité est méthodologiquement faible.  Et un benchmark qui
ignore le plafond humain (« deux paléographes ne sont pas même
d'accord ») crée des classements faussement optimistes.  Ce
module livre deux familles complémentaires :

1. **Inter-annotator agreement (IAA)** — quand un document a
   plusieurs GT (deux paléographes, par ex.), Cohen κ et
   Krippendorff α mesurent l'accord au niveau caractère.
   Lecture : *« le CER de Pero (4,2 %) approche le plafond
   humain (κ = 0,89). »*

2. **Stabilité multi-runs** — quand on relance la même
   pipeline LLM N fois sur les mêmes documents, on mesure :
   variance du CER, taux de tokens divergents entre runs,
   CER pairwise moyen.

Périmètre Sprint 83
-------------------
**Couche de calcul uniquement** — fonctions pures, pas
d'intégration runner ni de vue HTML.  L'extension du loader
pour accepter ``doc_001.gt.A.txt`` / ``doc_001.gt.B.txt`` est
documentée comme dépendance future ; en attendant le sprint
dédié, on prend deux strings GT en entrée.

Méthode
-------
*IAA caractère par caractère.*  On aligne les deux GT par
``difflib.SequenceMatcher`` au niveau caractère et on construit
une table de contingence ``(annotator_a_char, annotator_b_char)``
sur les positions ``equal`` ou ``replace``.  Cohen κ utilise
cette table directement.  Krippendorff α utilise la version
matricielle (différence binaire pour le mode nominal).

*Stabilité multi-runs.*  ``compute_multirun_stability(runs)``
prend une liste de N transcriptions du **même** document et
renvoie variance/écart-type/coefficient de variation du CER si
référence fournie ; sinon, taux pairwise de divergence
(intersection-vs-union des tokens).
"""

from __future__ import annotations

import logging
import statistics
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Helpers d'alignement caractère par caractère
# ──────────────────────────────────────────────────────────────────────────


def _aligned_char_pairs(
    text_a: str, text_b: str,
) -> list[tuple[str, str]]:
    """Aligne ``text_a`` et ``text_b`` caractère par caractère.

    Retourne la liste des paires alignées sur les segments
    ``equal`` et ``replace`` de ``SequenceMatcher`` (les ``insert``
    et ``delete`` sont ignorés — pas d'alignement valide).
    """
    if not text_a and not text_b:
        return []
    import difflib
    matcher = difflib.SequenceMatcher(None, text_a, text_b, autojunk=False)
    pairs: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                pairs.append((text_a[i1 + k], text_b[j1 + k]))
        elif tag == "replace":
            paired = min(i2 - i1, j2 - j1)
            for k in range(paired):
                pairs.append((text_a[i1 + k], text_b[j1 + k]))
        # insert/delete : pas d'alignement bilatéral exploitable
    return pairs


__all__: list[str] = []


# ──────────────────────────────────────────────────────────────────────────
# 1. Cohen's kappa (deux annotateurs, accord nominal)
# ──────────────────────────────────────────────────────────────────────────


def cohen_kappa(
    annotations_a: Sequence,
    annotations_b: Sequence,
) -> Optional[float]:
    """Cohen's κ entre deux annotateurs sur des observations
    appariées.

    Définition :

        κ = (po - pe) / (1 - pe)

    où ``po`` est l'accord observé (proportion de paires égales)
    et ``pe`` l'accord attendu par hasard (somme sur les classes
    de p_a(c) × p_b(c)).

    Conventions :
    - retourne ``None`` si les deux séquences sont vides ou de
      tailles incompatibles ;
    - κ = 1.0 quand l'accord est parfait, 0.0 quand il égale le
      hasard, négatif si pire que le hasard ;
    - quand ``pe == 1`` (un seul label dans les deux séquences),
      retourne 1.0 si les séquences sont identiques, 0.0 sinon
      (κ est mathématiquement indéfini, on choisit une
      convention transparente documentée).
    """
    if len(annotations_a) != len(annotations_b):
        return None
    n = len(annotations_a)
    if n == 0:
        return None
    # Accord observé
    agree = sum(1 for a, b in zip(annotations_a, annotations_b) if a == b)
    p_o = agree / n
    # Accord attendu par hasard
    from collections import Counter
    count_a = Counter(annotations_a)
    count_b = Counter(annotations_b)
    classes = set(count_a) | set(count_b)
    p_e = sum(
        (count_a.get(c, 0) / n) * (count_b.get(c, 0) / n)
        for c in classes
    )
    if p_e >= 1.0 - 1e-12:
        # Indéfini ; convention : 1 si identité totale, 0 sinon
        return 1.0 if p_o >= 1.0 - 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


__all__.append("cohen_kappa")


# ──────────────────────────────────────────────────────────────────────────
# 2. Krippendorff's alpha (généralisation à N annotateurs)
# ──────────────────────────────────────────────────────────────────────────


def krippendorff_alpha(
    annotations_per_unit: Sequence[Sequence],
) -> Optional[float]:
    """Krippendorff's α en mode nominal pour N annotateurs.

    Parameters
    ----------
    annotations_per_unit:
        Liste d'unités, chaque unité étant la liste des
        annotations produites par les différents annotateurs sur
        cette unité.  ``None`` dans une cellule = annotation
        manquante (autorisée).

    Définition (Krippendorff 1980, équation pour métrique
    nominale) :

        α = 1 - D_o / D_e

    où ``D_o`` est le désaccord observé (paires en désaccord
    intra-unité, normalisées) et ``D_e`` le désaccord attendu
    par hasard.  ``α = 1`` accord parfait, ``α = 0`` hasard,
    négatif si pire.

    Conventions :
    - unités avec moins de 2 annotations valides : ignorées
      (Krippendorff convention) ;
    - retourne ``None`` si moins d'une unité utilisable ou
      ``D_e == 0`` (un seul label dans tout le corpus).
    """
    from collections import Counter
    # Valeurs observées au niveau corpus
    value_counts: Counter = Counter()
    pair_disagree = 0.0
    pair_total = 0.0
    for unit in annotations_per_unit:
        valid = [v for v in unit if v is not None]
        m = len(valid)
        if m < 2:
            continue
        # paires intra-unité (sans repetition, ordonné)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                pair_total += 1.0 / (m - 1)
                if valid[i] != valid[j]:
                    pair_disagree += 1.0 / (m - 1)
        for v in valid:
            value_counts[v] += 1
    if pair_total == 0:
        return None
    n_total = sum(value_counts.values())
    if n_total < 2:
        return None
    # Désaccord attendu (sur paires aléatoires sans remise)
    expected_disagree = 0.0
    for v_a, c_a in value_counts.items():
        for v_b, c_b in value_counts.items():
            if v_a != v_b:
                expected_disagree += c_a * c_b
    expected_disagree /= n_total * (n_total - 1)
    if expected_disagree <= 1e-12:
        return None
    d_o = pair_disagree / pair_total
    return 1.0 - (d_o / expected_disagree)


__all__.append("krippendorff_alpha")


# ──────────────────────────────────────────────────────────────────────────
# 3. Helpers IAA caractère
# ──────────────────────────────────────────────────────────────────────────


def compute_iaa(
    transcription_a: str,
    transcription_b: str,
) -> Optional[dict]:
    """Calcule κ et α au niveau caractère entre deux
    transcriptions du même document.

    Aligne via ``_aligned_char_pairs`` puis :
    - κ : sur la liste des paires alignées ;
    - α : sur les unités à 2 annotations (équivalent à κ sur ce
      cas, mais le cadre généralise à N annotateurs).

    Retourne ``None`` si pas d'alignement possible (transcriptions
    vides ou totalement disjointes).
    """
    pairs = _aligned_char_pairs(transcription_a, transcription_b)
    if not pairs:
        return None
    kappa = cohen_kappa([a for a, _ in pairs], [b for _, b in pairs])
    alpha = krippendorff_alpha([[a, b] for a, b in pairs])
    return {
        "n_aligned_chars": len(pairs),
        "cohen_kappa": kappa,
        "krippendorff_alpha": alpha,
        "agreement_rate": (
            sum(1 for a, b in pairs if a == b) / len(pairs)
        ),
    }


__all__.append("compute_iaa")


# ──────────────────────────────────────────────────────────────────────────
# 4. Stabilité multi-runs (variance CER, divergence pairwise)
# ──────────────────────────────────────────────────────────────────────────


def _split_words(text: str) -> list[str]:
    return text.split() if text else []


def compute_multirun_stability(
    runs: Sequence[str],
    *,
    reference: Optional[str] = None,
) -> Optional[dict]:
    """Mesure la stabilité de N runs successifs d'une même
    pipeline (typiquement LLM/VLM non déterministe) sur un
    document.

    Parameters
    ----------
    runs:
        Liste des transcriptions produites à chaque run (≥ 2).
    reference:
        Transcription de référence (GT). Si fournie, on calcule
        ``cer_per_run``, leur variance et leur coefficient de
        variation.

    Returns
    -------
    dict | None
        ``{
            "n_runs": int,
            "pairwise_disagreement_mean": float,  # divergence moyenne
            "pairwise_disagreement_max": float,
            "identical_run_rate": float,          # paires identiques / total
            "cer_per_run": Optional[list[float]],
            "cer_mean": Optional[float],
            "cer_stdev": Optional[float],
            "cer_cv": Optional[float],            # cv = stdev / mean
            "n_distinct_outputs": int,
        }``
        ou ``None`` si moins de 2 runs.
    """
    if len(runs) < 2:
        return None
    runs_list = list(runs)
    # Divergence pairwise (token-level Jaccard distance)
    n = len(runs_list)
    n_pairs = 0
    sum_disagree = 0.0
    max_disagree = 0.0
    n_identical = 0
    for i in range(n):
        for j in range(i + 1, n):
            n_pairs += 1
            tokens_i = set(_split_words(runs_list[i]))
            tokens_j = set(_split_words(runs_list[j]))
            union = tokens_i | tokens_j
            if not union:
                disagree = 0.0
            else:
                disagree = 1.0 - len(tokens_i & tokens_j) / len(union)
            sum_disagree += disagree
            if disagree > max_disagree:
                max_disagree = disagree
            if runs_list[i] == runs_list[j]:
                n_identical += 1
    pairwise_mean = sum_disagree / n_pairs if n_pairs else 0.0
    identical_rate = n_identical / n_pairs if n_pairs else 0.0
    distinct = len(set(runs_list))

    cer_per_run: Optional[list[float]] = None
    cer_mean: Optional[float] = None
    cer_stdev: Optional[float] = None
    cer_cv: Optional[float] = None
    if reference is not None:
        from picarones.core.metrics import _cer_from_strings
        cer_per_run = [_cer_from_strings(reference, r) for r in runs_list]
        cer_per_run = [v for v in cer_per_run if v is not None]
        if cer_per_run:
            cer_mean = statistics.fmean(cer_per_run)
            if len(cer_per_run) >= 2:
                cer_stdev = statistics.stdev(cer_per_run)
                cer_cv = (
                    cer_stdev / cer_mean if cer_mean and cer_mean > 0
                    else None
                )
    return {
        "n_runs": n,
        "pairwise_disagreement_mean": pairwise_mean,
        "pairwise_disagreement_max": max_disagree,
        "identical_run_rate": identical_rate,
        "n_distinct_outputs": distinct,
        "cer_per_run": cer_per_run,
        "cer_mean": cer_mean,
        "cer_stdev": cer_stdev,
        "cer_cv": cer_cv,
    }


__all__.append("compute_multirun_stability")
