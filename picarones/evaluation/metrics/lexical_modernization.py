"""Détection de la sur-normalisation lexicale par les LLM/VLM —
Sprint 80 (A.I.7).

Sprint 80 — A.I.7 du plan d'évolution 2026.

Pourquoi ce module
------------------
Le détecteur ``llm_hallucination_flag`` (Sprint 19) signale qu'un
moteur sur-normalise (« 0,05 % »).  Mais ce score agrégé ne dit
rien sur **quoi** corriger dans le prompt.  Ce module produit
une **table de fréquences détaillée** :

+----------------------+--------------------+------+----------+
| Forme historique GT  | Forme modernisée   | n GT | % modern |
+======================+====================+======+==========+
| maistre              | maître             |   47 |     85 % |
| nostre               | nostre             |   92 |      8 % |
| veoir                | voir               |   23 |    100 % |
+----------------------+--------------------+------+----------+

Lecture immédiate : *« le LLM modernise systématiquement
maistre → maître ; pour préserver l'orthographe historique, ajouter
au prompt "ne pas moderniser maistre, nostre, veoir" »*.

Méthode
-------
Alignement mot-à-mot via ``difflib.SequenceMatcher``.  Chaque
``replace`` ou ``equal`` produit une paire ``(gt_token,
hyp_token)``.  On accumule pour chaque ``gt_token`` :

- ``n_total`` : nombre d'occurrences du token dans la GT
- ``n_modernized`` : nombre d'occurrences où ``hyp_token != gt_token``
- ``variants`` : dict des hyp_tokens observés avec leur count

Stop-list
---------
L'utilisateur peut passer ``stop_list`` (ensemble de tokens GT à
ignorer).  Par défaut, vide — le module ne tente pas de deviner ce
qui est « moderne » ou « historique », c'est au chercheur de
fournir le filtre adapté à son corpus.

Sortie
------
``compute_lexical_modernization`` retourne une structure adaptée
au rendu HTML.  ``aggregate_lexical_modernization`` agrège
plusieurs documents.

Limites documentées
-------------------
- Tokenisation au niveau mot (split sur espace) — cohérent avec
  ``taxonomy.py`` et autres modules.  Pas de stemming ni de
  lemmatisation.
- La métrique mesure la **réécriture lexicale** ; elle n'attrape
  pas les modernisations infra-mot (perte du s long ſ qui se
  fond dans la même forme).  Pour ça, voir ``early_modern_typography``
  (Sprint 58) et ``equivalence_profile`` (Sprint 78).
"""

from __future__ import annotations

import difflib
import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def _split_words(text: Optional[str]) -> list[str]:
    """Tokenisation simple par split sur whitespace."""
    if not text:
        return []
    return text.split()


def compute_lexical_modernization(
    reference: Optional[str],
    hypothesis: Optional[str],
    *,
    stop_list: Optional[Iterable[str]] = None,
    case_sensitive: bool = False,
) -> dict:
    """Calcule le tableau de modernisation lexicale pour un document.

    Returns
    -------
    dict
        ``{
            "n_gt_tokens": int,
            "tokens": {
                gt_token: {
                    "n_total": int,
                    "n_modernized": int,
                    "rate_modernized": float,  # ∈ [0, 1]
                    "variants": {hyp_token: count, ...},
                },
                ...
            },
        }``
        Si ``reference`` est vide → ``tokens == {}``.
    """
    ref_tokens = _split_words(reference)
    hyp_tokens = _split_words(hypothesis)
    if not ref_tokens:
        return {"n_gt_tokens": 0, "tokens": {}}

    if not case_sensitive:
        ref_for_match = [t.lower() for t in ref_tokens]
        hyp_for_match = [t.lower() for t in hyp_tokens]
    else:
        ref_for_match = ref_tokens
        hyp_for_match = hyp_tokens

    stop = frozenset(
        (t.lower() if not case_sensitive else t)
        for t in (stop_list or [])
    )

    # On accumule par gt_token (forme display = forme originale,
    # match key = forme casée selon ``case_sensitive``).
    tokens_data: dict[str, dict] = {}

    matcher = difflib.SequenceMatcher(
        None, ref_for_match, hyp_for_match, autojunk=False,
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                gt_orig = ref_tokens[i1 + k]
                gt_match = ref_for_match[i1 + k]
                if gt_match in stop:
                    continue
                slot = tokens_data.setdefault(
                    gt_orig,
                    {"n_total": 0, "n_modernized": 0, "variants": {}},
                )
                slot["n_total"] += 1
        elif tag == "replace":
            # Apparier 1-à-1 quand possible
            paired = min(i2 - i1, j2 - j1)
            for k in range(paired):
                gt_orig = ref_tokens[i1 + k]
                gt_match = ref_for_match[i1 + k]
                if gt_match in stop:
                    continue
                hyp_orig = hyp_tokens[j1 + k]
                slot = tokens_data.setdefault(
                    gt_orig,
                    {"n_total": 0, "n_modernized": 0, "variants": {}},
                )
                slot["n_total"] += 1
                slot["n_modernized"] += 1
                slot["variants"][hyp_orig] = slot["variants"].get(hyp_orig, 0) + 1
            # Si plus de gt que de hyp, le reste des gt_tokens est
            # « perdu » — on les compte comme totaux mais pas comme
            # modernisés (on ne sait pas en quoi).
            for k in range(paired, i2 - i1):
                gt_orig = ref_tokens[i1 + k]
                gt_match = ref_for_match[i1 + k]
                if gt_match in stop:
                    continue
                slot = tokens_data.setdefault(
                    gt_orig,
                    {"n_total": 0, "n_modernized": 0, "variants": {}},
                )
                slot["n_total"] += 1
                slot["n_modernized"] += 1
                slot["variants"]["∅"] = slot["variants"].get("∅", 0) + 1
        elif tag == "delete":
            # gt présent, pas en hyp → modernisation par
            # suppression (ou perte pure)
            for k in range(i2 - i1):
                gt_orig = ref_tokens[i1 + k]
                gt_match = ref_for_match[i1 + k]
                if gt_match in stop:
                    continue
                slot = tokens_data.setdefault(
                    gt_orig,
                    {"n_total": 0, "n_modernized": 0, "variants": {}},
                )
                slot["n_total"] += 1
                slot["n_modernized"] += 1
                slot["variants"]["∅"] = slot["variants"].get("∅", 0) + 1

    # Calcul du taux par token
    for slot in tokens_data.values():
        total = slot["n_total"]
        slot["rate_modernized"] = (
            slot["n_modernized"] / total if total > 0 else 0.0
        )

    return {
        "n_gt_tokens": len(ref_tokens),
        "tokens": tokens_data,
    }


def aggregate_lexical_modernization(
    per_doc_results: Iterable[dict],
) -> dict:
    """Agrège des ``compute_lexical_modernization`` per-doc.

    Renvoie la structure agrégée corpus-wide avec la même forme
    que ``compute_lexical_modernization``.
    """
    agg_tokens: dict[str, dict] = {}
    n_gt_total = 0
    for doc_result in per_doc_results:
        if not doc_result:
            continue
        n_gt_total += doc_result.get("n_gt_tokens", 0)
        for gt, data in (doc_result.get("tokens") or {}).items():
            slot = agg_tokens.setdefault(
                gt, {"n_total": 0, "n_modernized": 0, "variants": {}},
            )
            slot["n_total"] += data.get("n_total", 0)
            slot["n_modernized"] += data.get("n_modernized", 0)
            for hyp_t, count in (data.get("variants") or {}).items():
                slot["variants"][hyp_t] = slot["variants"].get(hyp_t, 0) + count

    for slot in agg_tokens.values():
        total = slot["n_total"]
        slot["rate_modernized"] = (
            slot["n_modernized"] / total if total > 0 else 0.0
        )
    return {
        "n_gt_tokens": n_gt_total,
        "tokens": agg_tokens,
    }


def top_modernized_tokens(
    data: dict,
    *,
    n: int = 20,
    min_total: int = 1,
) -> list[tuple[str, dict]]:
    """Top-N tokens GT par taux de modernisation.

    Filtre les tokens dont ``n_total < min_total`` (anecdotiques).
    Tri par ``rate_modernized`` décroissant, tie-break par
    ``n_total`` décroissant.
    """
    tokens = data.get("tokens") or {}
    candidates = [
        (gt, slot) for gt, slot in tokens.items()
        if slot.get("n_total", 0) >= min_total
        and slot.get("n_modernized", 0) > 0
    ]
    candidates.sort(
        key=lambda pair: (
            -pair[1].get("rate_modernized", 0.0),
            -pair[1].get("n_total", 0),
            pair[0],
        ),
    )
    return candidates[:n]


__all__ = [
    "compute_lexical_modernization",
    "aggregate_lexical_modernization",
    "top_modernized_tokens",
]
