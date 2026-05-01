"""Helpers de câblage des métriques philologiques (Sprints 55-60) au runner.

Sprint 61 — câblage backend des 6 modules philologiques :

- ``unicode_blocks``    (Sprint 55)
- ``abbreviations``     (Sprint 56)
- ``mufi``              (Sprint 57)
- ``early_modern``      (Sprint 58)
- ``modern_archives``   (Sprint 59)
- ``roman_numerals``    (Sprint 60)

Principe « adaptive »
----------------------
Un module n'est inclus dans le résultat que si la **GT contient du
signal exploitable** pour ce module.  Cette logique évite de polluer
les rapports sur les corpus sans marqueurs philologiques (typique
sur des données XXIᵉ ou des transcriptions modernes propres).

Coût
----
Les 6 calculs sont O(N) sur la longueur du texte ; le surcoût total
par document est négligeable face à un appel OCR.  L'activation est
donc **automatique** (pas d'opt-in), contrairement aux backends NER
ou calibration qui exigent une dépendance externe ou des données
spécifiques.
"""

from __future__ import annotations

import logging
from typing import Optional

from picarones.measurements.abbreviations import compute_abbreviation_metrics
from picarones.measurements.early_modern_typography import compute_early_modern_metrics
from picarones.measurements.modern_archives import compute_modern_archives_metrics
from picarones.measurements.mufi import compute_mufi_coverage
from picarones.measurements.roman_numerals import compute_roman_numeral_metrics
from picarones.measurements.unicode_blocks import compute_unicode_block_accuracy

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Critères « le module a-t-il du signal sur ce document ? »
# ──────────────────────────────────────────────────────────────────────────
#
# Pour chaque module, on définit un prédicat sur le résultat : si vrai,
# le module est inclus ; sinon, il est omis pour ne pas alourdir le
# rapport.

def _has_unicode_signal(result: dict) -> bool:
    # Le module retourne toujours du signal dès que GT non-vide ; on
    # n'inclut que si la GT a au moins un caractère **hors Basic
    # Latin** (sinon le breakdown se réduit à 100 % Basic Latin et
    # n'apporte rien au lecteur).
    per_block = result.get("per_block", {})
    for block, stats in per_block.items():
        if block == "Basic Latin":
            continue
        if stats.get("total", 0) > 0:
            return True
    return False


def _has_abbreviation_signal(result: dict) -> bool:
    return result.get("n_abbreviations_in_reference", 0) > 0


def _has_mufi_signal(result: dict) -> bool:
    return result.get("n_mufi_chars_reference", 0) > 0


def _has_early_modern_signal(result: dict) -> bool:
    return result.get("n_markers_reference", 0) > 0


def _has_modern_archives_signal(result: dict) -> bool:
    return result.get("n_markers_reference", 0) > 0


def _has_roman_numeral_signal(result: dict) -> bool:
    return result.get("n_numerals_reference", 0) > 0


# Ordre fixé pour la reproductibilité des sorties.
_PHILOLOGICAL_MODULES: tuple[
    tuple[str, callable, callable], ...
] = (
    ("unicode_blocks",  compute_unicode_block_accuracy, _has_unicode_signal),
    ("abbreviations",   compute_abbreviation_metrics,   _has_abbreviation_signal),
    ("mufi",            compute_mufi_coverage,          _has_mufi_signal),
    ("early_modern",    compute_early_modern_metrics,   _has_early_modern_signal),
    ("modern_archives", compute_modern_archives_metrics, _has_modern_archives_signal),
    ("roman_numerals",  compute_roman_numeral_metrics,  _has_roman_numeral_signal),
)


# ──────────────────────────────────────────────────────────────────────────
# Calcul par document
# ──────────────────────────────────────────────────────────────────────────


def compute_philological_metrics(
    reference: Optional[str],
    hypothesis: Optional[str],
) -> Optional[dict]:
    """Calcule les 6 métriques philologiques pour un document.

    Retourne un dict avec une clé par module ayant du signal, ou
    ``None`` si aucun module n'en a (corpus sans marqueur
    philologique pertinent).

    En cas d'erreur dans un module individuel, le module est
    silencieusement omis et un warning est émis (les autres modules
    restent calculés).
    """
    ref = reference or ""
    if not ref:
        return None
    out: dict = {}
    for name, compute_fn, has_signal_fn in _PHILOLOGICAL_MODULES:
        try:
            result = compute_fn(ref, hypothesis or "")
        except Exception as exc:  # pragma: no cover — défense en profondeur
            logger.warning(
                "[philological_hooks] module %s a échoué : %s", name, exc,
            )
            continue
        if has_signal_fn(result):
            out[name] = result
    return out if out else None


# ──────────────────────────────────────────────────────────────────────────
# Agrégation corpus-wide par moteur
# ──────────────────────────────────────────────────────────────────────────


def _aggregate_unicode(per_doc: list[dict]) -> dict:
    total_correct = 0
    total_chars = 0
    per_block: dict[str, dict[str, int]] = {}
    for d in per_doc:
        for block, stats in d.get("per_block", {}).items():
            slot = per_block.setdefault(block, {"correct": 0, "total": 0})
            slot["correct"] += stats.get("correct", 0)
            slot["total"] += stats.get("total", 0)
            total_correct += stats.get("correct", 0)
            total_chars += stats.get("total", 0)
    out_per_block = {
        block: {
            "correct": slot["correct"],
            "total": slot["total"],
            "accuracy": (
                slot["correct"] / slot["total"] if slot["total"] > 0 else 0.0
            ),
        }
        for block, slot in sorted(per_block.items())
    }
    return {
        "global_accuracy": total_correct / total_chars if total_chars > 0 else 0.0,
        "n_chars_total": total_chars,
        "n_chars_correct": total_correct,
        "per_block": out_per_block,
        "doc_count": len(per_doc),
    }


def _aggregate_abbreviations(per_doc: list[dict]) -> dict:
    n_total = 0
    n_strict = 0
    n_expansion = 0
    per_abbr: dict[str, dict[str, int]] = {}
    for d in per_doc:
        n_total += d.get("n_abbreviations_in_reference", 0)
        n_strict += d.get("n_strict_preserved", 0)
        n_expansion += d.get("n_expansion_preserved", 0)
        for entry in d.get("per_abbreviation", []):
            slot = per_abbr.setdefault(
                entry["abbr"],
                {"total": 0, "strict": 0, "expansion": 0},
            )
            slot["total"] += 1
            if entry.get("strict_preserved"):
                slot["strict"] += 1
            if entry.get("expansion_preserved"):
                slot["expansion"] += 1
    return {
        "n_abbreviations_in_reference": n_total,
        "n_strict_preserved": n_strict,
        "n_expansion_preserved": n_expansion,
        "global_strict_score": n_strict / n_total if n_total > 0 else 0.0,
        "global_expansion_score": n_expansion / n_total if n_total > 0 else 0.0,
        "per_abbreviation": {
            abbr: {
                "n_total": slot["total"],
                "n_strict": slot["strict"],
                "n_expansion": slot["expansion"],
                "strict_score": slot["strict"] / slot["total"],
                "expansion_score": slot["expansion"] / slot["total"],
            }
            for abbr, slot in sorted(per_abbr.items())
        },
        "doc_count": len(per_doc),
    }


def _aggregate_mufi(per_doc: list[dict]) -> dict:
    n_total = 0
    n_preserved = 0
    per_char: dict[str, dict[str, int]] = {}
    for d in per_doc:
        n_total += d.get("n_mufi_chars_reference", 0)
        n_preserved += d.get("n_mufi_chars_preserved", 0)
        for ch, stats in d.get("per_char", {}).items():
            slot = per_char.setdefault(ch, {"total": 0, "preserved": 0})
            slot["total"] += stats.get("total", 0)
            slot["preserved"] += stats.get("preserved", 0)
    return {
        "n_mufi_chars_reference": n_total,
        "n_mufi_chars_preserved": n_preserved,
        "coverage": n_preserved / n_total if n_total > 0 else 0.0,
        "per_char": {
            ch: {
                "total": slot["total"],
                "preserved": slot["preserved"],
                "coverage": slot["preserved"] / slot["total"],
            }
            for ch, slot in sorted(per_char.items())
        },
        "doc_count": len(per_doc),
    }


def _aggregate_early_modern(per_doc: list[dict]) -> dict:
    n_total = 0
    n_preserved = 0
    per_cat: dict[str, dict[str, int]] = {}
    for d in per_doc:
        n_total += d.get("n_markers_reference", 0)
        n_preserved += d.get("n_markers_preserved", 0)
        for cat, stats in d.get("per_category", {}).items():
            slot = per_cat.setdefault(cat, {"total": 0, "preserved": 0})
            slot["total"] += stats.get("total", 0)
            slot["preserved"] += stats.get("preserved", 0)
    return {
        "n_markers_reference": n_total,
        "n_markers_preserved": n_preserved,
        "global_preservation": n_preserved / n_total if n_total > 0 else 0.0,
        "per_category": {
            cat: {
                "total": slot["total"],
                "preserved": slot["preserved"],
                "preservation": slot["preserved"] / slot["total"],
            }
            for cat, slot in sorted(per_cat.items())
        },
        "doc_count": len(per_doc),
    }


def _aggregate_modern_archives(per_doc: list[dict]) -> dict:
    n_total = 0
    n_strict = 0
    n_expansion = 0
    per_cat: dict[str, dict[str, int]] = {}
    for d in per_doc:
        n_total += d.get("n_markers_reference", 0)
        n_strict += d.get("n_strict_preserved", 0)
        n_expansion += d.get("n_expansion_preserved", 0)
        for cat, stats in d.get("per_category", {}).items():
            slot = per_cat.setdefault(
                cat, {"total": 0, "strict": 0, "expansion": 0},
            )
            slot["total"] += stats.get("n_total", 0)
            slot["strict"] += stats.get("n_strict_preserved", 0)
            slot["expansion"] += stats.get("n_expansion_preserved", 0)
    return {
        "n_markers_reference": n_total,
        "n_strict_preserved": n_strict,
        "n_expansion_preserved": n_expansion,
        "global_strict_score": n_strict / n_total if n_total > 0 else 0.0,
        "global_expansion_score": n_expansion / n_total if n_total > 0 else 0.0,
        "per_category": {
            cat: {
                "n_total": slot["total"],
                "n_strict_preserved": slot["strict"],
                "n_expansion_preserved": slot["expansion"],
                "strict_score": slot["strict"] / slot["total"],
                "expansion_score": slot["expansion"] / slot["total"],
            }
            for cat, slot in sorted(per_cat.items())
        },
        "doc_count": len(per_doc),
    }


def _aggregate_roman_numerals(per_doc: list[dict]) -> dict:
    from picarones.measurements.roman_numerals import ALL_STATUSES, VALUE_PRESERVING_STATUSES

    n_total = 0
    per_status: dict[str, int] = {s: 0 for s in ALL_STATUSES}
    for d in per_doc:
        n_total += d.get("n_numerals_reference", 0)
        for status, count in d.get("per_status", {}).items():
            per_status[status] = per_status.get(status, 0) + count
    n_strict = per_status.get("strict_preserved", 0)
    n_value = sum(per_status.get(s, 0) for s in VALUE_PRESERVING_STATUSES)
    return {
        "n_numerals_reference": n_total,
        "n_strict_preserved": n_strict,
        "n_value_preserved": n_value,
        "global_strict_score": n_strict / n_total if n_total > 0 else 0.0,
        "global_value_score": n_value / n_total if n_total > 0 else 0.0,
        "per_status": per_status,
        "doc_count": len(per_doc),
    }


_AGGREGATORS = {
    "unicode_blocks":   _aggregate_unicode,
    "abbreviations":    _aggregate_abbreviations,
    "mufi":             _aggregate_mufi,
    "early_modern":     _aggregate_early_modern,
    "modern_archives":  _aggregate_modern_archives,
    "roman_numerals":   _aggregate_roman_numerals,
}


def aggregate_philological_metrics(
    doc_metrics: list[Optional[dict]],
) -> Optional[dict]:
    """Agrège les ``philological_metrics`` per-document en un dict
    corpus-wide par module.

    Pour chaque module, on agrège uniquement les documents qui ont
    eu du signal pour ce module.  Si aucun module n'a été calculé
    sur aucun document, retourne ``None``.
    """
    by_module: dict[str, list[dict]] = {}
    for doc in doc_metrics:
        if not doc:
            continue
        for module, payload in doc.items():
            by_module.setdefault(module, []).append(payload)
    if not by_module:
        return None
    out: dict = {}
    for module, payloads in by_module.items():
        aggregator = _AGGREGATORS.get(module)
        if aggregator is None:  # pragma: no cover
            logger.warning(
                "[philological_hooks] aucun agrégateur pour %s", module,
            )
            continue
        out[module] = aggregator(payloads)
    return out if out else None


__all__ = [
    "compute_philological_metrics",
    "aggregate_philological_metrics",
]
