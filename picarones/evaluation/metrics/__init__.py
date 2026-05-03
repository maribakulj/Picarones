"""Métriques — calculs purs sur des paires (référence, hypothèse).

Cible du Sprint S10 du rewrite : déplacement (sans modification de
logique) des ~40 modules de calcul pur depuis
``picarones.measurements`` :

- ``cer.py``, ``wer.py``, ``mer.py``, ``wil.py`` — métriques jiwer
- ``mufi.py`` — couverture MUFI
- ``abbreviations.py`` — Capelli + tilde
- ``unicode_blocks.py`` — fidélité par bloc Unicode
- ``early_modern.py``, ``modern_archives.py``, ``roman_numerals.py``
- ``ner.py``, ``reading_order.py``, ``layout.py``
- ``readability.py``, ``searchability.py``, ``numerical_sequences.py``
- ``calibration.py``, ``confusion.py``, ``taxonomy.py``
- ``inter_engine.py``, ``specialization.py``, ``error_absorption.py``
- ``robustness.py``, ``image_quality.py``, ``image_predictive.py``
- ``hallucination.py``, ``lexical_modernization.py``
- ``rare_tokens.py``, ``equivalence_profile.py``, ``baseline_comparison.py``
- ``levers.py``, ``longitudinal.py``, ``throughput.py``
- ``marginal_cost.py``, ``cost_projection.py``, ``incremental_comparison.py``
- ``module_policy.py``, ``worst_lines.py``
- sous-package ``statistics/`` (Wilcoxon, Friedman/Nemenyi, etc.)

Règle de migration (S10) : un fichier déplacé = un seul commit avec
uniquement le déplacement et les nouveaux imports.  La logique reste
identique.  Les tests existants doivent continuer à passer via
re-exports temporaires dans l'ancien emplacement.
"""

from __future__ import annotations

__all__: list[str] = []
