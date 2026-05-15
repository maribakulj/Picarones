"""Modèle de données des métriques OCR/HTR (couche 3 — evaluation).

Abstractions pures pour représenter les métriques calculées sur
une paire (référence, hypothèse) — pas de dépendance externe (pas
de jiwer, pas de scipy).

Le calcul effectif via jiwer vit dans
:mod:`picarones.evaluation.metrics.text_metrics` (``compute_metrics``).
L'agrégation statistique vit ici car elle n'utilise que la stdlib
(``statistics``).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional


@dataclass
class MetricsResult:
    """Ensemble des métriques calculées pour une paire (référence, hypothèse).

    Sprint A14-S1 — A.I.0 P0 : les champs CER/WER/MER/WIL sont
    ``Optional[float]``.  Auparavant, en cas d'erreur de calcul (jiwer
    absent, exception levée), ces champs étaient remplis avec ``0.0``,
    ce qui était indistinguable d'un score parfait pour tout
    consommateur ne lisant pas systématiquement ``error``.  Désormais
    ils sont à ``None`` quand ``error`` est non-None — les agrégateurs
    filtrent déjà sur ``error is None``, les rendus HTML utilisent
    ``safe_round`` qui mappe ``None → 0.0`` à l'affichage seul, et un
    accès direct sans vérification d'erreur lève désormais un
    ``TypeError`` explicite plutôt que de retourner silencieusement
    une valeur factice.
    """

    cer: Optional[float] = None
    cer_nfc: Optional[float] = None
    cer_caseless: Optional[float] = None
    wer: Optional[float] = None
    wer_normalized: Optional[float] = None
    mer: Optional[float] = None
    wil: Optional[float] = None
    reference_length: int = 0
    hypothesis_length: int = 0
    error: Optional[str] = None
    # Audit scientifique (F1) — comptes bruts de l'alignement minimal
    # (jiwer/Levenshtein) nécessaires pour le CER/WER **micro-moyenné**
    # corpus-wide (Σ erreurs / Σ unités de référence), standard du domaine
    # OCR/HTR (ICDAR, OCR-D, HTR-United).  ``None`` si le calcul a échoué
    # ou pour les cas dégénérés (référence vide) où le dénominateur micro
    # n'est pas défini — l'agrégateur micro saute alors le document.
    cer_errors: Optional[int] = None
    """Distance d'édition caractère = substitutions + suppressions + insertions."""
    cer_ref_chars: Optional[int] = None
    """Longueur de référence en caractères = substitutions + suppressions + hits
    (dénominateur exact du CER, identique à celui utilisé par jiwer)."""
    wer_errors: Optional[int] = None
    """Distance d'édition mot = substitutions + suppressions + insertions."""
    wer_ref_words: Optional[int] = None
    """Nombre de mots de référence = substitutions + suppressions + hits."""
    cer_diplomatic: Optional[float] = None
    """CER calculé après normalisation diplomatique (ſ=s, u=v, i=j…).
    None si aucun profil diplomatique n'a été fourni à compute_metrics.
    """
    diplomatic_profile_name: Optional[str] = None
    """Nom du profil de normalisation diplomatique utilisé."""

    def as_dict(self) -> dict:
        def _round(v: Optional[float]) -> Optional[float]:
            return None if v is None else round(v, 6)
        d = {
            "cer": _round(self.cer),
            "cer_nfc": _round(self.cer_nfc),
            "cer_caseless": _round(self.cer_caseless),
            "wer": _round(self.wer),
            "wer_normalized": _round(self.wer_normalized),
            "mer": _round(self.mer),
            "wil": _round(self.wil),
            "reference_length": self.reference_length,
            "hypothesis_length": self.hypothesis_length,
            "error": self.error,
        }
        # Comptes bruts (F1) — sérialisés seulement s'ils sont présents
        # pour ne pas alourdir le JSON des cas dégénérés / en erreur.
        if self.cer_errors is not None and self.cer_ref_chars is not None:
            d["cer_errors"] = self.cer_errors
            d["cer_ref_chars"] = self.cer_ref_chars
        if self.wer_errors is not None and self.wer_ref_words is not None:
            d["wer_errors"] = self.wer_errors
            d["wer_ref_words"] = self.wer_ref_words
        if self.cer_diplomatic is not None:
            d["cer_diplomatic"] = round(self.cer_diplomatic, 6)
            d["diplomatic_profile_name"] = self.diplomatic_profile_name
        return d

    @property
    def cer_percent(self) -> Optional[float]:
        return None if self.cer is None else round(self.cer * 100, 2)

    @property
    def wer_percent(self) -> Optional[float]:
        return None if self.wer is None else round(self.wer * 100, 2)

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsResult":
        """Reconstruit depuis le dict produit par :meth:`as_dict`.

        Phase 2.2 du chantier post-rewrite : fidélité du round-trip
        ``as_dict → from_dict``.  Auparavant, ``ReportGenerator.from_json``
        contenait sa propre reconstruction partielle qui perdait
        ``cer_diplomatic`` et ``diplomatic_profile_name``.  Centraliser
        la désérialisation ici évite la dérive.
        """
        return cls(
            cer=data.get("cer"),
            cer_nfc=data.get("cer_nfc"),
            cer_caseless=data.get("cer_caseless"),
            wer=data.get("wer"),
            wer_normalized=data.get("wer_normalized"),
            mer=data.get("mer"),
            wil=data.get("wil"),
            reference_length=data.get("reference_length", 0),
            hypothesis_length=data.get("hypothesis_length", 0),
            error=data.get("error"),
            cer_errors=data.get("cer_errors"),
            cer_ref_chars=data.get("cer_ref_chars"),
            wer_errors=data.get("wer_errors"),
            wer_ref_words=data.get("wer_ref_words"),
            cer_diplomatic=data.get("cer_diplomatic"),
            diplomatic_profile_name=data.get("diplomatic_profile_name"),
        )


def aggregate_metrics(results: list[MetricsResult]) -> dict:
    """Calcule les statistiques agrégées sur un ensemble de résultats.

    Parameters
    ----------
    results:
        Liste de MetricsResult correspondant à plusieurs documents.

    Returns
    -------
    dict
        Statistiques : moyenne, médiane, min, max, std pour chaque métrique.
    """
    if not results:
        return {}

    def _stats(values: list[float]) -> dict:
        if not values:
            return {}
        return {
            "mean": round(statistics.mean(values), 6),
            "median": round(statistics.median(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        }

    metric_names = ["cer", "cer_nfc", "cer_caseless", "wer", "wer_normalized", "mer", "wil"]
    aggregated: dict = {}
    for metric in metric_names:
        # Sprint A14-S1 — défense en profondeur : double filtre.  Un
        # MetricsResult avec ``error`` doit avoir ses métriques à
        # ``None`` (cf. compute_metrics), mais on filtre aussi les
        # ``None`` directement au cas où un caller construirait un
        # MetricsResult partiel.
        values = [
            v for r in results
            if r.error is None
            for v in (getattr(r, metric),)
            if v is not None
        ]
        aggregated[metric] = _stats(values)

    # CER diplomatique (optionnel — présent seulement si calculé)
    diplo_values = [
        r.cer_diplomatic for r in results
        if r.error is None and r.cer_diplomatic is not None
    ]
    if diplo_values:
        aggregated["cer_diplomatic"] = _stats(diplo_values)
        # Nom du profil (même pour tous les docs d'un corpus)
        profile_name = next(
            (r.diplomatic_profile_name for r in results if r.diplomatic_profile_name),
            None,
        )
        if profile_name:
            aggregated["cer_diplomatic"]["profile"] = profile_name

    # ──────────────────────────────────────────────────────────────────
    # CER / WER **micro-moyennés** (audit scientifique F1)
    #
    # Standard du domaine OCR/HTR (ICDAR, OCR-D, HTR-United, Transkribus,
    # eScriptorium) : agréger les *comptes bruts* avant de diviser —
    #   CER_micro = Σ distance_édition / Σ caractères_référence
    # — et non moyenner des taux par document (macro), qui donne le même
    # poids à une légende de 10 caractères et à une page de 5 000.
    # Le micro-CER est la métrique corpus de référence ; mean/median
    # restent exposés ci-dessus comme diagnostics de dispersion.
    # ``None`` si aucun document n'a de comptes exploitables (cas d'un
    # jiwer absent ou de références toutes vides).
    def _micro(err_attr: str, ref_attr: str) -> Optional[dict]:
        total_err = 0
        total_ref = 0
        n_docs = 0
        for r in results:
            if r.error is not None:
                continue
            e = getattr(r, err_attr)
            d = getattr(r, ref_attr)
            if e is None or d is None:
                continue
            total_err += e
            total_ref += d
            n_docs += 1
        if n_docs == 0 or total_ref <= 0:
            return None
        return {
            "value": round(total_err / total_ref, 6),
            "total_errors": total_err,
            "total_reference_units": total_ref,
            "document_count": n_docs,
        }

    cer_micro = _micro("cer_errors", "cer_ref_chars")
    if cer_micro is not None:
        aggregated["cer_micro"] = cer_micro
    wer_micro = _micro("wer_errors", "wer_ref_words")
    if wer_micro is not None:
        aggregated["wer_micro"] = wer_micro

    aggregated["document_count"] = len(results)
    aggregated["failed_count"] = sum(1 for r in results if r.error is not None)

    return aggregated


__all__ = [
    "MetricsResult",
    "aggregate_metrics",
]
