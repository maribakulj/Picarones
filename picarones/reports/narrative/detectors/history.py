"""Détecteurs narratifs liés à *l'historique SQLite + multi-runs* (chantier 5).

3 détecteurs déplacés depuis ``narrative/detectors.py`` :

func:`detect_engine_off_baseline`
func:`detect_engine_unstable`
func:`detect_regression_in_history`
"""

from __future__ import annotations


from picarones.domain.facts import Fact, FactImportance, FactType
from picarones.reports.narrative.registry import register_detector


@register_detector(
    FactType.ENGINE_OFF_BASELINE,
    priority=150,
    importance=FactImportance.MEDIUM,
)
def detect_engine_off_baseline(benchmark_data: dict) -> list[Fact]:
    """Émet un Fact pour chaque moteur dont le CER courant s'écarte
    significativement de sa moyenne historique sur le **même corpus**.

    Lit ``benchmark_data["baseline_comparisons"]`` (liste de dicts
    produits par ``compute_engine_baseline`` du module
    ``baseline_comparison`` Sprint 73).  Si la clé est absente ou
    vide, le détecteur reste silencieux — typiquement le cas quand
    aucun historique SQLite n'a été chargé.

    Garde-fous :

    - Si ``n_runs < 5`` (déjà filtré par ``compute_engine_baseline``
      qui retourne ``None`` dans ce cas).
    - Si ``relative_delta`` n'est pas calculable (baseline = 0).
    - Importance ``HIGH`` si ``|relative_delta| ≥ 50 %``, sinon
      ``MEDIUM``.
    """
    comparisons = benchmark_data.get("baseline_comparisons") or []
    if not isinstance(comparisons, (list, tuple)):
        return []
    facts: list[Fact] = []
    for comp in comparisons:
        if not isinstance(comp, dict):
            continue
        if not comp.get("off_baseline"):
            continue
        rel = comp.get("relative_delta")
        if rel is None:
            continue
        engine = comp.get("engine_name")
        cer_current = comp.get("cer_current")
        cer_hist_mean = comp.get("cer_historical_mean")
        n_runs = comp.get("n_runs")
        if engine is None or cer_current is None or cer_hist_mean is None:
            continue
        importance = (
            FactImportance.HIGH if abs(float(rel)) >= 0.50
            else FactImportance.MEDIUM
        )
        facts.append(Fact(
            type=FactType.ENGINE_OFF_BASELINE,
            importance=importance,
            payload={
                "engine": engine,
                "cer_current_pct": round(float(cer_current) * 100, 2),
                "cer_historical_mean_pct": round(
                    float(cer_hist_mean) * 100, 2,
                ),
                "n_runs": int(n_runs or 0),
                "relative_delta_pct": round(float(rel) * 100, 1),
                "direction": "higher" if float(rel) > 0 else "lower",
            },
            engines_involved=(engine,),
        ))
    return facts


@register_detector(
    FactType.ENGINE_UNSTABLE,
    priority=160,
    importance=FactImportance.HIGH,
)
def detect_engine_unstable(benchmark_data: dict) -> list[Fact]:
    """Émet un Fact pour chaque moteur dont la stabilité multi-runs
    est insuffisante (Sprint 83 + 90).

    Lit ``benchmark_data["multirun_stability"]`` : liste de dicts
    avec ``engine_name`` + champs de ``compute_multirun_stability``
    (cer_cv, identical_run_rate, n_runs, etc.).  Si la clé est
    absente ou vide, le détecteur reste silencieux — typiquement
    le cas quand l'utilisateur n'a pas exécuté `--repeats N`.

    Garde-fous :

    - ``n_runs ≥ 2`` (déjà filtré par
      ``compute_multirun_stability`` qui retourne ``None``).
    - Déclenche si ``cer_cv > 0.10`` (variance relative > 10 % du
      CER moyen) **ou** ``identical_run_rate < 0.50`` (moins
      d'une paire de runs sur deux est identique).
    - Importance ``HIGH`` (l'instabilité discrédite les
      conclusions).
    """
    stabilities = benchmark_data.get("multirun_stability") or []
    if not isinstance(stabilities, (list, tuple)):
        return []
    facts: list[Fact] = []
    for stab in stabilities:
        if not isinstance(stab, dict):
            continue
        engine = stab.get("engine_name") or stab.get("engine")
        if not engine:
            continue
        n_runs = stab.get("n_runs")
        if not isinstance(n_runs, int) or n_runs < 2:
            continue
        cer_cv = stab.get("cer_cv")
        identical_rate = stab.get("identical_run_rate")
        # Critères de déclenchement
        cv_high = (
            isinstance(cer_cv, (int, float)) and float(cer_cv) > 0.10
        )
        runs_diverge = (
            isinstance(identical_rate, (int, float))
            and float(identical_rate) < 0.50
        )
        if not (cv_high or runs_diverge):
            continue
        payload: dict = {
            "engine": engine,
            "n_runs": int(n_runs),
        }
        if isinstance(cer_cv, (int, float)):
            payload["cer_cv"] = float(cer_cv)
            payload["cer_cv_pct"] = round(float(cer_cv) * 100, 1)
        if isinstance(identical_rate, (int, float)):
            payload["identical_run_rate"] = float(identical_rate)
            payload["identical_run_rate_pct"] = round(
                float(identical_rate) * 100, 1,
            )
        # Champs additionnels pour la phrase de synthèse
        cer_mean = stab.get("cer_mean")
        cer_stdev = stab.get("cer_stdev")
        if isinstance(cer_mean, (int, float)):
            payload["cer_mean_pct"] = round(float(cer_mean) * 100, 2)
        if isinstance(cer_stdev, (int, float)):
            payload["cer_stdev_pct"] = round(float(cer_stdev) * 100, 2)
        n_distinct = stab.get("n_distinct_outputs")
        if isinstance(n_distinct, int):
            payload["n_distinct_outputs"] = int(n_distinct)
        facts.append(Fact(
            type=FactType.ENGINE_UNSTABLE,
            importance=FactImportance.HIGH,
            payload=payload,
            engines_involved=(engine,),
        ))
    return facts


@register_detector(
    FactType.REGRESSION_IN_HISTORY,
    priority=170,
    importance=FactImportance.MEDIUM,
)
def detect_regression_in_history(benchmark_data: dict) -> list[Fact]:
    """Émet un Fact pour chaque moteur dont l'historique montre
    une dégradation : pente positive significative ou rupture
    brutale

    Lit ``benchmark_data["longitudinal_trends"]`` : liste de
    dicts produits par ``compute_corpus_longitudinal`` du module
    ``longitudinal``.  Si la clé est absente ou vide, le
    détecteur reste silencieux — typiquement le cas quand
    aucun historique n'a été chargé ou que la série est trop
    courte.

    Garde-fous :

    - ``n_runs ≥ 3`` (déjà filtré par
      ``compute_engine_longitudinal``).
    - Déclenche si **soit** ``trend.slope`` traduit une
      régression d'au moins ``slope_threshold`` (en CER/jour,
      défaut équivalent à +1 point CER sur 365 jours), **soit**
      ``change_point.delta > change_threshold`` (défaut
      0.01 = +1 point de CER d'un segment à l'autre).
    - Importance ``HIGH`` si la dégradation cumulée
      ``absolute_delta`` ≥ 5 points de CER.
    """
    trends = benchmark_data.get("longitudinal_trends") or []
    if not isinstance(trends, (list, tuple)):
        return []
    slope_threshold = (
        0.01 / 365.0  # +1 point de CER sur 365 jours minimum
    )
    change_threshold = 0.01
    facts: list[Fact] = []
    for entry in trends:
        if not isinstance(entry, dict):
            continue
        engine = entry.get("engine_name")
        if not engine:
            continue
        n_runs = entry.get("n_runs")
        if not isinstance(n_runs, int) or n_runs < 3:
            continue
        trend = entry.get("trend") or {}
        cp = entry.get("change_point")
        slope = trend.get("slope")
        slope_high = (
            isinstance(slope, (int, float))
            and float(slope) > slope_threshold
        )
        cp_high = (
            isinstance(cp, dict)
            and isinstance(cp.get("delta"), (int, float))
            and float(cp["delta"]) > change_threshold
        )
        if not (slope_high or cp_high):
            continue
        absolute_delta = entry.get("absolute_delta") or 0.0
        importance = (
            FactImportance.HIGH
            if isinstance(absolute_delta, (int, float))
            and abs(float(absolute_delta)) >= 0.05
            else FactImportance.MEDIUM
        )
        payload: dict = {
            "engine": engine,
            "n_runs": int(n_runs),
            "absolute_delta_pct": round(
                float(absolute_delta) * 100, 2,
            ) if isinstance(absolute_delta, (int, float)) else 0.0,
            "first_cer_pct": round(
                float(entry.get("first_cer") or 0.0) * 100, 2,
            ),
            "last_cer_pct": round(
                float(entry.get("last_cer") or 0.0) * 100, 2,
            ),
        }
        if slope_high:
            payload["slope_per_year_pct"] = round(
                float(slope) * 365 * 100, 2,
            )
            payload["r_squared"] = round(
                float(trend.get("r_squared") or 0.0), 3,
            )
            payload["pattern"] = "trend"
        if cp_high:
            payload["change_point_timestamp"] = str(
                cp.get("timestamp") or "?",
            )
            payload["change_delta_pct"] = round(
                float(cp["delta"]) * 100, 2,
            )
            payload["mean_before_pct"] = round(
                float(cp.get("mean_before") or 0.0) * 100, 2,
            )
            payload["mean_after_pct"] = round(
                float(cp.get("mean_after") or 0.0) * 100, 2,
            )
            # Si on a aussi une rupture, le pattern domine
            payload["pattern"] = (
                "trend_and_change_point" if slope_high else "change_point"
            )
        facts.append(Fact(
            type=FactType.REGRESSION_IN_HISTORY,
            importance=importance,
            payload=payload,
            engines_involved=(engine,),
        ))
    return facts


# ---------------------------------------------------------------------------
# Sprint A3 (item B-3) — détecteur IMPORTER_FALLBACK_TRIGGERED
# ---------------------------------------------------------------------------


@register_detector(
    FactType.IMPORTER_FALLBACK_TRIGGERED,
    # Priorité 180 — en queue, après les détecteurs de tendance historique.
    # L'incident d'importer est *informationnel sur l'acquisition*, pas
    # sur le ranking ou la performance d'un moteur — il vient logiquement
    # après tout le reste de la synthèse.
    priority=180,
    importance=FactImportance.MEDIUM,
)
def detect_importer_fallback(benchmark_data: dict) -> list[Fact]:
    """Émet un Fact par incident d'importer en mode dégradé.

    Lit ``benchmark_data["importer_fallbacks"]`` (liste de dicts
    produite par ``picarones.adapters.corpus._fallback_log.consume_fallback_log()``).
    Si la clé est absente ou vide, le détecteur reste silencieux —
    typiquement le cas pour un benchmark qui n'utilise pas d'importer
    distant (corpus local).

    Importance HIGH si **plusieurs incidents** sur le même importer
    (signal d'une indisponibilité prolongée plutôt qu'un échec
    isolé) ; MEDIUM sinon.
    """
    fallbacks = benchmark_data.get("importer_fallbacks") or []
    if not fallbacks:
        return []

    # Compte par importer pour détecter les incidents répétés.
    counts: dict[str, int] = {}
    for entry in fallbacks:
        if isinstance(entry, dict):
            counts[str(entry.get("importer", "unknown"))] = (
                counts.get(str(entry.get("importer", "unknown")), 0) + 1
            )

    facts: list[Fact] = []
    for entry in fallbacks:
        if not isinstance(entry, dict):
            continue
        importer = str(entry.get("importer", "unknown"))
        operation = str(entry.get("operation", "unknown"))
        importance = (
            FactImportance.HIGH if counts.get(importer, 0) >= 2 else FactImportance.MEDIUM
        )
        payload: dict = {
            "importer": importer,
            "operation": operation,
            "incidents_for_importer": counts.get(importer, 1),
        }
        if entry.get("error"):
            payload["error_repr"] = str(entry["error"])
        facts.append(Fact(
            type=FactType.IMPORTER_FALLBACK_TRIGGERED,
            importance=importance,
            payload=payload,
            engines_involved=(),
        ))
    return facts
