"""Modélisation des coûts — APIs cloud et temps d'inférence local.

Sert uniquement à la vue Pareto coût/qualité du rapport (Sprint 5).
Les prix sont indicatifs et vieillissent vite : voir ``picarones/data/pricing.yaml``
pour les hypothèses, dates et URLs de référence.

Conventions
-----------
- Unité monétaire : EUR (conversion indicative depuis USD quand applicable).
- Coût exprimé par **1 000 pages** traitées.
- Coût local = temps moyen d'inférence × taux horaire (paramétrable).
- Empreinte carbone optionnelle : kWh × intensité g CO₂/kWh du réseau
  d'exécution (mix France bas carbone par défaut pour le local,
  moyenne cloud hyperscaler pour les APIs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_PRICING_PATH = Path(__file__).parent.parent / "data" / "pricing.yaml"


@dataclass(frozen=True)
class PricingDefaults:
    """Valeurs par défaut du fichier de prix (section ``meta``)."""

    last_updated: Optional[str] = None
    currency: str = "EUR"
    hourly_rate_local_cpu_eur: float = 0.08
    hourly_rate_local_gpu_eur: float = 1.20
    grid_intensity_local: float = 58.0
    grid_intensity_cloud: float = 380.0


@dataclass
class EngineCost:
    """Coût estimé d'un moteur sur 1 000 pages, avec traçabilité des hypothèses.

    La représentation est immuable après construction : une fois que l'utilisateur
    a choisi un taux horaire local, toutes les instances partagent cette
    hypothèse par injection explicite dans ``build_costs_for_benchmark``.
    """

    engine_key: str
    """Nom ou modèle servant de clé dans la table (ex. ``"gpt-4o"``, ``"tesseract"``)."""

    type: str  # "local" | "cloud_api" | "unknown"

    cost_per_1k_pages_eur: Optional[float] = None
    """Coût par 1 000 pages en euros. ``None`` si les données sont insuffisantes."""

    currency: str = "EUR"

    # Source / date
    pricing_source_url: Optional[str] = None
    pricing_date: Optional[str] = None

    # Pour les APIs cloud : prix brut
    api_price_per_1k_pages: Optional[float] = None

    # Pour le local : temps d'inférence et taux horaire utilisés
    local_mean_seconds_per_page: Optional[float] = None
    hourly_rate_eur: Optional[float] = None

    # Empreinte carbone (estimation — étiquetée "expérimentale" dans le rapport)
    kwh_per_1k_pages: Optional[float] = None
    grid_intensity_g_co2_per_kwh: Optional[float] = None
    co2_per_1k_pages_g: Optional[float] = None

    notes: Optional[str] = None

    assumptions: list[str] = field(default_factory=list)
    """Liste d'hypothèses textuelles à afficher sous le graphique."""

    def as_dict(self) -> dict:
        return {
            "engine_key": self.engine_key,
            "type": self.type,
            "cost_per_1k_pages_eur": self.cost_per_1k_pages_eur,
            "currency": self.currency,
            "pricing_source_url": self.pricing_source_url,
            "pricing_date": self.pricing_date,
            "api_price_per_1k_pages": self.api_price_per_1k_pages,
            "local_mean_seconds_per_page": self.local_mean_seconds_per_page,
            "hourly_rate_eur": self.hourly_rate_eur,
            "kwh_per_1k_pages": self.kwh_per_1k_pages,
            "grid_intensity_g_co2_per_kwh": self.grid_intensity_g_co2_per_kwh,
            "co2_per_1k_pages_g": self.co2_per_1k_pages_g,
            "notes": self.notes,
            "assumptions": list(self.assumptions),
        }


def load_pricing_database(path: Optional[Path] = None) -> tuple[PricingDefaults, dict]:
    """Charge la table de prix YAML.

    Retourne ``(defaults, engines_table)`` où ``engines_table`` est un dict
    ``{engine_key: raw_entry}``.
    """
    path = Path(path) if path else _DEFAULT_PRICING_PATH
    if not path.exists():
        logger.warning("[pricing] fichier %s introuvable", path)
        return PricingDefaults(), {}
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as e:
        logger.warning("[pricing] échec parsing %s : %s", path, e)
        return PricingDefaults(), {}

    meta = data.get("meta", {}) or {}
    defaults = PricingDefaults(
        last_updated=meta.get("last_updated"),
        currency=meta.get("currency", "EUR"),
        hourly_rate_local_cpu_eur=float(meta.get("default_hourly_rate_local_cpu_eur", 0.08)),
        hourly_rate_local_gpu_eur=float(meta.get("default_hourly_rate_local_gpu_eur", 1.20)),
        grid_intensity_local=float(meta.get("default_grid_intensity_g_co2_per_kwh", 58.0)),
        grid_intensity_cloud=float(meta.get("cloud_grid_intensity_g_co2_per_kwh", 380.0)),
    )
    engines_table = data.get("engines", {}) or {}
    return defaults, engines_table


def _match_key(engine_name: str, llm_model: Optional[str], table: dict) -> Optional[str]:
    """Cherche la meilleure clé pour ce moteur dans la table.

    Stratégie : d'abord le nom du modèle LLM (pour les pipelines), puis le
    nom OCR, puis un match partiel (substring) comme filet de sécurité.
    """
    candidates = [llm_model, engine_name]
    for c in candidates:
        if c and c in table:
            return c
    # Matching partiel — utile pour "tesseract → gpt-4o" ou "gpt-4o-vision"
    for c in candidates:
        if not c:
            continue
        for key in table:
            if key in c:
                return key
    return None


def estimate_cost(
    engine_name: str,
    *,
    llm_model: Optional[str] = None,
    is_pipeline: bool = False,
    measured_seconds_per_page: Optional[float] = None,
    table: Optional[dict] = None,
    defaults: Optional[PricingDefaults] = None,
    hourly_rate_override_eur: Optional[float] = None,
) -> EngineCost:
    """Calcule le ``EngineCost`` pour un moteur donné.

    Parameters
    ----------
    engine_name:
        Nom public du moteur (ex. ``"tesseract"``, ``"tesseract → gpt-4o"``).
    llm_model:
        Si pipeline OCR+LLM, le modèle LLM utilisé — prioritaire pour la
        lookup car c'est lui qui domine le coût.
    is_pipeline:
        Indique un pipeline OCR+LLM (change la sémantique de lookup).
    measured_seconds_per_page:
        Temps moyen observé sur le benchmark courant. Remplace la valeur
        indicative de la table si fournie (plus fiable).
    table, defaults:
        Overrides pour tests ou usage institutionnel.
    hourly_rate_override_eur:
        Taux horaire à utiliser pour le calcul local (sinon valeur table
        ou défaut).
    """
    if table is None or defaults is None:
        _defaults, _table = load_pricing_database()
        defaults = defaults or _defaults
        table = table or _table

    key = _match_key(engine_name, llm_model if is_pipeline else None, table)
    if key is None:
        return EngineCost(
            engine_key=engine_name,
            type="unknown",
            assumptions=["Aucune entrée dans la table de prix pour ce moteur."],
        )

    entry = table[key]
    etype = str(entry.get("type", "unknown"))
    notes = entry.get("notes")
    assumptions: list[str] = []
    currency = defaults.currency

    cost_eur: Optional[float] = None
    api_price: Optional[float] = None
    local_seconds = measured_seconds_per_page
    hourly_rate = None

    if etype == "cloud_api":
        api_price = entry.get("api_price_per_1k_pages")
        if api_price is not None:
            cost_eur = float(api_price)
            assumptions.append(
                f"Prix API indicatif : {cost_eur:.2f} €/1000 pages "
                f"(source : {entry.get('pricing_source_url', '—')}, {entry.get('pricing_date', 'date inconnue')})."
            )
    elif etype == "local":
        indicative_seconds = entry.get("local_mean_seconds_per_page")
        if local_seconds is None and indicative_seconds is not None:
            local_seconds = float(indicative_seconds)
            assumptions.append(
                f"Temps d'inférence indicatif : {local_seconds:.1f} s/page (non mesuré sur ce benchmark)."
            )
        elif local_seconds is not None:
            assumptions.append(
                f"Temps d'inférence mesuré : {local_seconds:.1f} s/page (moyenne sur le corpus)."
            )

        hourly_rate = (
            hourly_rate_override_eur
            if hourly_rate_override_eur is not None
            else entry.get("hourly_rate_override_eur")
        )
        if hourly_rate is None:
            # Heuristique : si l'entrée précise un override GPU, sinon CPU
            hourly_rate = (
                defaults.hourly_rate_local_gpu_eur
                if "gpu" in str(notes or "").lower()
                else defaults.hourly_rate_local_cpu_eur
            )
        hourly_rate = float(hourly_rate)

        if local_seconds is not None and hourly_rate is not None:
            cost_eur = (local_seconds / 3600.0) * hourly_rate * 1000.0
            assumptions.append(
                f"Taux horaire appliqué : {hourly_rate:.2f} €/h "
                f"(défaut {'GPU' if hourly_rate >= 0.5 else 'CPU'})."
            )

    # Empreinte carbone optionnelle
    kwh_1k = entry.get("kwh_per_1k_pages")
    grid = (
        entry.get("grid_intensity_g_co2_per_kwh")
        or (defaults.grid_intensity_cloud if etype == "cloud_api" else defaults.grid_intensity_local)
    )
    co2_g = None
    if kwh_1k is not None and grid is not None:
        co2_g = float(kwh_1k) * float(grid)

    return EngineCost(
        engine_key=key,
        type=etype,
        cost_per_1k_pages_eur=cost_eur,
        currency=currency,
        pricing_source_url=entry.get("pricing_source_url"),
        pricing_date=entry.get("pricing_date"),
        api_price_per_1k_pages=api_price,
        local_mean_seconds_per_page=local_seconds,
        hourly_rate_eur=hourly_rate,
        kwh_per_1k_pages=float(kwh_1k) if kwh_1k is not None else None,
        grid_intensity_g_co2_per_kwh=float(grid) if grid is not None else None,
        co2_per_1k_pages_g=co2_g,
        notes=notes,
        assumptions=assumptions,
    )


def build_costs_for_benchmark(
    engines_summary: list[dict],
    durations_by_engine: dict[str, float],
    *,
    hourly_rate_local_eur: Optional[float] = None,
    pricing_path: Optional[Path] = None,
) -> dict[str, dict]:
    """Calcule le coût de chaque moteur d'un benchmark.

    Returns
    -------
    dict ``{engine_name: EngineCost.as_dict()}``.
    """
    defaults, table = load_pricing_database(pricing_path)
    out: dict[str, dict] = {}
    for e in engines_summary:
        name = e.get("name")
        if not name:
            continue
        measured = durations_by_engine.get(name)
        llm_model = None
        pipeline_info = e.get("pipeline_info") or {}
        if pipeline_info:
            llm_model = pipeline_info.get("llm_model")
        cost = estimate_cost(
            engine_name=name,
            llm_model=llm_model,
            is_pipeline=bool(e.get("is_pipeline")),
            measured_seconds_per_page=measured,
            table=table,
            defaults=defaults,
            hourly_rate_override_eur=hourly_rate_local_eur,
        )
        out[name] = cost.as_dict()
    return out
