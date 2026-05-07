"""Snapshots de reproductibilité pour le rapport HTML (Sprint 27).

Le rapport HTML auto-contenu doit pouvoir être *rejoué* sans avoir
accès au code source du moment où il a été généré : un lecteur en
2026 doit pouvoir comprendre exactement quelle table de prix, quelle
définition de métrique, quel profil de normalisation, et quelle
version de Picarones ont produit les chiffres affichés.

Avant le Sprint 27, le rapport intégrait uniquement
``pareto.pricing_meta.last_updated`` — une simple date de mise à jour
qui ne disait rien sur le contenu de la table. Si quelqu'un modifiait
``picarones/data/pricing.yaml`` après génération, il était impossible
de reconstituer ce qu'avait vu le lecteur du rapport.

Quatre snapshots sont produits par ce module et embarqués dans
``report_data.snapshots`` :

- ``pricing``       — YAML brut intégral de la table de prix.
- ``glossary``      — entrées du glossaire pour la langue du rapport.
- ``normalization`` — profil de normalisation effectivement appliqué.
- ``environment``   — version Picarones, Python, plateforme, commit git
                      si dispo, liste figée des dépendances installées.

Garanties
---------
- **Déterminisme** : sur entrées identiques, ``snapshot_all()`` produit
  un dict bit-à-bit identique. Les listes sont triées, les timestamps
  sont absents.
- **Pas d'effet de bord** : le module ne modifie aucun état global ;
  les chemins YAML sont uniquement lus, jamais écrits.
- **Dégradé non bloquant** : si pyyaml est absent, si ``pricing.yaml``
  n'existe pas, si git n'est pas installé, le snapshot retourne un
  dict ``{"available": False, "reason": "..."}`` plutôt que de lever.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, Optional

from picarones import __version__

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing snapshot
# ---------------------------------------------------------------------------

def pricing_snapshot(pricing_path: Optional[Path] = None) -> dict[str, Any]:
    """Retourne le YAML brut + dict parsé de la table de prix utilisée.

    Si ``pricing_path`` n'est pas fourni, utilise le chemin par défaut
    de ``picarones.measurements.pricing._DEFAULT_PRICING_PATH``.
    """
    if pricing_path is None:
        try:
            from picarones.measurements.pricing import _DEFAULT_PRICING_PATH
            pricing_path = _DEFAULT_PRICING_PATH
        except ImportError:
            return {"available": False, "reason": "module pricing introuvable"}

    pricing_path = Path(pricing_path)
    if not pricing_path.exists():
        return {
            "available": False,
            "reason": f"pricing.yaml introuvable : {pricing_path}",
            "expected_path": str(pricing_path),
        }

    try:
        raw = pricing_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "available": False,
            "reason": f"lecture impossible : {exc}",
            "expected_path": str(pricing_path),
        }

    try:
        import yaml
        data = yaml.safe_load(raw) or {}
    except (ImportError, Exception) as exc:
        # Pas de yaml ou parsing en échec — on garde le brut quand même.
        logger.warning("[snapshot] parsing pricing.yaml échoué : %s", exc)
        data = {}

    return {
        "available": True,
        "source_path": str(pricing_path),
        "filename": pricing_path.name,
        "size_bytes": len(raw.encode("utf-8")),
        "raw_yaml": raw,
        "data": data,
    }


# ---------------------------------------------------------------------------
# Glossary snapshot
# ---------------------------------------------------------------------------

def glossary_snapshot(
    lang: str = "fr",
    used_keys: Optional[list[str] | set[str]] = None,
) -> dict[str, Any]:
    """Retourne les entrées du glossaire qui figurent dans le rapport.

    ``used_keys`` permet de ne snapshotter que les termes effectivement
    référencés (réduit la taille). ``None`` → toutes les entrées de la
    langue (mode conservateur).
    """
    try:
        from picarones.reports_v2.glossary import load_glossary, SUPPORTED_LANGS
    except ImportError:
        return {"available": False, "reason": "module glossary introuvable"}

    full = load_glossary(lang) or {}
    if not full:
        return {
            "available": False,
            "reason": f"aucune entrée pour lang={lang!r}",
            "supported_langs": SUPPORTED_LANGS,
        }

    if used_keys is not None:
        keys = set(used_keys)
        entries = {k: v for k, v in full.items() if k in keys}
    else:
        entries = dict(full)

    # Tri pour reproductibilité bit-à-bit.
    entries_sorted = {k: entries[k] for k in sorted(entries)}

    return {
        "available": True,
        "lang": lang,
        "entry_count": len(entries_sorted),
        "entries": entries_sorted,
    }


# ---------------------------------------------------------------------------
# Normalization profile snapshot
# ---------------------------------------------------------------------------

def normalization_snapshot(profile: Any) -> dict[str, Any]:
    """Sérialise un ``NormalizationProfile``.

    Couvre les profils built-in (``medieval_french``, ``nfc``, …) et les
    profils custom YAML chargés au runtime — l'objectif est qu'un
    lecteur du rapport puisse régénérer exactement la même
    normalisation à partir de ce snapshot.
    """
    if profile is None:
        return {"available": False, "reason": "aucun profil fourni"}

    # NormalizationProfile est un dataclass — on accède aux champs par
    # nom plutôt que via ``asdict`` pour bien contrôler le format.
    try:
        return {
            "available": True,
            "name": getattr(profile, "name", "unknown"),
            "nfc": bool(getattr(profile, "nfc", True)),
            "caseless": bool(getattr(profile, "caseless", False)),
            "diplomatic_table": dict(getattr(profile, "diplomatic_table", {}) or {}),
            "exclude_chars": sorted(getattr(profile, "exclude_chars", set()) or set()),
            "description": getattr(profile, "description", ""),
        }
    except Exception as exc:
        return {"available": False, "reason": f"sérialisation échouée : {exc}"}


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------

def _git_commit(repo_path: Optional[Path] = None) -> Optional[str]:
    """Retourne le commit git court (12 chars) si on est dans un repo, sinon None."""
    cwd = repo_path or Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        return out[:12] if out else None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _installed_packages(limit: int = 200) -> list[str]:
    """Liste figée des paquets installés au format ``name==version``.

    Triée par nom (case-insensitive) pour reproductibilité. Cappée à
    ``limit`` paquets pour ne pas exploser le poids du rapport.
    """
    try:
        pkgs: list[str] = []
        seen: set[str] = set()
        for d in distributions():
            try:
                name = (d.metadata.get("Name") or "").strip()
                version = (d.version or "").strip()
            except Exception:
                continue
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            pkgs.append(f"{name}=={version}")
        pkgs.sort(key=str.lower)
        return pkgs[:limit]
    except Exception as exc:  # pragma: no cover — défense en profondeur
        logger.warning("[snapshot] enum dépendances échoué : %s", exc)
        return []


def environment_snapshot(repo_path: Optional[Path] = None) -> dict[str, Any]:
    """Retourne version Picarones, Python, plateforme, commit, deps figées."""
    return {
        "available": True,
        "picarones_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": _git_commit(repo_path),
        "installed_packages": _installed_packages(),
    }


# ---------------------------------------------------------------------------
# API agrégée
# ---------------------------------------------------------------------------

def snapshot_all(
    *,
    lang: str = "fr",
    glossary_used_keys: Optional[list[str] | set[str]] = None,
    pricing_path: Optional[Path] = None,
    normalization_profile: Any = None,
    repo_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Construit le bloc ``snapshots`` à embarquer dans ``report_data``."""
    return {
        "pricing": pricing_snapshot(pricing_path=pricing_path),
        "glossary": glossary_snapshot(lang=lang, used_keys=glossary_used_keys),
        "normalization": normalization_snapshot(normalization_profile),
        "environment": environment_snapshot(repo_path=repo_path),
        "schema_version": 1,
    }


__all__ = [
    "pricing_snapshot",
    "glossary_snapshot",
    "normalization_snapshot",
    "environment_snapshot",
    "snapshot_all",
]
