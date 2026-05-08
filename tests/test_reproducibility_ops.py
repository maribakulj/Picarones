"""Tests Sprint A8 — reproductibilité opérationnelle.

Items M-1, M-2, M-12, M-18, m-11, m-13, m-14 de l'audit
``institutional-readiness-2026-05.md``.

Valide la chaîne **lock file + Docker digest + snapshots** comme
contrat opérationnel pour la reproductibilité institutionnelle.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# M-1 — Lock files
# ---------------------------------------------------------------------------


def test_runtime_lock_exists() -> None:
    """``requirements.lock`` doit exister à la racine et être un lock file
    valide (généré par uv pip compile)."""
    lock = REPO_ROOT / "requirements.lock"
    assert lock.exists(), "requirements.lock manquant — sortie de `uv pip compile`"
    text = lock.read_text(encoding="utf-8")
    assert "uv pip compile" in text or "==" in text, (
        "Le lock file ne semble pas avoir été généré par uv "
        "(en-tête `uv pip compile` absente)."
    )


def test_dev_lock_exists() -> None:
    """``requirements-dev.lock`` doit exister (extras dev + web)."""
    lock = REPO_ROOT / "requirements-dev.lock"
    assert lock.exists(), "requirements-dev.lock manquant"
    text = lock.read_text(encoding="utf-8")
    # Doit contenir au moins pytest et fastapi (extras dev + web).
    assert "pytest" in text
    assert "fastapi" in text


def test_runtime_lock_pins_versions() -> None:
    """Tout package du lock runtime doit être épinglé à une version
    exacte (``==``)."""
    lock = REPO_ROOT / "requirements.lock"
    lines = [
        line.strip()
        for line in lock.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith(" ")
    ]
    for line in lines:
        # Tolère ``-r`` et ``-c`` (références à d'autres lock files)
        if line.startswith(("-r", "-c", "-e")):
            continue
        # Format attendu : ``package==1.2.3`` ou ``package==1.2.3 ; marker``
        # (pas de ``>=``, ``<=``, ``~=`` qui sont des bornes non-épinglées).
        assert "==" in line, (
            f"Lock runtime non-épinglé : `{line}`. "
            "Régénérer avec `uv pip compile pyproject.toml -o requirements.lock`."
        )


# ---------------------------------------------------------------------------
# M-2 — Dockerfile pinning
# ---------------------------------------------------------------------------


def test_dockerfile_pins_python_patch() -> None:
    """Le Dockerfile doit utiliser un patch précis (3.11.x) plutôt
    que le stream ``3.11-slim``."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    # Cherche ARG PYTHON_BASE_IMAGE qui pointe vers une version patch
    m = re.search(
        r"ARG\s+PYTHON_BASE_IMAGE\s*=\s*python:(\d+\.\d+\.\d+)",
        dockerfile,
    )
    assert m, (
        "Dockerfile ne déclare pas ARG PYTHON_BASE_IMAGE=python:X.Y.Z-slim "
        "— rétrocompat avec ``python:3.11-slim`` (stream) introduit "
        "des builds non reproductibles."
    )


def test_dockerfile_uses_arg_in_from() -> None:
    """Les directives ``FROM`` doivent référencer ``${PYTHON_BASE_IMAGE}``."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    from_lines = [
        line for line in dockerfile.splitlines() if line.startswith("FROM")
    ]
    assert from_lines, "Aucune directive FROM trouvée dans le Dockerfile"
    for line in from_lines:
        assert "${PYTHON_BASE_IMAGE}" in line, (
            f"FROM non aligné sur ARG : `{line}`"
        )


# ---------------------------------------------------------------------------
# M-18 — .dockerignore + .env.example
# ---------------------------------------------------------------------------


def test_dockerignore_exists_and_excludes_git() -> None:
    """``.dockerignore`` doit exister et exclure au moins ``.git``,
    ``tests/``, ``__pycache__``."""
    f = REPO_ROOT / ".dockerignore"
    assert f.exists(), ".dockerignore manquant à la racine"
    content = f.read_text(encoding="utf-8")
    for required in [".git", "tests", "__pycache__", ".venv", ".pytest_cache"]:
        assert required in content, (
            f".dockerignore n'exclut pas `{required}` — build context inutilement gros."
        )


def test_env_example_exists_and_documents_keys() -> None:
    """``.env.example`` doit exister et lister les variables clés."""
    f = REPO_ROOT / ".env.example"
    assert f.exists(), ".env.example manquant"
    content = f.read_text(encoding="utf-8")
    required_vars = [
        "PICARONES_PUBLIC_MODE",
        "PICARONES_CSRF_REQUIRED",
        "PICARONES_CSRF_SECRET",
        "PICARONES_BROWSE_ROOTS",
        "PICARONES_MAX_UPLOAD_MB",
        "PICARONES_PORT",
    ]
    missing = [v for v in required_vars if v not in content]
    assert not missing, (
        f"Variables manquantes dans .env.example : {missing}"
    )


# ---------------------------------------------------------------------------
# M-12 — Doc snapshots
# ---------------------------------------------------------------------------


def test_reproducibility_snapshots_doc_exists() -> None:
    """``docs/reference/reproducibility-snapshots.md`` doit exister et
    documenter la procédure end-to-end (S60 — Diataxis)."""
    f = REPO_ROOT / "docs" / "reference" / "reproducibility-snapshots.md"
    assert f.exists()
    text = f.read_text(encoding="utf-8")
    # Sections clés attendues.
    for section in [
        "Pourquoi des snapshots",
        "Ce qu'un snapshot contient",
        "Comment rejouer un benchmark",
        "Limites assumées",
    ]:
        assert section in text, f"Section manquante : {section!r}"


# ---------------------------------------------------------------------------
# m-11 — Versionnement testdata
# ---------------------------------------------------------------------------


def test_testdata_version_yaml_exists() -> None:
    """``tests/.testdata/VERSION.yaml`` doit exister et lister les
    fixtures versionnables."""
    f = REPO_ROOT / "tests" / ".testdata" / "VERSION.yaml"
    assert f.exists(), "VERSION.yaml manquant pour le versionnement testdata"
    data = yaml.safe_load(f.read_text(encoding="utf-8"))
    assert data is not None
    assert "version" in data
    assert "corpus_de_reference" in data, (
        "Le corpus de référence Sprint A5 doit être listé."
    )


def test_testdata_paths_exist() -> None:
    """Toute fixture listée dans VERSION.yaml doit pointer vers un
    fichier existant."""
    f = REPO_ROOT / "tests" / ".testdata" / "VERSION.yaml"
    if not f.exists():
        pytest.skip("VERSION.yaml absent")
    data = yaml.safe_load(f.read_text(encoding="utf-8"))
    missing: list[str] = []
    for entry in data.get("corpus_de_reference") or []:
        path = REPO_ROOT / entry["path"]
        if not path.exists():
            missing.append(entry["path"])
    assert not missing, (
        f"Fixtures listées mais absentes du repo : {missing}"
    )


# ---------------------------------------------------------------------------
# m-13 — requirements.txt aligné
# ---------------------------------------------------------------------------


def test_requirements_txt_points_to_lock() -> None:
    """``requirements.txt`` ne doit plus contenir de bornes ``>=``
    individuelles — c'est un alias vers ``requirements.lock``."""
    f = REPO_ROOT / "requirements.txt"
    assert f.exists()
    content = f.read_text(encoding="utf-8")
    # Doit contenir la directive ``-r requirements.lock``
    assert "-r requirements.lock" in content, (
        "requirements.txt doit pointer vers requirements.lock via `-r`."
    )
    # Tolérance : autoriser quelques commentaires explicatifs mais pas
    # de spec ``package>=version`` indépendante du lock.
    spec_lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
        and not line.startswith("#")
        and not line.startswith("-")
    ]
    assert not spec_lines, (
        f"Lignes de spec non-locked dans requirements.txt : {spec_lines}"
    )


# ---------------------------------------------------------------------------
# m-14 — Pricing staleness
# ---------------------------------------------------------------------------


def test_pricing_yaml_has_valid_until() -> None:
    """``picarones/data/pricing.yaml`` doit déclarer ``meta.valid_until``."""
    f = REPO_ROOT / "picarones" / "data" / "pricing.yaml"
    data = yaml.safe_load(f.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    assert "valid_until" in meta, (
        "pricing.yaml doit déclarer `meta.valid_until` (Sprint A8 m-14)"
    )
    # Doit être au format ISO date YYYY-MM-DD
    assert re.match(r"\d{4}-\d{2}-\d{2}", str(meta["valid_until"])), (
        f"valid_until mal formé : {meta['valid_until']!r}"
    )


def test_pricing_staleness_detector_registered() -> None:
    """Le détecteur ``detect_pricing_staleness`` doit être enregistré."""
    # Trigger registration
    import picarones.evaluation.metrics  # noqa: F401
    from picarones.domain.facts import FactType
    from picarones.reports_v2.narrative.detectors import DETECTORS_BY_TYPE

    assert FactType.PRICING_STALENESS_WARNING in DETECTORS_BY_TYPE


def test_pricing_staleness_detector_silent_when_valid() -> None:
    """Le détecteur reste silencieux si ``today <= valid_until``."""
    from datetime import date, timedelta

    from picarones.reports_v2.narrative.detectors.pareto import (
        detect_pricing_staleness,
    )

    future = (date.today() + timedelta(days=180)).isoformat()
    benchmark_data = {
        "snapshots": {"pricing": {"meta": {"valid_until": future}}}
    }
    facts = detect_pricing_staleness(benchmark_data)
    assert facts == []


def test_pricing_staleness_detector_fires_when_expired() -> None:
    """Le détecteur émet un Fact si la date est dépassée."""
    from datetime import date, timedelta

    from picarones.domain.facts import FactType
    from picarones.reports_v2.narrative.detectors.pareto import (
        detect_pricing_staleness,
    )

    past = (date.today() - timedelta(days=30)).isoformat()
    benchmark_data = {
        "snapshots": {"pricing": {"meta": {"valid_until": past}}}
    }
    facts = detect_pricing_staleness(benchmark_data)
    assert len(facts) == 1
    assert facts[0].type == FactType.PRICING_STALENESS_WARNING
    assert facts[0].payload["days_overdue"] == 30


def test_pricing_staleness_detector_silent_on_missing_data() -> None:
    """Le détecteur ne crashe pas si la clé est absente."""
    from picarones.reports_v2.narrative.detectors.pareto import (
        detect_pricing_staleness,
    )

    # Pas de snapshots
    assert detect_pricing_staleness({}) == []
    # Snapshots vides
    assert detect_pricing_staleness({"snapshots": {}}) == []
    # valid_until mal formé
    bad = {"snapshots": {"pricing": {"meta": {"valid_until": "not-a-date"}}}}
    assert detect_pricing_staleness(bad) == []
