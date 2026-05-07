"""Fixtures partagées du harness de régression.

Trois axes :

1. **Corpus de référence** : 3 tailles (small / medium / large) ;
   les images sont générées synthétiquement à la première
   utilisation pour rester reproductibles cross-OS sans déposer de
   blob binaire dans git.
2. **Golden snapshots** : sortie capturée du legacy, mise en cache
   sous ``golden/<phase>/<corpus>/<module>.<ext>``.  Régénérée à
   l'usage avec ``pytest --regen-golden``.
3. **Comparateurs** : helpers d'égalité bit-for-bit, sémantique
   HTML, ensemble de Facts.  Vivent dans ``_helpers/``.

Le harness est exclu du run pytest par défaut via le marker
``regression`` (cf. ``pyproject.toml``) — il s'exécute en CI
dédié pour ne pas ralentir la boucle de dev locale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pytest

HARNESS_ROOT = Path(__file__).resolve().parent
CORPORA_DIR = HARNESS_ROOT / "corpora"
GOLDEN_DIR = HARNESS_ROOT / "golden"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Ajoute ``--regen-golden`` pour régénérer les snapshots."""
    parser.addoption(
        "--regen-golden",
        action="store_true",
        default=False,
        help=(
            "Régénère les golden snapshots du harness de régression "
            "depuis l'état legacy actuel.  À utiliser quand on accepte "
            "explicitement une régression intentionnelle (cf. "
            "docs/migration/regression-tolerances.md)."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Enregistre le marker ``regression``."""
    config.addinivalue_line(
        "markers",
        "regression: tests de régression legacy ↔ rewrite ; exclus "
        "par défaut, opt-in via ``pytest -m regression``.",
    )


# ──────────────────────────────────────────────────────────────────
# Corpus
# ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def small_corpus_dir() -> Path:
    """Corpus *small* : 3 documents synthétiques.

    Génération unique à la première utilisation par session.  Les
    images sont des PNG noir-sur-blanc avec une chaîne lisible
    figée par document, ce qui garantit la reproductibilité de
    Tesseract cross-OS (à version de binaire constante, le rendu
    PIL est identique).
    """
    out = CORPORA_DIR / "small"
    out.mkdir(parents=True, exist_ok=True)
    _generate_synthetic_corpus(
        out,
        documents=[
            ("doc01", "BENEDICTUS DEUS"),
            ("doc02", "Anno Domini MCMXVII"),
            ("doc03", "Folio 23 recto"),
        ],
    )
    return out


@pytest.fixture(scope="session")
def medium_corpus_dir() -> Path:
    """Corpus *medium* : 30 documents synthétiques.

    Mêmes contraintes que ``small_corpus_dir`` ; le contenu varie
    pour exercer les statistiques sur un échantillon plus large.
    """
    out = CORPORA_DIR / "medium"
    out.mkdir(parents=True, exist_ok=True)
    docs = [
        (f"doc{i:03d}", f"Sample text number {i:03d}")
        for i in range(1, 31)
    ]
    _generate_synthetic_corpus(out, documents=docs)
    return out


# ──────────────────────────────────────────────────────────────────
# Golden snapshots
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def golden_path(request: pytest.FixtureRequest):
    """Factory de chemins de snapshot.

    Usage ::

        def test_phaseN_xxx(golden_path):
            path = golden_path("phase1", "small", "tesseract.txt")
            # path est garanti dans GOLDEN_DIR ; le caller doit
            # l'écrire (au régen) ou le lire (en assertion).

    Le chemin retourné est ``golden/<phase>/<corpus>/<filename>``.
    Le répertoire parent est créé si nécessaire.
    """

    def _make(phase: str, corpus: str, filename: str) -> Path:
        path = GOLDEN_DIR / phase / corpus / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    return _make


@pytest.fixture
def regen_golden(request: pytest.FixtureRequest) -> bool:
    """``True`` si l'utilisateur a passé ``--regen-golden``."""
    return bool(request.config.getoption("--regen-golden"))


def assert_golden_match(
    actual: str | bytes,
    golden_path: Path,
    *,
    regen: bool,
    encoding: str = "utf-8",
) -> None:
    """Compare ``actual`` au contenu de ``golden_path``.

    Si ``regen=True`` ou si le fichier golden n'existe pas, écrit
    ``actual`` au lieu de comparer.  Échoue sinon en cas de
    divergence.
    """
    if isinstance(actual, str):
        if regen or not golden_path.exists():
            golden_path.write_text(actual, encoding=encoding)
            return
        expected = golden_path.read_text(encoding=encoding)
        assert actual == expected, (
            f"Golden mismatch sur {golden_path}.\n"
            f"--- expected ---\n{expected[:500]}\n"
            f"--- actual ---\n{actual[:500]}\n"
            f"\nRégénérer avec ``pytest --regen-golden`` si la "
            "régression est intentionnelle (cf. "
            "regression-tolerances.md)."
        )
    else:
        if regen or not golden_path.exists():
            golden_path.write_bytes(actual)
            return
        expected_b = golden_path.read_bytes()
        assert actual == expected_b, (
            f"Golden mismatch (bytes) sur {golden_path}.\n"
            "Régénérer avec ``pytest --regen-golden`` si "
            "intentionnel."
        )


# ──────────────────────────────────────────────────────────────────
# Comparateurs sémantiques
# ──────────────────────────────────────────────────────────────────


def assert_floats_equal(
    actual: float,
    expected: float,
    *,
    eps: float = 1e-9,
    label: str = "value",
) -> None:
    """Égalité flottante au ε près (cf. regression-tolerances.md)."""
    assert abs(actual - expected) <= eps, (
        f"{label}: actual={actual!r} expected={expected!r} "
        f"diff={abs(actual - expected):.3e} > eps={eps:.0e}"
    )


def assert_set_equal(
    actual: Iterable[Any],
    expected: Iterable[Any],
    *,
    label: str = "set",
) -> None:
    """Égalité ensembliste (ordre ignoré).

    Utilisé typiquement pour les `Pareto front`, l'ensemble des
    Facts narratifs, l'ensemble des lignes CSV.
    """
    a = set(actual)
    e = set(expected)
    missing = e - a
    extra = a - e
    assert not (missing or extra), (
        f"{label}: ensembles différents.\n"
        f"  manquants ({len(missing)}): {sorted(missing)[:10]}\n"
        f"  en trop  ({len(extra)}): {sorted(extra)[:10]}"
    )


def assert_json_semantic_equal(
    actual: dict | list,
    expected: dict | list,
    *,
    label: str = "json",
) -> None:
    """Égalité JSON : sérialisation déterministe puis diff.

    Les deux structures sont sérialisées via
    ``json.dumps(sort_keys=True, ensure_ascii=False, indent=2)``
    avant comparaison — l'ordre des clés ne compte pas, le
    whitespace non plus.
    """
    a = json.dumps(actual, sort_keys=True, ensure_ascii=False, indent=2)
    e = json.dumps(expected, sort_keys=True, ensure_ascii=False, indent=2)
    assert a == e, (
        f"{label}: JSON différents.\n--- expected ---\n{e[:500]}\n"
        f"--- actual ---\n{a[:500]}"
    )


# ──────────────────────────────────────────────────────────────────
# Corpus generation (synthetic)
# ──────────────────────────────────────────────────────────────────


def _generate_synthetic_corpus(
    out_dir: Path,
    *,
    documents: list[tuple[str, str]],
) -> None:
    """Génère un corpus synthétique : pour chaque ``(doc_id, text)``,
    écrit ``out_dir/<doc_id>.png`` (image avec le texte rendu) et
    ``out_dir/<doc_id>.gt.txt`` (la GT).

    Idempotent : si tous les fichiers existent, ne fait rien.
    """
    pytest.importorskip("PIL")
    # Pillow expose ``Image``, ``ImageDraw``, ``ImageFont`` comme
    # **sous-modules**, pas comme attributs du package ``PIL`` ;
    # ``import PIL`` seul ne les attache pas.  Imports explicites
    # ici (Pillow est une dep optionnelle du harness — d'où le
    # ``importorskip`` et le déport en local).
    from PIL import Image, ImageDraw, ImageFont

    for doc_id, text in documents:
        png = out_dir / f"{doc_id}.png"
        gt = out_dir / f"{doc_id}.gt.txt"
        if png.exists() and gt.exists():
            continue
        img = Image.new("RGB", (600, 100), color="white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=32)
        except OSError:
            font = ImageFont.load_default()
        draw.text((20, 30), text, fill="black", font=font)
        img.save(png)
        gt.write_text(text, encoding="utf-8")
