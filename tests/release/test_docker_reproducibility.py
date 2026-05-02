"""Tests Sprint A16 — reproductibilité du build Docker.

Items couverts :

- **digest-pinning** : la base image ``python:3.11.13-slim`` est
  référencée par ``@sha256:...`` (et pas seulement par tag), pour
  geler l'image binaire bit-à-bit entre deux ``docker build``.

- **lock file** : ``requirements-docker.lock`` existe, est aligné sur
  les extras consommés par le Dockerfile (``[web,llm]``), et couvre
  toutes les top-level dependencies déclarées dans ``pyproject.toml``.

- **install path** : le Dockerfile consomme le lock avec ``--no-deps``
  (pas de re-résolution dynamique de l'arbre transitif), et installe
  Picarones en éditable séparément (toujours en ``--no-deps`` puisque
  le lock contient déjà ses dépendances).

Ces tests sont cheap (lecture de fichiers, regex) — ils ne lancent
pas ``uv pip compile`` ni ``docker build``. La validation drift-free
stricte du lock contre ``uv pip compile`` reste à arbitrer (nécessite
``uv`` en CI ; non bloquant tant que le lock est régénéré à la main
quand pyproject change, et que le test #7 attrape les oublis grossiers).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
LOCK = REPO_ROOT / "requirements-docker.lock"
PYPROJECT = REPO_ROOT / "pyproject.toml"

DIGEST_PATTERN = re.compile(
    r"python:3\.\d+(\.\d+)?-slim@sha256:[a-f0-9]{64}"
)


# ---------------------------------------------------------------------------
# Digest pinning
# ---------------------------------------------------------------------------


def _read_dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_pins_python_by_digest() -> None:
    """L'``ARG PYTHON_BASE_IMAGE`` doit pointer une image avec digest
    ``@sha256:...`` — sinon le build n'est pas reproductible."""
    text = _read_dockerfile()
    args = re.findall(
        r"^ARG PYTHON_BASE_IMAGE=(\S+)",
        text,
        re.MULTILINE,
    )
    assert args, "Aucun ARG PYTHON_BASE_IMAGE trouvé dans le Dockerfile"
    for value in args:
        assert DIGEST_PATTERN.match(value), (
            f"ARG PYTHON_BASE_IMAGE={value!r} n'est pas pinné par digest. "
            "Forme attendue : ``python:3.11.13-slim@sha256:<64hex>``."
        )


def test_runtime_and_builder_share_same_digest() -> None:
    """Les deux ``ARG PYTHON_BASE_IMAGE`` (builder + runtime) doivent
    avoir le même digest, sinon les couches OS divergent."""
    text = _read_dockerfile()
    args = re.findall(
        r"^ARG PYTHON_BASE_IMAGE=(\S+)",
        text,
        re.MULTILINE,
    )
    assert len(args) == 2, (
        f"Attendu 2 ARG PYTHON_BASE_IMAGE (builder + runtime), trouvé {len(args)}"
    )
    assert args[0] == args[1], (
        f"Builder ({args[0]}) et runtime ({args[1]}) doivent référencer "
        "exactement le même digest — sinon la couche de base diverge."
    )


# ---------------------------------------------------------------------------
# Lock file
# ---------------------------------------------------------------------------


def test_lock_file_exists() -> None:
    assert LOCK.exists(), (
        f"{LOCK.name} doit exister (Sprint A16). "
        "Le générer via : ``uv pip compile pyproject.toml "
        "--extra web --extra llm -o requirements-docker.lock``."
    )


def test_lock_file_header_mentions_correct_extras() -> None:
    """Le commentaire d'en-tête de ``uv pip compile`` doit mentionner
    les extras ``web`` et ``llm`` — sinon le lock a été généré pour
    autre chose et ne reflète pas ce que le Dockerfile installe."""
    head = LOCK.read_text(encoding="utf-8").splitlines()[:5]
    header = "\n".join(head)
    assert "--extra web" in header, (
        f"Lock header doit mentionner ``--extra web`` ; trouvé : {header!r}"
    )
    assert "--extra llm" in header, (
        f"Lock header doit mentionner ``--extra llm`` ; trouvé : {header!r}"
    )


def test_lock_file_nonempty() -> None:
    """Au moins ~50 lignes (deps web+llm = fastapi, anthropic, openai,
    mistralai et leurs transitives — typiquement ~140 lignes)."""
    lines = LOCK.read_text(encoding="utf-8").splitlines()
    n_pkg_lines = sum(1 for ln in lines if re.match(r"^[a-z0-9].*==", ln))
    assert n_pkg_lines >= 30, (
        f"Lock file ne contient que {n_pkg_lines} packages — "
        "anormal pour [web,llm]. Re-générer le lock."
    )


# ---------------------------------------------------------------------------
# Couverture top-level pyproject ↔ lock
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    """PEP 503 normalization : lowercase, [-_.] → -."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _extract_pyproject_top_level_deps() -> set[str]:
    """Renvoie les noms (normalisés) des deps top-level installées
    par le Dockerfile : ``[project] dependencies`` + extras ``web`` + ``llm``."""
    text = PYPROJECT.read_text(encoding="utf-8")

    def _names_in_block(block: str) -> set[str]:
        names: set[str] = set()
        for line in block.splitlines():
            line = line.strip().strip(",").strip('"').strip("'")
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
            if m:
                names.add(_normalize(m.group(1)))
        return names

    deps_block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL | re.MULTILINE)
    web_block = re.search(r"^web\s*=\s*\[(.*?)\]", text, re.DOTALL | re.MULTILINE)
    llm_block = re.search(r"^llm\s*=\s*\[(.*?)\]", text, re.DOTALL | re.MULTILINE)

    assert deps_block, "[project] dependencies introuvable dans pyproject.toml"
    assert web_block, "extra ``web`` introuvable dans pyproject.toml"
    assert llm_block, "extra ``llm`` introuvable dans pyproject.toml"

    return (
        _names_in_block(deps_block.group(1))
        | _names_in_block(web_block.group(1))
        | _names_in_block(llm_block.group(1))
    )


def _extract_lock_pkg_names() -> set[str]:
    text = LOCK.read_text(encoding="utf-8")
    names: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==", line)
        if m:
            names.add(_normalize(m.group(1)))
    return names


def test_lock_covers_pyproject_top_level_deps() -> None:
    """Toutes les top-level deps de pyproject ([project] + extras
    web + llm) doivent apparaître dans le lock. Si une dep est ajoutée
    à pyproject sans régénérer le lock, ce test attrape l'oubli."""
    pyproject_deps = _extract_pyproject_top_level_deps()
    lock_pkgs = _extract_lock_pkg_names()
    missing = pyproject_deps - lock_pkgs
    # ``picarones`` lui-même n'apparaît pas dans le lock (auto-référence).
    missing.discard("picarones")
    assert not missing, (
        f"Top-level deps présentes dans pyproject.toml mais absentes du "
        f"lock : {sorted(missing)}. "
        "Re-générer via : ``uv pip compile pyproject.toml --extra web "
        "--extra llm -o requirements-docker.lock``."
    )


# ---------------------------------------------------------------------------
# Install path dans le Dockerfile
# ---------------------------------------------------------------------------


def test_dockerfile_copies_lock_file() -> None:
    text = _read_dockerfile()
    assert "COPY requirements-docker.lock" in text, (
        "Dockerfile doit COPY requirements-docker.lock dans le builder, "
        "sinon le lock n'est pas disponible au moment du pip install."
    )


def test_dockerfile_installs_from_lock_with_no_deps() -> None:
    """Le Dockerfile doit installer ``-r requirements-docker.lock`` avec
    ``--no-deps`` (pas de re-résolution) — sinon le lock ne sert à rien."""
    text = _read_dockerfile()
    assert "-r requirements-docker.lock" in text, (
        "Dockerfile doit installer depuis le lock : "
        "``pip install --no-deps -r requirements-docker.lock``."
    )
    # Cherche la ligne d'install et vérifie le ``--no-deps``.
    install_lines = [
        ln for ln in text.splitlines()
        if "requirements-docker.lock" in ln and "pip install" in ln
    ]
    assert install_lines, "Ligne d'install du lock introuvable"
    for ln in install_lines:
        assert "--no-deps" in ln, (
            f"Install du lock sans ``--no-deps`` : {ln!r}. "
            "Sans ``--no-deps``, pip re-résoudrait l'arbre et casserait "
            "la reproductibilité."
        )


def test_dockerfile_installs_picarones_no_deps() -> None:
    """Picarones lui-même doit être installé en ``--no-deps`` car le
    lock contient déjà toutes ses dépendances (sinon double install)."""
    text = _read_dockerfile()
    # On cherche la ligne ``pip install ... -e .`` (avec ou sans extras).
    editable_lines = [
        ln for ln in text.splitlines()
        if "pip install" in ln and re.search(r"-e\s+\.(\[|$|\s)", ln)
    ]
    assert editable_lines, "Pas de ``pip install -e .`` dans le Dockerfile"
    # Au moins une de ces lignes doit avoir --no-deps.
    has_no_deps = any("--no-deps" in ln for ln in editable_lines)
    assert has_no_deps, (
        "Aucune ligne ``pip install -e .`` n'utilise ``--no-deps`` ; "
        "le lock contient déjà les deps, double install évitable."
    )


def test_dockerfile_does_not_use_implicit_extras() -> None:
    """Avec le lock approach, on ne doit PLUS faire ``-e .[web,llm]`` —
    les extras sont déjà résolus dans le lock. Si cette ligne réapparaît,
    pip va re-résoudre depuis pyproject et le build cesse d'être
    reproductible."""
    text = _read_dockerfile()
    # Cherche ``-e ".[web,llm]"`` ou variants.
    forbidden = re.findall(r'-e\s+["\']?\.\[[^\]]+\]', text)
    assert not forbidden, (
        f"Dockerfile contient encore un install avec extras : {forbidden}. "
        "Avec le lock, utiliser ``pip install --no-deps -e .`` à la place."
    )
