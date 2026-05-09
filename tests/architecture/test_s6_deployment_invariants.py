"""Sprint S6 — garde-fous de reproductibilité institutionnelle.

Ces tests verrouillent les contraintes de déploiement BnF :

S6.1 / S6.2 — Tesseract version pinée dans Dockerfile
S6.3 — Bornes supérieures sur les dépendances Python
S6.4 — OLLAMA_ORIGINS restreint en docker-compose
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# ──────────────────────────────────────────────────────────────────────
# S6.2 — Le Dockerfile pin Tesseract à une version Debian précise
# ──────────────────────────────────────────────────────────────────────


class TestTesseractPinnedInDockerfile:
    """Sans pin Tesseract, deux builds peuvent ramener des versions
    différentes (5.3.0 vs 5.4.1) → CER mesurés divergent → benchmarks
    non reproductibles.  Inacceptable pour publication scientifique.
    """

    def setup_method(self) -> None:
        self.text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    def test_tesseract_ocr_has_explicit_version_pin(self) -> None:
        # Pattern : ``tesseract-ocr=<version>`` (avec ``=``, pas
        # juste ``tesseract-ocr``).
        match = re.search(
            r"tesseract-ocr=(\d+\.\d+\.\d+(?:-\d+)?)",
            self.text,
        )
        assert match, (
            "Le Dockerfile installe ``tesseract-ocr`` sans pin de "
            "version.  Pour la reproductibilité institutionnelle "
            "(BnF), ajouter ``=5.3.0-2`` (ou autre version Debian "
            "courante explicite)."
        )

    def test_all_language_models_have_pin(self) -> None:
        """Les modèles de langues (fra, lat, eng, ...) doivent eux
        aussi être pinnés.  Une version différente de
        ``tesseract-ocr-fra`` peut changer le résultat OCR sur les
        manuscrits français."""
        languages = ("fra", "lat", "eng", "deu", "ita", "spa")
        for lang in languages:
            pattern = rf"tesseract-ocr-{lang}=[\w.\-:]+"
            assert re.search(pattern, self.text), (
                f"Modèle ``tesseract-ocr-{lang}`` non pinné dans le "
                f"Dockerfile."
            )


# ──────────────────────────────────────────────────────────────────────
# S6.3 — Bornes supérieures sur les dépendances core
# ──────────────────────────────────────────────────────────────────────


class TestDependencyUpperBounds:
    """Sans borne supérieure, ``pip install picarones`` en 2027 peut
    remonter ``click==9.0`` qui casse l'API."""

    def setup_method(self) -> None:
        self.text = (REPO_ROOT / "pyproject.toml").read_text(
            encoding="utf-8",
        )

    def test_core_deps_have_upper_bound(self) -> None:
        """Chaque dépendance core listée doit avoir un caplock
        ``<X.0`` (où X est la majeure suivante)."""
        # Cherche le bloc ``dependencies = [...]``.
        m = re.search(
            r"^dependencies\s*=\s*\[(.*?)^\]",
            self.text,
            re.DOTALL | re.MULTILINE,
        )
        assert m, "Bloc ``dependencies`` introuvable dans pyproject.toml"
        block = m.group(1)
        # Liste les lignes ``"name>=X.Y..."``.
        dep_lines = re.findall(r'"([a-zA-Z][\w\-]*)>=[^"]+"', block)

        unbounded = []
        for name in dep_lines:
            # Pattern ``"name>=X.Y...,<Z..."`` ou ``"name>=X.Y...,<Z.W"``
            pattern = rf'"{re.escape(name)}>=[^"]+,\s*<[^"]+"'
            if not re.search(pattern, block):
                unbounded.append(name)

        assert not unbounded, (
            f"Dépendances sans borne supérieure : {unbounded}.\n"
            f"Ajouter ``<MAJEURE_SUIVANTE.0`` à chacune dans "
            f"pyproject.toml."
        )


# ──────────────────────────────────────────────────────────────────────
# S6.4 — OLLAMA_ORIGINS n'est pas ``*`` par défaut
# ──────────────────────────────────────────────────────────────────────


class TestOllamaOriginsRestricted:
    """``OLLAMA_ORIGINS=*`` permet à n'importe quel site web d'appeler
    l'API Ollama interne via le navigateur de l'utilisateur (CSRF
    cross-origin)."""

    def test_docker_compose_does_not_set_ollama_origins_wildcard(
        self,
    ) -> None:
        text = (REPO_ROOT / "docker-compose.yml").read_text(
            encoding="utf-8",
        )
        # ``OLLAMA_ORIGINS=*`` brut (sans variable d'env override)
        # doit être absent.
        assert "OLLAMA_ORIGINS=*" not in text, (
            "``docker-compose.yml`` configure ``OLLAMA_ORIGINS=*`` "
            "qui désactive la protection CORS de Ollama.  Restreindre "
            "à un origin explicite ou à une variable d'env "
            "``${OLLAMA_ORIGINS}`` avec un défaut sécurisé."
        )

    def test_docker_compose_uses_env_override_for_ollama_origins(
        self,
    ) -> None:
        """La config doit utiliser la forme ``${OLLAMA_ORIGINS:-...}``
        avec un défaut sécurisé pour permettre une override
        contrôlée par l'opérateur."""
        text = (REPO_ROOT / "docker-compose.yml").read_text(
            encoding="utf-8",
        )
        assert "OLLAMA_ORIGINS=${OLLAMA_ORIGINS" in text, (
            "``docker-compose.yml`` doit utiliser "
            "``OLLAMA_ORIGINS=${OLLAMA_ORIGINS:-...}`` pour permettre "
            "une override par variable d'env (avec un défaut "
            "restrictif)."
        )
