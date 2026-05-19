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


class TestTesseractInDockerfile:
    """Le Dockerfile doit installer ``tesseract-ocr`` + les 6
    modèles de langues du corpus institutionnel BnF (fra, lat,
    eng, deu, ita, spa).

    Sprint S6.1 a tenté un pin exact ``=5.3.0-2`` mais Debian
    point-release rebump fréquemment, cassant le build.  La
    reproductibilité passe désormais par :

    1. Base image Python pinée par digest SHA256.
    2. ``requirements-docker.lock`` côté Python.
    3. ``RunManifest.dependencies_lock`` qui capture la version
       Tesseract effective au runtime (``tesseract --version``).
    """

    def setup_method(self) -> None:
        self.text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    def test_tesseract_ocr_installed(self) -> None:
        # Pattern : ``tesseract-ocr`` au début d'un mot (suivi de
        # whitespace, ``\``, ou ``=``), pour ne pas matcher
        # ``tesseract-ocr-fra`` etc.
        assert re.search(r"\btesseract-ocr(?:[\s\\=]|$)", self.text), (
            "Le Dockerfile n'installe pas ``tesseract-ocr``."
        )

    def test_all_language_models_installed(self) -> None:
        """Les modèles de langues du corpus BnF doivent tous être
        installés (fra, lat, eng, deu, ita, spa).
        """
        languages = ("fra", "lat", "eng", "deu", "ita", "spa")
        for lang in languages:
            pattern = rf"tesseract-ocr-{lang}(?:[\s\\=]|$)"
            assert re.search(pattern, self.text), (
                f"Modèle ``tesseract-ocr-{lang}`` non installé dans "
                f"le Dockerfile."
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


class TestComposeStartabilityCoherence:
    """Régression P0 — ``docker compose up`` (chemin local documenté)
    doit démarrer SANS configuration : un défaut ``CSRF_REQUIRED=1``
    sans ``CSRF_SECRET`` fait échouer ``validate_csrf_config`` au
    lifespan.  Le durcissement CSRF vit dans l'override prod, qui
    EXIGE le secret via la substitution Compose ``:?``."""

    def _compose(self, name: str) -> str:
        return (REPO_ROOT / name).read_text(encoding="utf-8")

    def test_local_compose_does_not_force_csrf_required(self) -> None:
        text = self._compose("docker-compose.yml")
        assert "PICARONES_CSRF_REQUIRED=${PICARONES_CSRF_REQUIRED:-0}" in text, (
            "Le compose local ne doit PAS forcer CSRF_REQUIRED=1 : "
            "sans CSRF_SECRET, validate_csrf_config refuse le "
            "démarrage et ``docker compose up`` casse."
        )

    def test_local_compose_default_env_passes_startup_guards(self) -> None:
        """Simule les defaults du compose local et vérifie que les
        garde-fous lifespan ne lèvent pas."""
        import os

        from picarones.interfaces.web.security import (
            check_deployment_coherence,
            validate_csrf_config,
        )

        snapshot = dict(os.environ)
        try:
            for k in (
                "PICARONES_CSRF_REQUIRED", "PICARONES_CSRF_SECRET",
                "PICARONES_SECURE_COOKIES", "SPACE_ID",
            ):
                os.environ.pop(k, None)
            os.environ["PICARONES_PUBLIC_MODE"] = "1"  # défaut compose local
            validate_csrf_config()          # ne doit pas lever
            check_deployment_coherence()    # ne doit pas lever
        finally:
            os.environ.clear()
            os.environ.update(snapshot)

    def test_prod_override_requires_csrf_secret(self) -> None:
        """``docker-compose.prod.yml`` doit exiger le secret via la
        substitution Compose ``${PICARONES_CSRF_SECRET:?...}`` —
        Compose refuse AVANT de lancer le conteneur, message clair."""
        text = self._compose("docker-compose.prod.yml")
        assert "PICARONES_CSRF_SECRET=${PICARONES_CSRF_SECRET:?" in text, (
            "L'override prod doit rendre le secret OBLIGATOIRE "
            "(forme ``:?`` — échec Compose explicite, pas un crash "
            "conteneur au lifespan)."
        )
        assert "PICARONES_CSRF_REQUIRED=1" in text
        assert "PICARONES_SECURE_COOKIES=1" in text


class TestDockerPortCoherence:
    """Régression P0.4 — le conteneur sert sur 7860 (EXPOSE + CMD).
    Tout mapping ``8000:8000`` est cassé (rien n'écoute sur :8000
    côté conteneur).  Verrouille l'absence de drift Dockerfile /
    Makefile / compose."""

    def test_dockerfile_serves_and_exposes_7860(self) -> None:
        text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "EXPOSE 7860" in text
        assert '"--port", "7860"' in text
        assert "8000:8000" not in text, (
            "commentaires Dockerfile : mapping 8000:8000 trompeur "
            "(le conteneur sert sur 7860)."
        )

    def test_dockerfile_version_label_not_hardcoded(self) -> None:
        text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert 'LABEL version="1.0.0"' not in text, (
            "version Docker figée à 1.0.0 — dérive de la version "
            "réelle (setuptools-scm).  Utiliser ARG PICARONES_VERSION."
        )
        assert "ARG PICARONES_VERSION" in text

    def test_makefile_docker_targets_use_7860(self) -> None:
        text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        # La cible docker-run mappait 8000:8000 → conteneur muet.
        assert "-p 8000:8000" not in text, (
            "make docker-run mappait 8000:8000 alors que le conteneur "
            "sert sur 7860 — cible cassée."
        )
        assert "-p 7860:7860" in text
        assert "picarones:1.0.0" not in text, (
            "tag Docker figé picarones:1.0.0 — dériver la version."
        )
