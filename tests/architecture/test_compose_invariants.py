"""Invariants statiques sur ``docker-compose.yml`` et
``docker-compose.prod.yml``.

Ce test ne lance pas ``docker compose config`` (qui demande Docker
installé et fait un merge dynamique).  Il vérifie des propriétés
*structurelles* du YAML pour figer ce qu'on tient à conserver :

1. **Le service Ollama est gated par un profil Compose**.  Sans
   cela, ``docker compose up`` lancerait Ollama par défaut, alors
   que le service est optionnel (un opérateur institutionnel n'en
   veut généralement pas — l'override prod n'a pas à le désactiver
   manuellement parce que le profil le neutralise déjà).

2. **L'override prod force CSRF_REQUIRED + SECURE_COOKIES et exige
   le secret**.  Régression possible si quelqu'un retire le ``:?``
   pensant simplifier.

3. **Le port publié par défaut est 7860 dans les deux fichiers**.
   Le port a déjà migré de 8000 à 7860 — un retour en arrière
   désaligne le compose, le Dockerfile et le HuggingFace Space.

La validation syntaxique du merge (``docker compose config``) est
faite par le job CI ``compose-check`` et la cible ``make
compose-check`` ; ce test couvre l'aspect *sémantique* qu'un
``config`` ne capture pas.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_COMPOSE = REPO_ROOT / "docker-compose.yml"
PROD_COMPOSE = REPO_ROOT / "docker-compose.prod.yml"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    assert isinstance(doc, dict), f"{path.name} : YAML racine non-dict"
    return doc


# ─────────────────────────────────────────────────────────────────────
# 1. Ollama est optionnel (profil Compose)
# ─────────────────────────────────────────────────────────────────────


def test_ollama_service_is_profile_gated_in_base() -> None:
    """Le service ``ollama`` du compose de base doit déclarer
    ``profiles: [ollama]`` (ou un nom équivalent) pour qu'il ne
    démarre PAS par défaut.

    Sans cette clause, ``docker compose up`` lance Ollama, ce qui
    consomme un GPU / 4 GB RAM même quand l'utilisateur n'en veut
    pas, et expose le port 11434 sur le host.
    """
    base = _load(BASE_COMPOSE)
    services = base.get("services") or {}
    assert "ollama" in services, (
        "Service ``ollama`` absent du compose de base.  Si retiré "
        "volontairement, supprimer aussi ce test."
    )
    ollama_cfg = services["ollama"]
    profiles = ollama_cfg.get("profiles")
    assert profiles, (
        "Le service ``ollama`` n'a pas de clause ``profiles:`` — "
        "il démarre par défaut.  Ajouter ``profiles: [ollama]`` "
        "pour le rendre opt-in via ``docker compose --profile ollama up``."
    )
    assert isinstance(profiles, list) and len(profiles) >= 1, (
        f"``profiles:`` doit être une liste non vide, vu : {profiles!r}"
    )


def test_prod_compose_does_not_re_enable_ollama() -> None:
    """L'override prod ne doit pas activer Ollama (en retirant son
    profil ou en redéfinissant le service sans profil).

    Le déploiement institutionnel n'utilise pas Ollama — c'est le
    LLM provider qui change (cloud API).  Réactiver Ollama en prod
    serait une régression silencieuse.
    """
    prod = _load(PROD_COMPOSE)
    services = prod.get("services") or {}
    if "ollama" not in services:
        return  # pas de redéfinition = profil hérité = ok
    ollama_override = services["ollama"]
    profiles = ollama_override.get("profiles")
    assert profiles, (
        "``docker-compose.prod.yml`` redéfinit le service ``ollama`` "
        "sans ``profiles:`` — il serait activé en prod alors que le "
        "déploiement institutionnel utilise un LLM cloud.  Soit "
        "retirer la redéfinition, soit garder un profil explicite."
    )


# ─────────────────────────────────────────────────────────────────────
# 2. L'override prod durcit la sécurité
# ─────────────────────────────────────────────────────────────────────


def _env_dict(env_block: list | dict) -> dict[str, str]:
    """Compose accepte ``environment:`` au format liste
    (``["KEY=value", ...]``) ou dict.  Normaliser en dict."""
    if isinstance(env_block, dict):
        return {k: str(v) for k, v in env_block.items()}
    out: dict[str, str] = {}
    for item in env_block or []:
        s = str(item)
        if "=" in s:
            k, _, v = s.partition("=")
            out[k] = v
        else:
            out[s] = ""
    return out


def test_prod_compose_forces_csrf_required() -> None:
    """``PICARONES_CSRF_REQUIRED=1`` doit être hardcoded en prod
    (pas une valeur par défaut ``${... :-0}``)."""
    prod = _load(PROD_COMPOSE)
    env = _env_dict(prod["services"]["picarones"].get("environment", []))
    val = env.get("PICARONES_CSRF_REQUIRED", "")
    assert val == "1", (
        f"``PICARONES_CSRF_REQUIRED`` en prod = {val!r}, attendu '1'.  "
        f"L'override prod doit forcer CSRF — pas le rendre opt-in."
    )


def test_prod_compose_forces_secure_cookies() -> None:
    """``PICARONES_SECURE_COOKIES=1`` doit être hardcoded en prod."""
    prod = _load(PROD_COMPOSE)
    env = _env_dict(prod["services"]["picarones"].get("environment", []))
    val = env.get("PICARONES_SECURE_COOKIES", "")
    assert val == "1", (
        f"``PICARONES_SECURE_COOKIES`` en prod = {val!r}, attendu '1'."
    )


def test_prod_compose_requires_csrf_secret() -> None:
    """``PICARONES_CSRF_SECRET`` en prod doit utiliser la syntaxe
    ``${VAR:?msg}`` qui fait échouer Compose si la variable n'est pas
    posée.  Sans ça, le conteneur démarrerait avec un secret vide ou
    un fallback dangereux."""
    prod_text = PROD_COMPOSE.read_text(encoding="utf-8")
    assert "PICARONES_CSRF_SECRET=${PICARONES_CSRF_SECRET:?" in prod_text, (
        "``docker-compose.prod.yml`` doit déclarer "
        "``PICARONES_CSRF_SECRET=${PICARONES_CSRF_SECRET:?...}`` "
        "(la syntaxe ``:?`` fait échouer Compose à l'évaluation si "
        "la variable n'est pas posée).  Sinon le secret peut être "
        "vide en prod."
    )


# ─────────────────────────────────────────────────────────────────────
# 3. Port 7860 stable
# ─────────────────────────────────────────────────────────────────────


def test_base_compose_publishes_port_7860() -> None:
    """Le compose de base publie le service Picarones sur 7860 (port
    aligné avec HuggingFace Space et le Dockerfile)."""
    base = _load(BASE_COMPOSE)
    ports = base["services"]["picarones"].get("ports") or []
    assert any(":7860" in str(p) for p in ports), (
        f"Aucun mapping de port vers 7860 trouvé dans docker-compose.yml "
        f"(ports vus : {ports}).  Port HuggingFace Space = 7860."
    )


def test_prod_compose_publishes_port_7860() -> None:
    """L'override prod doit aussi pointer sur 7860."""
    prod = _load(PROD_COMPOSE)
    ports = prod["services"]["picarones"].get("ports") or []
    assert any(":7860" in str(p) for p in ports), (
        f"Aucun mapping vers 7860 dans docker-compose.prod.yml "
        f"(ports vus : {ports})."
    )
