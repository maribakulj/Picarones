"""Tests Sprint A4 — protection CSRF (item B-11 de l'audit).

Couvre les deux modes de l'application :

- **Mode public** (``PICARONES_CSRF_REQUIRED`` non défini) : le
  middleware est no-op, tous les POST passent sans token. C'est le
  régime du HuggingFace Space où il n'y a pas de session
  authentifiée à protéger.
- **Mode institutionnel** (``PICARONES_CSRF_REQUIRED=1``) : tout
  POST/PUT/DELETE/PATCH non exempt exige un token cookie + header
  signé HMAC. Régime cible BnF derrière SSO.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def public_client(monkeypatch) -> TestClient:
    """Client en mode public (CSRF désactivé)."""
    monkeypatch.delenv("PICARONES_CSRF_REQUIRED", raising=False)
    # On ré-importe l'app pour que le middleware lise la variable à
    # chaque requête (il le fait déjà via ``is_csrf_required()`` à chaque
    # appel — pas besoin de reload du module).
    from picarones.interfaces.web.app import app
    return TestClient(app)


@pytest.fixture
def institutional_client(monkeypatch) -> TestClient:
    """Client en mode institutionnel (CSRF activé)."""
    monkeypatch.setenv("PICARONES_CSRF_REQUIRED", "1")
    monkeypatch.setenv("PICARONES_CSRF_SECRET", "test-secret-do-not-use-in-prod" * 2)
    from picarones.interfaces.web.app import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Mode public — bypass complet
# ---------------------------------------------------------------------------


def test_public_mode_post_succeeds_without_token(public_client: TestClient) -> None:
    """En mode public, un POST sans token CSRF doit passer normalement."""
    # ``/api/lang/{lang_code}`` est un POST simple sans dépendance lourde
    r = public_client.post("/api/lang/fr")
    assert r.status_code == 200, r.text


def test_public_mode_csrf_endpoint_returns_disabled(public_client: TestClient) -> None:
    """``/api/csrf/token`` doit signaler ``enabled=false`` en mode public."""
    r = public_client.get("/api/csrf/token")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["token"] is None


# ---------------------------------------------------------------------------
# Mode institutionnel — protection active
# ---------------------------------------------------------------------------


def test_institutional_post_without_token_returns_403(
    institutional_client: TestClient,
) -> None:
    """En mode institutionnel, un POST sans token doit renvoyer 403."""
    r = institutional_client.post("/api/lang/fr")
    assert r.status_code == 403
    assert "CSRF" in r.json()["detail"]


def test_institutional_post_with_only_cookie_returns_403(
    institutional_client: TestClient,
) -> None:
    """Cookie présent mais header absent → 403 (le pattern double-submit
    exige les deux)."""
    # 1. Récupérer un token via /api/csrf/token (qui pose le cookie)
    r = institutional_client.get("/api/csrf/token")
    token = r.json()["token"]
    # 2. POST avec le cookie automatique mais sans header
    r2 = institutional_client.post("/api/lang/fr")
    assert r2.status_code == 403
    assert token  # sanity : le token a bien été retourné


def test_institutional_post_with_only_header_returns_403(
    institutional_client: TestClient,
) -> None:
    """Header présent mais cookie absent → 403."""
    # On force l'absence de cookie en utilisant un nouveau client sans état
    from picarones.interfaces.web.app import app
    fresh = TestClient(app)
    r = fresh.post(
        "/api/lang/fr",
        headers={"X-CSRF-Token": "deadbeef.cafebabe"},
    )
    assert r.status_code == 403


def test_institutional_post_with_valid_token_succeeds(
    institutional_client: TestClient,
) -> None:
    """Token cookie + header identiques et signature valide → 200."""
    r = institutional_client.get("/api/csrf/token")
    token = r.json()["token"]
    assert token is not None

    # Le cookie est posé automatiquement par TestClient ; on injecte
    # le même token dans le header.
    r2 = institutional_client.post(
        "/api/lang/fr",
        headers={"X-CSRF-Token": token},
    )
    assert r2.status_code == 200, r2.text


def test_institutional_post_with_mismatched_token_returns_403(
    institutional_client: TestClient,
) -> None:
    """Cookie et header différents → 403 (anti-CSRF en double-submit)."""
    institutional_client.get("/api/csrf/token")  # pose le cookie
    r = institutional_client.post(
        "/api/lang/fr",
        headers={"X-CSRF-Token": "0011223344556677.aabbccddeeff0011"},
    )
    assert r.status_code == 403


def test_institutional_post_with_forged_signature_returns_403(
    institutional_client: TestClient,
    monkeypatch,
) -> None:
    """Token au bon format mais signature non-HMAC valide → 403."""
    # On forge un token cookie+header identiques mais signé avec un
    # secret bidon — le middleware doit le rejeter.
    forged = "deadbeefcafebabe1234567890abcdef.000102030405060708090a0b0c0d0e0f"
    institutional_client.cookies.set("picarones_csrf", forged)
    r = institutional_client.post(
        "/api/lang/fr",
        headers={"X-CSRF-Token": forged},
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Endpoints exemptés
# ---------------------------------------------------------------------------


def test_health_is_csrf_exempt(institutional_client: TestClient) -> None:
    """``/health`` doit rester accessible sans token même en mode CSRF."""
    r = institutional_client.get("/health")
    assert r.status_code == 200


def test_csrf_token_endpoint_does_not_require_token(
    institutional_client: TestClient,
) -> None:
    """``/api/csrf/token`` lui-même ne doit pas exiger un token (sinon
    impossible de bootstraper)."""
    # Le endpoint est en GET donc CSRF ne s'applique pas, mais on
    # vérifie aussi qu'il est dans la liste des exemptions (au cas où
    # un PR futur le passerait en POST).
    from picarones.interfaces.web.security import CSRF_EXEMPT_PATH_PREFIXES
    assert any(
        "/api/csrf/token".startswith(p) for p in CSRF_EXEMPT_PATH_PREFIXES
    )


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------


def test_generate_then_verify_token_round_trip(monkeypatch) -> None:
    """``generate_csrf_token`` produit un token que ``verify_csrf_token``
    valide. Garantit le round-trip de signature."""
    monkeypatch.setenv("PICARONES_CSRF_SECRET", "round-trip-secret-32-bytes-ok!")
    from picarones.interfaces.web.security import generate_csrf_token, verify_csrf_token

    token = generate_csrf_token()
    assert verify_csrf_token(token) is True


def test_verify_token_rejects_garbage(monkeypatch) -> None:
    monkeypatch.setenv("PICARONES_CSRF_SECRET", "round-trip-secret-32-bytes-ok!")
    from picarones.interfaces.web.security import verify_csrf_token

    assert verify_csrf_token(None) is False
    assert verify_csrf_token("") is False
    assert verify_csrf_token("no-dot") is False
    assert verify_csrf_token("bad.hex") is False
    assert verify_csrf_token("deadbeef.deadbeef") is False  # ok format, signature wrong


def test_csrf_disabled_by_default(monkeypatch) -> None:
    """Garantit qu'on a bien posé la rétrocompat HuggingFace : sans la
    variable d'env, ``is_csrf_required()`` retourne False."""
    monkeypatch.delenv("PICARONES_CSRF_REQUIRED", raising=False)
    from picarones.interfaces.web.security import is_csrf_required

    assert is_csrf_required() is False


@pytest.mark.parametrize("value,expected", [
    ("1", True),
    ("true", True),
    ("yes", True),
    # NB : pattern aligné avec ``is_public_mode()`` — sensible à la casse
    # (convention pré-existante). ``TRUE`` n'est pas accepté par design.
    ("0", False),
    ("false", False),
    ("", False),
    ("anything", False),
    ("TRUE", False),  # documenté : majuscule rejetée
])
def test_csrf_env_var_parsing(monkeypatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("PICARONES_CSRF_REQUIRED", value)
    from picarones.interfaces.web.security import is_csrf_required

    assert is_csrf_required() is expected
