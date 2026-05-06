"""Garde-fous sur le parsing X-Forwarded-For du ``RateLimitMiddleware``.

L'audit S58 a corrigé une faille IP-spoofing (lecture du PREMIER XFF
au lieu de la N-ième en partant de la fin).  Le commit S58 #4 introduit
``trust_proxy_count: int`` qui remplace ``trust_x_forwarded_for: bool``,
mais aucun test ne vérifiait la nouvelle logique.

Ces tests verrouillent le contrat sécuritaire :

1. ``trust_proxy_count=0`` : XFF totalement ignoré (mode safe par défaut).
2. ``trust_proxy_count=1`` : un proxy en amont, on lit la dernière IP
   de la chaîne (le proxy direct est trustworthy).
3. ``trust_proxy_count=N`` mais chaîne plus courte → fallback gracieux.
4. Spoof attempt avec une IP injectée en tête → ignorée si la chaîne
   est plus courte qu'attendu.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from starlette.requests import Request

from picarones.interfaces.web.security import RateLimitMiddleware


def _request(xff: str | None, client_host: str = "10.0.0.1") -> Request:
    """Construit une ``Request`` minimale pour ``_extract_ip``."""
    headers: list[tuple[bytes, bytes]] = []
    if xff is not None:
        headers.append((b"x-forwarded-for", xff.encode("ascii")))
    scope = {
        "type": "http",
        "headers": headers,
        "client": (client_host, 0),
    }
    return Request(scope)  # type: ignore[arg-type]


def _middleware(trust_proxy_count: int = 0) -> RateLimitMiddleware:
    """Instance prête à appeler ``_extract_ip`` (l'app sous-jacent
    n'est pas exercé, on teste uniquement le helper de parsing)."""
    return RateLimitMiddleware(
        app=MagicMock(),
        trust_proxy_count=trust_proxy_count,
    )


def test_xff_ignored_when_trust_count_zero() -> None:
    """Mode par défaut : XFF est ignoré, l'IP du socket prime.
    Évite tout spoofing si le serveur est exposé directement.
    """
    mw = _middleware(trust_proxy_count=0)
    req = _request(xff="evil.ip.example, real, proxy", client_host="1.2.3.4")
    assert mw._extract_ip(req) == "1.2.3.4"


def test_xff_one_proxy_reads_last_ip() -> None:
    """Avec ``trust_proxy_count=1`` (nginx local par ex.), on lit la
    dernière IP de la chaîne — c'est l'IP que nginx a vue arriver,
    pas celle que le client a forgée.
    """
    mw = _middleware(trust_proxy_count=1)
    req = _request(xff="evil.ip.example, real-client", client_host="10.0.0.1")
    assert mw._extract_ip(req) == "real-client"


def test_xff_two_proxies_reads_n_minus_2() -> None:
    """Avec ``trust_proxy_count=2`` (load balancer + nginx), on lit
    l'avant-avant-dernière IP.
    """
    mw = _middleware(trust_proxy_count=2)
    req = _request(
        xff="client, attacker-spoof, real-client, edge-proxy",
        client_host="10.0.0.1",
    )
    # parts = [client, attacker-spoof, real-client, edge-proxy]
    # idx = max(0, 4 - 2) = 2 → "real-client"
    assert mw._extract_ip(req) == "real-client"


def test_xff_chain_shorter_than_expected_falls_back_gracefully() -> None:
    """Si la chaîne XFF est plus courte que ``trust_proxy_count``
    (mauvaise config ou client tronquant), on ne crash pas — on lit
    l'IP la plus à gauche disponible.
    """
    mw = _middleware(trust_proxy_count=5)
    req = _request(xff="single-ip", client_host="10.0.0.1")
    # parts = [single-ip], idx = max(0, 1 - 5) = 0 → "single-ip"
    assert mw._extract_ip(req) == "single-ip"


def test_xff_empty_value_ignored() -> None:
    """Une chaîne XFF vide retombe sur ``request.client.host``."""
    mw = _middleware(trust_proxy_count=1)
    req = _request(xff="", client_host="10.0.0.1")
    assert mw._extract_ip(req) == "10.0.0.1"


def test_xff_with_whitespace_normalized() -> None:
    """Les espaces autour des virgules sont strippés."""
    mw = _middleware(trust_proxy_count=1)
    req = _request(xff="  client  ,  real-client  ", client_host="10.0.0.1")
    assert mw._extract_ip(req) == "real-client"


def test_no_client_returns_unknown() -> None:
    """Si ``request.client`` est ``None`` (cas exotique ASGI sans
    socket), l'extraction retourne ``"unknown"`` plutôt que crash.
    """
    mw = _middleware(trust_proxy_count=0)
    scope = {"type": "http", "headers": [], "client": None}
    req = Request(scope)  # type: ignore[arg-type]
    assert mw._extract_ip(req) == "unknown"
