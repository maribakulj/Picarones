"""Helpers HTTP partagés par les importeurs IIIF / Gallica / HTR-United.

Chantier 4 du plan d'évolution post-Sprint 97 — fusion Gallica vers IIIF.

Auparavant les fonctions ``_validate_url`` et ``_download_url`` étaient
dupliquées entre :mod:`picarones.adapters.corpus.iiif` (lignes 310-344) et
:mod:`picarones.adapters.corpus.gallica` (lignes 125-155). Le module Gallica
faisait 549 lignes dont une bonne partie réimplémentait les mêmes
abstractions HTTP que IIIF (validation de schéma, retry exponentiel,
gestion des codes HTTP).

Ce module privé centralise ces helpers. Les deux importeurs (et tout
nouveau importateur HTTP futur) les utilisent. Comportement public
inchangé — uniquement de la factorisation.
"""

from __future__ import annotations

import ipaddress
import logging
import time
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_USER_AGENT = (
    "Picarones/1.0 (OCR benchmark platform; "
    "https://github.com/maribakulj/Picarones)"
)

#: Hostnames qui résolvent vers du loopback ou des métadonnées
#: cloud — refusés littéralement avant même la résolution DNS.
_BLOCKED_HOSTNAMES: frozenset[str] = frozenset({
    "localhost",
    "ip6-localhost",
    "ip6-loopback",
    # Cloud metadata
    "metadata.google.internal",
    "metadata",  # alias court interne GCP
    # AWS metadata host alias
    "instance-data",
})


def _is_blocked_host(hostname: str) -> bool:
    """Vrai si ``hostname`` (déjà lowercased) doit être refusé.

    Trois catégories :

    1. Hostname littéral dans :data:`_BLOCKED_HOSTNAMES`.
    2. Adresse IP littérale qui tombe dans une plage réservée :
       loopback (``127.0.0.0/8``, ``::1``), lien-local
       (``169.254.0.0/16``, ``fe80::/10``), privé RFC 1918
       (``10/8``, ``172.16/12``, ``192.168/16``), unique-local
       IPv6 (``fc00::/7``), unspecified (``0.0.0.0``, ``::``).
    3. ``ipaddress`` lèvera ``ValueError`` pour un hostname non-IP ;
       dans ce cas on retombe sur le check littéral seul (la
       résolution DNS effective sera celle d'``urllib`` au moment
       du fetch — défense statique uniquement, pas anti-rebinding).
    """
    if hostname in _BLOCKED_HOSTNAMES:
        return True

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False

    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_unspecified
        or ip.is_reserved
        or ip.is_multicast
    )


def validate_http_url(url: str) -> None:
    """Valide une URL externe avant fetch — anti-SSRF statique.

    Sprint S1.6 — durcissement.  Refuse :

    - Schémas non-HTTP (``file://``, ``ftp://``, ``data:``,
      ``javascript:``, ``gopher://``, etc.).
    - URL sans hostname (``http:///``).
    - Hostname dans :data:`_BLOCKED_HOSTNAMES` (``localhost``,
      ``metadata.google.internal``, ...).
    - IP littérale dans loopback / lien-local / privé RFC 1918 /
      unspecified / réservé / multicast.

    Limite explicite : cette fonction est une **défense statique**.
    Elle ne protège pas contre :

    - DNS rebinding (un FQDN public qui résout vers 127.0.0.1
      après TOCTOU).
    - Redirections HTTP qui pointent vers du loopback (utiliser
      ``allow_redirects=False`` côté caller, ou re-valider chaque
      hop).

    Pour ces deux derniers, le caller doit prendre des mesures
    complémentaires (résolveur custom, pas de redirection auto).

    Raises
    ------
    ValueError
        Si l'URL ne passe pas la validation.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Schéma URL non autorisé '{parsed.scheme}' "
            f"(seuls http/https sont acceptés) : {url}"
        )
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError(f"URL sans hostname : {url}")
    if _is_blocked_host(hostname):
        raise ValueError(
            f"Hostname '{hostname}' refusé : adresse interne, "
            f"loopback ou métadonnée cloud (anti-SSRF) — {url}"
        )


def download_url(
    url: str,
    *,
    retries: int = 4,
    backoff: float = 2.0,
    timeout: int = 60,
    user_agent: str = _DEFAULT_USER_AGENT,
    extra_headers: Optional[dict[str, str]] = None,
) -> bytes:
    """Télécharge une URL avec retry exponentiel.

    Parameters
    ----------
    url:
        URL à télécharger. Validée par :func:`validate_http_url`.
    retries:
        Nombre total de tentatives (défaut 4).
    backoff:
        Base du backoff exponentiel : attente = ``backoff ** attempt``
        secondes (défaut 2.0 → 0, 2, 4, 8 s).
    timeout:
        Timeout HTTP par tentative en secondes (défaut 60).
    user_agent:
        Header ``User-Agent`` envoyé. Défaut : Picarones identifié.
    extra_headers:
        Headers supplémentaires (ex : ``{"Accept": "application/json"}``).

    Raises
    ------
    ValueError
        Si l'URL n'a pas un schéma autorisé.
    RuntimeError
        Si toutes les tentatives échouent.
    """
    validate_http_url(url)
    headers = {"User-Agent": user_agent}
    if extra_headers:
        headers.update(extra_headers)
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        if attempt > 0:
            wait = backoff ** attempt
            logger.debug(
                "Retry %d/%d dans %.1fs — %s",
                attempt, retries - 1, wait, url,
            )
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_exc = exc
            logger.warning("Erreur téléchargement %s : %s", url, exc)
    raise RuntimeError(
        f"Impossible de télécharger {url} après {retries} tentatives",
    ) from last_exc


__all__ = ["validate_http_url", "download_url"]
