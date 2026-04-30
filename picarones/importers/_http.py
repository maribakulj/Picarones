"""Helpers HTTP partagés par les importeurs IIIF / Gallica / HTR-United.

Chantier 4 du plan d'évolution post-Sprint 97 — fusion Gallica vers IIIF.

Auparavant les fonctions ``_validate_url`` et ``_download_url`` étaient
dupliquées entre :mod:`picarones.importers.iiif` (lignes 310-344) et
:mod:`picarones.importers.gallica` (lignes 125-155). Le module Gallica
faisait 549 lignes dont une bonne partie réimplémentait les mêmes
abstractions HTTP que IIIF (validation de schéma, retry exponentiel,
gestion des codes HTTP).

Ce module privé centralise ces helpers. Les deux importeurs (et tout
nouveau importateur HTTP futur) les utilisent. Comportement public
inchangé — uniquement de la factorisation.
"""

from __future__ import annotations

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


def validate_http_url(url: str) -> None:
    """Lève ``ValueError`` si le schéma de l'URL n'est pas http/https.

    Garde-fou contre les URLs ``file://``, ``ftp://``, ``data:`` qui
    permettraient à un manifeste IIIF malveillant de lire des fichiers
    locaux ou de contourner la politique réseau.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Schéma URL non autorisé '{parsed.scheme}' "
            f"(seuls http/https sont acceptés) : {url}"
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
