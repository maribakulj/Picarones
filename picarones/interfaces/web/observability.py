"""Sprint S6.5 / S6.6 — observabilité institutionnelle.

Deux briques pour rendre l'app exploitable par une équipe ops BnF :

1. ``JsonLogFormatter`` — sortie logs au format JSON, indexable par
   ELK / Splunk / Datadog sans grep.  Champs : ``timestamp``,
   ``level``, ``logger``, ``message``, ``request_id`` (si dans le
   contexte), ``exc_info`` aplati.

2. ``request_id_middleware`` — assigne un identifiant unique à
   chaque requête HTTP (ou récupère ``X-Request-Id`` si fourni
   par le reverse-proxy), le pose dans ``request.state.request_id``
   et l'expose en header de réponse.  Le handler global
   d'exceptions (Sprint S3.2) le réutilise dans le payload 500.

Activation
----------

Les deux sont opt-in via env vars :

- ``PICARONES_LOG_FORMAT=json`` → installe le formatter JSON sur
  le root logger.
- Le middleware request_id est toujours actif (coût quasi-nul,
  utile en debug même hors prod).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from fastapi import Request


# ──────────────────────────────────────────────────────────────────────
# JSON log formatter
# ──────────────────────────────────────────────────────────────────────


class JsonLogFormatter(logging.Formatter):
    """Formatter logging stdlib qui sérialise chaque record en JSON
    sur une seule ligne.

    Format minimal mais exploitable :

    .. code-block:: json

        {"timestamp": "2026-05-09T12:00:00.123Z", "level": "INFO",
         "logger": "picarones.web.app", "message": "...",
         "request_id": "abc123def456"}

    Champs additionnels : ``exc_type`` + ``exc_message`` aplatis si
    ``exc_info`` est présent (la stack trace complète reste dans
    ``stack`` pour les ingesters qui la veulent).
    """

    def format(self, record: logging.LogRecord) -> str:
        # ``record.created`` est un timestamp UNIX float ; on génère
        # un ISO 8601 UTC compatible cross-OS sans dépendre de
        # ``time.strftime("%f")`` (non supporté sur Windows).
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S") + (
            f".{int(record.msecs):03d}Z"
        )
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Le middleware request_id pose ``request_id`` sur les records
        # via un filter (cf. ``RequestIdFilter`` ci-dessous).
        rid = getattr(record, "request_id", None)
        if rid:
            payload["request_id"] = rid

        # Exception info aplatie pour les ingesters JSON.
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exc_type"] = exc_type.__name__ if exc_type else "Unknown"
            payload["exc_message"] = str(exc_value) if exc_value else ""
            payload["stack"] = self.formatException(record.exc_info)

        # Extras passés via ``logger.info("msg", extra={"key": value})``.
        # On évite d'écraser les champs ci-dessus.
        reserved = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process",
            "request_id", "message", "asctime",
        }
        for key, value in record.__dict__.items():
            if key not in reserved and not key.startswith("_"):
                # Skip non-JSON-serializable values silently.
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = repr(value)

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def install_json_logging() -> None:
    """Installe ``JsonLogFormatter`` sur le root logger.

    Appelé au démarrage de l'app si ``PICARONES_LOG_FORMAT=json``.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    # Remplace les handlers existants (uvicorn pose un handler par
    # défaut avec format texte).
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def is_json_logging_requested() -> bool:
    return os.environ.get("PICARONES_LOG_FORMAT", "").strip().lower() == "json"


# ──────────────────────────────────────────────────────────────────────
# Request-Id middleware
# ──────────────────────────────────────────────────────────────────────

#: Header standard exposé par les reverse-proxies (nginx, traefik,
#: Cloudflare).  Si déjà fourni → on respecte (tracing distributé).
REQUEST_ID_HEADER = "x-request-id"


async def request_id_middleware(
    request: Request, call_next,
):
    """Pose ``request.state.request_id`` et l'expose en header.

    Logique :

    1. Si le client (ou un reverse-proxy en amont) fournit déjà
       ``X-Request-Id``, on l'adopte (avec un cap de 64 chars pour
       éviter le log injection via un header gigantesque).
    2. Sinon, génère un UUID4 hex tronqué à 12 chars (compromis
       lisibilité / unicité).
    3. Pose sur ``request.state.request_id`` pour les handlers et
       sur la réponse en header ``X-Request-Id``.
    """
    incoming = request.headers.get(REQUEST_ID_HEADER, "").strip()
    if (
        incoming
        and len(incoming) <= 64
        # Anti log injection : refus des caractères de contrôle
        # (newline, NUL, etc.) qui pourraient corrompre le log
        # structuré.  On accepte ASCII imprimable + tiret/underscore.
        and all(c.isprintable() and ord(c) < 128 for c in incoming)
    ):
        rid = incoming
    else:
        rid = uuid.uuid4().hex[:12]
    request.state.request_id = rid

    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = rid
    return response


# ──────────────────────────────────────────────────────────────────────
# Filter pour propager request_id aux logs
# ──────────────────────────────────────────────────────────────────────


class RequestIdFilter(logging.Filter):
    """Logging filter qui injecte ``record.request_id`` à partir
    d'une variable contextvars (mise à jour par le middleware).

    Utile quand un endpoint déclenche un log via un module qui n'a
    pas accès directement à ``request.state``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Si l'attribut a déjà été posé (extra={"request_id": ...})
        # on respecte ; sinon on tente la contextvar.
        if not hasattr(record, "request_id"):
            try:
                from picarones.interfaces.web.observability import (
                    _request_id_var,
                )
                rid = _request_id_var.get()
                if rid:
                    record.request_id = rid
            except (ImportError, LookupError):
                pass
        return True


import contextvars  # noqa: E402

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "picarones_request_id", default=None,
)


__all__ = [
    "JsonLogFormatter",
    "REQUEST_ID_HEADER",
    "RequestIdFilter",
    "install_json_logging",
    "is_json_logging_requested",
    "request_id_middleware",
]
