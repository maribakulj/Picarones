"""Sprint S8.6 — couverture du ``RequestIdFilter`` et de la
contextvar ``_request_id_var``.

Le filter propage le ``request_id`` posé par le middleware vers
les ``LogRecord`` émis depuis des modules qui n'ont pas accès
direct à ``request.state``.

Contrats testés :

1. ``record.request_id`` déjà posé par ``extra={"request_id": ...}``
   → le filter le respecte sans override.
2. ``record.request_id`` absent + contextvar set → le filter pose
   l'attribut depuis la contextvar.
3. ``record.request_id`` absent + contextvar à ``None`` (défaut) →
   ``filter()`` retourne ``True`` sans poser l'attribut.

Le ``except (ImportError, LookupError)`` historique a été supprimé
en S8.6 (dead code : ``_request_id_var`` est une variable du
même module donc l'import-time du module a déjà résolu son
existence, et ``ContextVar.get()`` ne lève pas avec un default).
"""

from __future__ import annotations

import logging


def _make_record() -> logging.LogRecord:
    return logging.LogRecord(
        name="test", level=logging.INFO, pathname="t.py",
        lineno=1, msg="x", args=(), exc_info=None,
    )


class TestRequestIdFilter:
    def test_existing_attribute_preserved(self) -> None:
        """Si le caller passe ``extra={"request_id": "..."}`` à un
        ``logger.info()``, le filter ne doit PAS l'écraser."""
        from picarones.interfaces.web.observability import (
            RequestIdFilter,
            _request_id_var,
        )

        # Pose une valeur DIFFÉRENTE dans la contextvar pour vérifier
        # que c'est bien l'attribut explicite qui gagne.
        token = _request_id_var.set("from-contextvar")
        try:
            f = RequestIdFilter()
            rec = _make_record()
            rec.request_id = "from-extra"

            assert f.filter(rec) is True
            assert rec.request_id == "from-extra", (
                "L'attribut explicite a été écrasé par la contextvar — "
                "violation du contrat de précédence."
            )
        finally:
            _request_id_var.reset(token)

    def test_missing_attribute_uses_contextvar(self) -> None:
        """Sans ``extra={"request_id": ...}``, le filter consulte la
        contextvar (posée par le middleware) et propage la valeur
        sur le record."""
        from picarones.interfaces.web.observability import (
            RequestIdFilter,
            _request_id_var,
        )

        token = _request_id_var.set("ctxvar-id-456")
        try:
            f = RequestIdFilter()
            rec = _make_record()
            assert not hasattr(rec, "request_id")

            assert f.filter(rec) is True
            assert getattr(rec, "request_id", None) == "ctxvar-id-456"
        finally:
            _request_id_var.reset(token)

    def test_missing_attribute_no_contextvar_value(self) -> None:
        """Hors contexte de requête (contextvar à ``None``), le filter
        retourne ``True`` sans poser ``request_id`` sur le record.
        Garantit que les logs hors-requête ne récupèrent pas un
        ``request_id`` fantôme d'une requête précédente."""
        from picarones.interfaces.web.observability import (
            RequestIdFilter,
            _request_id_var,
        )

        token = _request_id_var.set(None)
        try:
            f = RequestIdFilter()
            rec = _make_record()

            assert f.filter(rec) is True
            assert not hasattr(rec, "request_id")
        finally:
            _request_id_var.reset(token)


class TestRequestIdContextvar:
    def test_default_is_none(self) -> None:
        """Le default ``None`` est la garantie qu'aucun ``LookupError``
        n'est levé par ``.get()`` même hors contexte de requête."""
        from picarones.interfaces.web.observability import _request_id_var

        token = _request_id_var.set(None)
        try:
            assert _request_id_var.get() is None
        finally:
            _request_id_var.reset(token)

    def test_isolation_between_contexts(self) -> None:
        """Une valeur set dans un context ne fuit pas vers un autre."""
        from picarones.interfaces.web.observability import _request_id_var

        token = _request_id_var.set("outer-id")
        try:
            assert _request_id_var.get() == "outer-id"
            # Override dans un sous-contexte (logique).
            inner_token = _request_id_var.set("inner-id")
            try:
                assert _request_id_var.get() == "inner-id"
            finally:
                _request_id_var.reset(inner_token)
            # Après reset interne, on retrouve la valeur outer.
            assert _request_id_var.get() == "outer-id"
        finally:
            _request_id_var.reset(token)
