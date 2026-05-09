"""Sprint S8.6 — couverture du ``RequestIdFilter`` + contextvar.

Avant : 85% (lignes 186-196 du filter non couvertes — la branche
contextvar fallback).

Cible : exercer le filter directement et vérifier que :
- l'attribut existant est respecté ;
- la contextvar est consultée si l'attribut manque ;
- le LookupError silencieux ne propage pas.
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
        from picarones.interfaces.web.observability import RequestIdFilter

        f = RequestIdFilter()
        rec = _make_record()
        rec.request_id = "explicit-id"

        assert f.filter(rec) is True
        assert rec.request_id == "explicit-id"

    def test_missing_attribute_uses_contextvar(self) -> None:
        from picarones.interfaces.web.observability import (
            RequestIdFilter,
            _request_id_var,
        )

        # Pose une valeur dans la contextvar.
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
        from picarones.interfaces.web.observability import (
            RequestIdFilter,
            _request_id_var,
        )

        # Reset la contextvar.
        token = _request_id_var.set(None)
        try:
            f = RequestIdFilter()
            rec = _make_record()

            # Le filter ne lève pas, retourne True, ne pose pas request_id.
            assert f.filter(rec) is True
            assert getattr(rec, "request_id", None) is None
        finally:
            _request_id_var.reset(token)


class TestRequestIdContextvar:
    def test_default_is_none(self) -> None:
        from picarones.interfaces.web.observability import _request_id_var

        # Reset pour un test isolé.
        token = _request_id_var.set(None)
        try:
            assert _request_id_var.get() is None
        finally:
            _request_id_var.reset(token)

    def test_set_and_get(self) -> None:
        from picarones.interfaces.web.observability import _request_id_var

        token = _request_id_var.set("xyz")
        try:
            assert _request_id_var.get() == "xyz"
        finally:
            _request_id_var.reset(token)
