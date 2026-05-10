"""Helpers partagés par les tests ``tests/measurements/``.

Centralise des utilitaires de test réutilisés par plusieurs fichiers
sprint, en particulier le garde-fou anti-hallucination du moteur
narratif.
"""

from __future__ import annotations

import re
from typing import Any


def numbers_in_payload(payload: Any) -> set[str]:
    """Collecte toutes les représentations numériques d'un payload de Fact.

    Inclut les variantes usuelles produites par ``str.format`` :
    ``5``, ``5.0``, ``5.00``, ``5.000``, ``5.0000``, etc., pour tolérer
    les patterns ``{x}`` et ``{x:.2f}`` dans les templates narratifs.

    Le walk est récursif (dict / list / tuple) et parse également les
    chaînes via ``\\d+(?:\\.\\d+)?`` pour couvrir les payloads où une
    valeur numérique a été pré-formatée en string.
    """
    out: set[str] = set()

    def _add_variants(v: Any) -> None:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        out.add(str(v))
        out.add(str(f))
        if f == int(f):
            out.add(str(int(f)))
        for dec in (1, 2, 3, 4):
            out.add(f"{f:.{dec}f}")

    def _walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _walk(v)
        elif isinstance(x, bool):
            return
        elif isinstance(x, (int, float)):
            _add_variants(x)
        elif isinstance(x, str):
            for n in re.findall(r"\d+(?:\.\d+)?", x):
                _add_variants(n)

    _walk(payload)
    return out


__all__ = ["numbers_in_payload"]
