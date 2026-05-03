"""CLI Click — Sprint S22.

Cible : déplacement (sans réécriture lourde) de
``picarones.cli.{run,report,demo,serve,...}``.  Chaque commande
Click devient un mince wrapper autour d'un service de ``app/``.

Règle : pas d'accès direct à un moteur OCR ou à un calcul de
métrique depuis une commande Click.  Si tu écris ``from
picarones.engines.tesseract import ...`` dans une commande, c'est
un signal qu'il manque un service côté ``app/``.
"""

from __future__ import annotations

__all__: list[str] = []
