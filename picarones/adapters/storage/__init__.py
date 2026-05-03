"""Adaptateurs de stockage — Sprint S20.

Cible : déplacement de ``picarones.web.jobs`` (SQLite job store) et
de ``picarones.measurements.history`` (SQLite history).  Plus
généralement : tout ce qui touche au filesystem ou à une DB locale.

Pattern : un ``Storage`` est instancié par un ``app/services/``,
pas créé ad-hoc dans un router FastAPI ou un module métier.  Ça
permet d'injecter un mock en test, de basculer SQLite → Postgres
si besoin, et de centraliser les permissions/quotas.
"""

from __future__ import annotations

__all__: list[str] = []
