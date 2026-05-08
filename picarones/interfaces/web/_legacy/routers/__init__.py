"""Routers FastAPI par thématique.

Chaque module de ce package expose un ``router`` (APIRouter) qui
regroupe une famille cohérente d'endpoints. ``app.py`` se contente
de monter ces routers via ``app.include_router(...)``.

Découpage :

- :mod:`system`        ``/api/status``, ``/api/lang``
- :mod:`engines`       ``/api/engines``, ``/api/models/{provider}``
- :mod:`corpus`        ``/api/corpus/*``
- :mod:`normalization` ``/api/normalization/profiles``
- :mod:`config`        ``/api/config/save``, ``/api/config/load``
- :mod:`synthesis`     ``/api/benchmark/{job_id}/synthesis_preview``
- :mod:`history`       ``/api/history/regressions``
- :mod:`reports`       ``/api/reports``, ``/reports/{filename}``
- :mod:`importers`     ``/api/htr-united/*``, ``/api/huggingface/*``
- :mod:`benchmark`     ``/api/benchmark/*`` (start, status, cancel, stream, run)
- :mod:`home`          ``/`` (SPA)
"""
