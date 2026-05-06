"""Re-export — Sprint A14-S11. Le contenu canonique vit dans
``picarones.adapters.llm.mistral_adapter``.

Ré-expose explicitement ``_TEXT_ONLY_MODELS`` (importé par les
tests Sprint 15).
"""

from __future__ import annotations

from picarones.adapters.llm.mistral_adapter import *  # noqa: F401,F403
from picarones.adapters.llm.mistral_adapter import _TEXT_ONLY_MODELS  # noqa: F401
