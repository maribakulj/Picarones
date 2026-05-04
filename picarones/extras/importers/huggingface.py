"""Re-export — Sprint A14-S11. Le contenu canonique vit dans
``picarones.adapters.corpus.huggingface``.

Ré-expose explicitement ``_REFERENCE_DATASETS`` (importé par les
tests web).
"""

from __future__ import annotations

from picarones.adapters.corpus.huggingface import *  # noqa: F401,F403
from picarones.adapters.corpus.huggingface import _REFERENCE_DATASETS  # noqa: F401
