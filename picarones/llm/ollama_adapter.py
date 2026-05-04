"""Re-export — Sprint A14-S11. Le contenu canonique vit dans
``picarones.adapters.llm.ollama_adapter``.

L'ancien chemin ``picarones.llm.ollama_adapter`` est conservé pour ne casser
aucun consommateur.  Au S22, ce re-export disparaîtra.
"""

from __future__ import annotations

from picarones.adapters.llm.ollama_adapter import *  # noqa: F401,F403
