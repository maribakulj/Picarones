"""Helpers partagés entre les renderers de ``reports_v2``.

Ce sous-package abrite les utilitaires purs et stables :

- ``colors`` — palettes de couleurs (Okabe-Ito + classique).
- ``diff_utils`` — calcul de diff char/word (re-export de ``evaluation``).
- ``render_helpers`` — fonctions de rendu HTML/CSS communes.
- ``assets`` — bundling JS + CSS + glossaire dans le rapport
  autonome.

Phase 5 du retrait du legacy.  Ces modules viennent de
``picarones.report.*`` ; les chemins legacy restent disponibles
via des shims avec ``DeprecationWarning`` jusqu'à ce que tous les
renderers thématiques aient migré.
"""

from __future__ import annotations

__all__: list[str] = []
