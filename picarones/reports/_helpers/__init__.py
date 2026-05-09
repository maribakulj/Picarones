"""Helpers partagés entre les renderers de ``reports``.

Ce sous-package abrite les utilitaires purs et stables :

- ``colors`` — palettes de couleurs (Okabe-Ito + classique).
- ``diff_utils`` — calcul de diff char/word (re-export de ``evaluation``).
- ``render_helpers`` — fonctions de rendu HTML/CSS communes.
- ``assets`` — bundling JS + CSS + glossaire dans le rapport
  autonome.
"""

from __future__ import annotations

__all__: list[str] = []
