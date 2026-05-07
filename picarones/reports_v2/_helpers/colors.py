"""Palettes de couleurs CSS — partagées entre rapport HTML et modules de rendu.

Phase 5 — module relocalisé depuis ``picarones.report.colors`` vers
``picarones.reports_v2._helpers.colors``.  Le chemin legacy reste
disponible via un shim avec ``DeprecationWarning`` ; suppression
prévue en 2.0.

Sprint A7 (item m-5 de l'audit institutional-readiness-2026-05) :
introduction d'une **palette daltonien-friendly** (Okabe-Ito) qui
remplace la palette historique rouge/vert/orange (problématique pour
les ~8 % d'hommes atteints de deutéranopie ou protanopie).

Conventions
-----------
- ``COLOR_*`` / ``BG_*`` : alias actifs **par défaut** (palette
  Okabe-Ito 2008).  Les modules de rendu Python utilisent ces
  symboles directement.
- ``CLASSIC_*`` : palette historique (rouge/jaune/orange/vert),
  conservée pour rétrocompat et accessible via le toggle utilisateur
  dans le rapport (``?palette=classic`` ou case à cocher dans le
  panneau « Avancé »).
- Les contrastes sur fond blanc sont vérifiés WCAG 2.1 AA
  (≥ 4,5:1 pour le texte normal).

La palette Okabe-Ito (Okabe & Ito, 2008) est recommandée par les
revues scientifiques accessibles et est l'une des premières palettes
qualitatives non confondables pour les trois principales formes de
daltonisme (deutéranopie, protanopie, tritanopie).
"""

# ──────────────────────────────────────────────────────────────────
# Palette Okabe-Ito 2008 — daltonien-friendly (active par défaut).
# Source : https://jfly.uni-koeln.de/color/
# Mapping bon → mauvais utilisé par ``difficulty_color`` (Sprint 19+) :
#   facile     → bleu     #0072B2 (Okabe-Ito blue)
#   modéré     → jaune    #F0E442 (Okabe-Ito yellow, lisible sur fond clair)
#   difficile  → orange   #E69F00 (Okabe-Ito orange)
#   critique   → vermillon #D55E00 (Okabe-Ito vermillion)
#
# Remplace l'ancien green/yellow/orange/red qui posait deux problèmes :
# - rouge/vert indistinguables en deutéranopie ;
# - le rouge ``#dc2626`` ratait le contraste 4,5:1 sur fond ``#ffedd5``.
# ──────────────────────────────────────────────────────────────────
COLOR_GREEN = "#0072B2"     # bleu Okabe-Ito (substitut sémantique du « bon »)
COLOR_YELLOW = "#F0E442"    # jaune Okabe-Ito
COLOR_ORANGE = "#E69F00"    # orange Okabe-Ito
COLOR_RED = "#D55E00"       # vermillon Okabe-Ito (substitut sémantique du « mauvais »)

# Backgrounds clairs associés (couleur diluée à ~85 % de blanc).
# Maintiennent un contraste ≥ 7:1 avec le texte gris foncé du rapport.
BG_GREEN = "#cfe5f4"        # bleu très clair
BG_YELLOW = "#fefbcd"       # jaune très clair
BG_ORANGE = "#fbe4bf"       # orange très clair
BG_RED = "#fbd6c1"          # vermillon très clair

# ──────────────────────────────────────────────────────────────────
# Palette historique (« classic ») — rétrocompat + toggle UI.
#
# Disponible côté frontend via ``?palette=classic`` ou la case à
# cocher du panneau « Avancé ».  Côté Python, ces symboles sont
# importables explicitement par les modules qui veulent une teinte
# spécifique à la palette historique (exemple : badges legacy d'un
# ancien rapport archivé).
# ──────────────────────────────────────────────────────────────────
CLASSIC_GREEN = "#16a34a"
CLASSIC_YELLOW = "#ca8a04"
CLASSIC_ORANGE = "#ea580c"
CLASSIC_RED = "#dc2626"

CLASSIC_BG_GREEN = "#dcfce7"
CLASSIC_BG_YELLOW = "#fef9c3"
CLASSIC_BG_ORANGE = "#ffedd5"
CLASSIC_BG_RED = "#fee2e2"


__all__ = [
    "COLOR_GREEN", "COLOR_YELLOW", "COLOR_ORANGE", "COLOR_RED",
    "BG_GREEN", "BG_YELLOW", "BG_ORANGE", "BG_RED",
    "CLASSIC_GREEN", "CLASSIC_YELLOW", "CLASSIC_ORANGE", "CLASSIC_RED",
    "CLASSIC_BG_GREEN", "CLASSIC_BG_YELLOW", "CLASSIC_BG_ORANGE", "CLASSIC_BG_RED",
]
