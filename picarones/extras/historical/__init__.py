"""Métriques philologiques pour documents historiques (Cercle 3).

Modules orientés cas d'usage patrimoniaux par période :

- :mod:`unicode_blocks`         — précision par bloc Unicode (toutes périodes)
- :mod:`abbreviations`          — score d'expansion d'abréviations (médiéval)
- :mod:`mufi`                   — couverture MUFI v4.0 (médiéval, PUA)
- :mod:`early_modern_typography` — ﬂ, ﬁ, ſ, ã, &, ı (XVIᵉ-XVIIIᵉ siècles)
- :mod:`modern_archives`        — Mme/Mlle/°/†/₶ (XIXᵉ-XXᵉ siècles)
- :mod:`roman_numerals`         — numéraux romains (toutes périodes)
- :mod:`lexical_modernization`  — top tokens GT modernisés par le moteur
- :mod:`philological_runner`    — orchestration adaptive des 6 modules

Utilité
-------
Ces métriques répondent à la question éditoriale *« quels caractères
historiques ce moteur restitue-t-il fidèlement ? »*. Elles ne
participent pas à la décision « peut-on déployer ce moteur en prod ? »
quand le corpus est moderne (les modules retournent ``None`` via
adaptive masking sur un texte sans signal philologique).

Plugin séparable
----------------
Distribué via l'extra pip ``picarones[historical]``. Les imports
historiques ``from picarones.core.unicode_blocks import ...`` restent
fonctionnels via des fichiers-shims dans :mod:`picarones.core`.

Phase B du chantier de refonte en 3 cercles — voir
:doc:`docs/architecture-cercles.md`.
"""
