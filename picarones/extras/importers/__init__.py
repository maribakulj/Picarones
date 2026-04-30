"""Importeurs de corpus depuis sources distantes (Cercle 3).

Phase C du chantier de refonte en 3 cercles.

Importeurs livrés
-----------------
- :mod:`_http`         — helpers HTTP partagés (validate_http_url, download_url)
- :mod:`iiif`          — manifestes IIIF v2/v3 (Bodleian, BnF, Vatican…)
- :mod:`htr_united`    — datasets HTR-United (CC0, GitHub)
- :mod:`gallica`       — BnF Gallica (SRU + IIIF + OCR brut)
- :mod:`huggingface`   — datasets HuggingFace ⚠ **expérimental**
- :mod:`escriptorium`  — projets eScriptorium ⚠ **expérimental**

Modules expérimentaux
---------------------
``huggingface`` et ``escriptorium`` émettent un ``UserWarning`` à
l'import. Ils sont fonctionnellement présents mais leur usage en
production n'est pas garanti — l'API HuggingFace Datasets évolue
fréquemment et eScriptorium n'a qu'un test isolé. À utiliser à vos
risques jusqu'à ce qu'un cas d'usage institutionnel valide leur API.

Plugin séparable
----------------
Distribué via l'extra pip ``picarones[importers]``. Les imports
historiques ``from picarones.importers.iiif import ...`` restent
fonctionnels via des fichiers-shims dans :mod:`picarones.importers`.
"""
