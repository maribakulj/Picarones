"""Configuration pytest pour les tests d'intégration *live*.

Sprint A14-S55 (fix audit #9) : les 13 adapters (5 OCR + 4 LLM +
4 VLM) n'avaient aucun test contre une vraie API ni un vrai binaire
système.  Tous les tests étaient mockés.  Un upgrade silencieux de
l'API tierce (changement de schéma JSON, breaking dans un SDK) ne
pouvait être détecté qu'à la livraison BnF.

Ce sous-package contient les tests **live** :

- skippés gracieusement si l'API ou le binaire est absent ;
- vérifient le contrat bout-en-bout (input → API → output) sans
  assertion de qualité ;
- non exécutés en CI par défaut — opt-in via la marker ``live``.

Usage
-----

::

    # En local avec les bonnes variables d'env :
    pytest tests/integration/live/ -v
    pytest tests/integration/live/ -m live -v

    # Pour exécuter UN adapter spécifique :
    pytest tests/integration/live/test_tesseract_live.py -v

Marker
------
Les tests live portent la marker ``@pytest.mark.live`` pour qu'un
``pytest -m 'not live'`` les skipe automatiquement (utile en CI
standard).
"""

from __future__ import annotations



def pytest_configure(config) -> None:
    """Enregistre le marker ``live`` (évite UnknownMarkerWarning)."""
    config.addinivalue_line(
        "markers",
        "live: tests d'intégration contre vraie API/binaire (skip si "
        "credentials absents).  Opt-out via -m 'not live'.",
    )
