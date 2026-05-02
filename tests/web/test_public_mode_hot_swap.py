"""Tests Sprint A5 — bascule à chaud du mode public (M-13).

Le mode public est piloté par la variable d'environnement
``PICARONES_PUBLIC_MODE``. ``picarones.web.security.is_public_mode()``
la lit à **chaque appel** plutôt qu'au démarrage, ce qui permet à un
opérateur de basculer le mode sans redémarrer le serveur.

Cette suite vérifie que la bascule à chaud fonctionne :

1. Au démarrage en mode dev, ``assert_engines_allowed`` accepte les
   moteurs cloud ; après ``setenv PICARONES_PUBLIC_MODE=1``, le même
   appel les refuse.
2. Inversement : démarrage public → bascule dev → cloud autorisé.
3. Aucun cache global ne mémorise l'ancienne valeur.
"""

from __future__ import annotations

import pytest

from picarones.web.security import (
    assert_engines_allowed,
    assert_llm_provider_allowed,
    is_public_mode,
)


def test_public_mode_off_allows_cloud_engines(monkeypatch) -> None:
    """Mode dev : moteurs cloud autorisés sans réserve."""
    monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
    assert is_public_mode() is False
    # Ne doit pas lever
    assert_engines_allowed(["mistral_ocr", "google_vision", "azure_doc_intel"])


def test_public_mode_on_blocks_cloud_engines(monkeypatch) -> None:
    """Mode public : moteurs cloud refusés (clés mutualisées côté serveur)."""
    monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
    assert is_public_mode() is True
    with pytest.raises(PermissionError):
        assert_engines_allowed(["mistral_ocr"])


def test_hot_swap_dev_to_public(monkeypatch) -> None:
    """Bascule à chaud dev → public. Le même appel passe puis échoue
    sans redémarrage du process."""
    monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
    # Phase 1 : dev → cloud autorisé
    assert_engines_allowed(["mistral_ocr"])  # ne lève pas

    # Phase 2 : bascule à chaud
    monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
    with pytest.raises(PermissionError):
        assert_engines_allowed(["mistral_ocr"])


def test_hot_swap_public_to_dev(monkeypatch) -> None:
    """Bascule inverse : public → dev. Le même cloud refusé puis accepté."""
    monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
    with pytest.raises(PermissionError):
        assert_engines_allowed(["google_vision"])

    monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
    assert_engines_allowed(["google_vision"])  # ne lève pas


def test_hot_swap_llm_provider_check(monkeypatch) -> None:
    """``assert_llm_provider_allowed`` doit aussi être sensible à la
    bascule à chaud."""
    monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
    assert_llm_provider_allowed("openai")  # dev : ok

    monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
    with pytest.raises(PermissionError):
        assert_llm_provider_allowed("openai")


def test_engines_allowed_partial_block(monkeypatch) -> None:
    """En mode public, si la liste contient cloud + local, l'erreur
    doit identifier précisément quel(s) moteur(s) sont refusés."""
    monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
    with pytest.raises(PermissionError) as exc_info:
        assert_engines_allowed(["tesseract", "mistral_ocr", "pero_ocr"])
    msg = str(exc_info.value)
    # Le message doit mentionner le moteur cloud refusé (pour un
    # diagnostic clair côté frontend).
    assert "mistral_ocr" in msg


def test_empty_engine_list_passes_in_both_modes(monkeypatch) -> None:
    """Une liste vide ne doit jamais lever (même en mode public)."""
    monkeypatch.delenv("PICARONES_PUBLIC_MODE", raising=False)
    assert_engines_allowed([])

    monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
    assert_engines_allowed([])


def test_local_engines_always_allowed(monkeypatch) -> None:
    """Tesseract / Pero (locaux) ne doivent jamais être bloqués."""
    monkeypatch.setenv("PICARONES_PUBLIC_MODE", "1")
    assert_engines_allowed(["tesseract"])
    assert_engines_allowed(["pero_ocr"])
    assert_engines_allowed(["tesseract", "pero_ocr"])
