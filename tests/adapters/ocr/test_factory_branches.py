"""Sprint S8.6 — couverture des branches d'erreur de ``ocr_adapter_from_name``.

Avant : 28% (uniquement la branche ``precomputed`` testée par
``tests/adapters/ocr/test_factory.py``).

Cible : 80%+ — couvre les branches qui lèvent ``ValueError`` :
- nom inconnu,
- adapters cloud sans dépendance optionnelle (réellement simulés
  via ``patch.dict(sys.modules, {...: None})`` qui force le
  ``from picarones.adapters.ocr.<engine> import ...`` à lever
  ``ImportError`` — le pattern Python standard pour bloquer un
  import).

Pourquoi pas de ``try/except`` permissif côté test
--------------------------------------------------
Avant rewrite : les tests acceptaient ``ValueError`` *ou*
``TypeError`` *ou* l'instanciation réussie selon que le SDK était
installé localement, ce qui voulait dire que la branche d'erreur
n'était jamais réellement vérifiée — du coverage theater.  Le
rewrite force ``ImportError`` peu importe l'environnement et
asserte la *transformation* en ``ValueError`` avec le mot-clé
``indisponible`` (contrat documenté de la factory).
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from picarones.adapters.ocr.factory import ocr_adapter_from_name


class TestUnknownName:
    def test_unknown_adapter_raises(self) -> None:
        with pytest.raises(ValueError, match="inconnu|unknown|disponible"):
            ocr_adapter_from_name("xyz_not_a_real_engine")

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            ocr_adapter_from_name("")


class TestKnownAdaptersInstantiate:
    """Les adapters dont les SDK sont déjà installés doivent
    instancier sans ``ImportError``."""

    def test_tesseract_instantiates(self) -> None:
        # Tesseract est dans dependencies de base.
        adapter = ocr_adapter_from_name("tesseract")
        assert adapter.name == "tesseract"

    def test_precomputed_instantiates(self) -> None:
        adapter = ocr_adapter_from_name("precomputed", source_label="bnf")
        assert "precomputed_bnf" in adapter.name


class TestAliasesAccepted:
    def test_aliases_resolve_to_canonical(self) -> None:
        # Test que la table d'alias fonctionne (cas standard
        # ``Tesseract``/``TESSERACT``).
        adapter = ocr_adapter_from_name("Tesseract")
        assert adapter.name == "tesseract"


class TestOptionalDepsMissing:
    """Quand un module wrapper d'adapter cloud ne peut être importé
    (SDK manquant en environnement minimal), la factory doit
    transformer l'``ImportError`` en ``ValueError`` avec un message
    qui contient ``indisponible`` et nomme la dépendance optionnelle
    à installer.  Contrat : un caller (CLI ou web) doit pouvoir
    n'attraper que ``ValueError`` pour gérer la liste des moteurs
    disponibles dynamiquement, sans avoir à connaître la mécanique
    d'import des SDK cloud.

    Le ``patch.dict(sys.modules, {... : None})`` est le pattern
    standard pour faire échouer un ``from x import y`` : Python
    voit ``None`` dans ``sys.modules`` et lève
    ``ModuleNotFoundError`` (sous-classe d'``ImportError``).
    """

    @pytest.mark.parametrize(
        ("canonical_name", "module_path", "sdk_label"),
        [
            ("pero_ocr", "picarones.adapters.ocr.pero_ocr", "pero"),
            ("mistral_ocr", "picarones.adapters.ocr.mistral_ocr", "mistral"),
            (
                "google_vision",
                "picarones.adapters.ocr.google_vision",
                "google",
            ),
            (
                "azure_doc_intel",
                "picarones.adapters.ocr.azure_doc_intel",
                "azure",
            ),
        ],
    )
    def test_cloud_adapter_without_sdk_raises_value_error(
        self,
        canonical_name: str,
        module_path: str,
        sdk_label: str,
    ) -> None:
        with patch.dict(sys.modules, {module_path: None}):
            with pytest.raises(ValueError) as exc_info:
                ocr_adapter_from_name(canonical_name)

        msg = str(exc_info.value).lower()
        assert "indisponible" in msg, (
            f"message d'erreur attendu avec 'indisponible' pour "
            f"{canonical_name}, reçu : {msg!r}"
        )
        assert sdk_label in msg or canonical_name in msg, (
            f"message d'erreur doit nommer la dépendance "
            f"{sdk_label!r} ou l'adapter {canonical_name!r}, "
            f"reçu : {msg!r}"
        )

    def test_alias_resolution_then_sdk_missing(self) -> None:
        """L'alias court (``pero``) doit être résolu en canonique
        (``pero_ocr``) AVANT le check de dépendance — sinon la
        branche d'alias échoue silencieusement et la factory rapporte
        ``unknown engine`` à tort."""
        with patch.dict(
            sys.modules, {"picarones.adapters.ocr.pero_ocr": None},
        ):
            with pytest.raises(ValueError, match="indisponible"):
                ocr_adapter_from_name("pero")  # alias

    def test_imports_restored_after_patch(self) -> None:
        """Garantit que le ``patch.dict`` ne fuit pas entre tests :
        après contexte, le module redevient importable normalement.
        """
        with patch.dict(
            sys.modules, {"picarones.adapters.ocr.pero_ocr": None},
        ):
            with pytest.raises(ValueError, match="indisponible"):
                ocr_adapter_from_name("pero_ocr")
        # Hors patch : si Pero OCR est installé localement, l'import
        # doit fonctionner ; sinon on accepte l'erreur ``indisponible``
        # mais PAS celle d'un patch.dict resté actif.
        try:
            ocr_adapter_from_name("pero_ocr")
        except ValueError as exc:
            # OK : SDK pas installé localement, mais le message ne
            # doit PAS être ``halted; None in sys.modules`` (signe
            # que le patch a fui).
            assert "halted" not in str(exc).lower(), (
                f"le patch.dict a fui hors du contexte : {exc}"
            )
        except TypeError:
            # OK : SDK installé mais ``config_path`` requis.
            pass
