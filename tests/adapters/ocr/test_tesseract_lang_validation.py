"""Phase 1.2 du plan d'audit — TesseractAdapter valide le format
de ``lang`` à la construction (refuse les injections CLI).

Risque parée : ``lang`` est in fine concaténé par pytesseract à la
ligne de commande ``tesseract -l <lang>``.  Sans validation, un
appelant qui passe ``lang="fra --user-words /etc/passwd"`` lirait
un fichier arbitraire (Tesseract honore ce flag).

La validation côté UI (``get_tesseract_langs()``) protégeait le
chemin web, mais pas les usages programmatiques ni la CLI.  Phase
1.2 ajoute une défense locale dans ``__init__``.
"""

from __future__ import annotations

import pytest

from picarones.adapters.ocr.base import OCRAdapterError
from picarones.adapters.ocr.tesseract import TesseractAdapter


class TestTesseractLangAccepted:
    """Codes Tesseract canoniques acceptés."""

    @pytest.mark.parametrize(
        "lang",
        [
            "fra",
            "eng",
            "lat",
            "frk",         # Fraktur
            "deu",
            "fra+eng",     # combinaison standard
            "lat+deu+eng",
            "Latin",       # script (3+ lettres)
            "Cyrillic",
        ],
    )
    def test_valid_lang_accepted(self, lang: str) -> None:
        adapter = TesseractAdapter(lang=lang)
        assert adapter.lang == lang


class TestTesseractLangRejected:
    """Toute valeur exploitable pour injection CLI doit lever."""

    @pytest.mark.parametrize(
        "lang",
        [
            # Injection classique : un espace permet d'ajouter un flag
            # Tesseract qui lit un fichier arbitraire.
            "fra --user-words /etc/passwd",
            "fra --tessdata-dir /tmp",
            # Doubles tirets sans espace = même attaque.
            "fra--user-words",
            # Slash : chemin / path traversal.
            "fra/eng",
            "../etc",
            # Caractères de séparation shell.
            "fra;ls",
            "fra|cat",
            "fra`whoami`",
            "fra$IFS",
            "fra\nrm",
            # Vide ou trop court.
            "",
            "f",
            "fr",
            # Caractères non-ASCII (peuvent contourner la regex naive).
            "frà",
            "français",
            # Combinaison mal formée.
            "fra+",
            "+fra",
            "fra++eng",
            # Avec chiffres (pas un code ISO 639-3).
            "fra1",
            "1fra",
        ],
    )
    def test_invalid_lang_raises(self, lang: str) -> None:
        with pytest.raises(OCRAdapterError, match="lang invalide"):
            TesseractAdapter(lang=lang)


def test_default_lang_is_valid() -> None:
    """Régression : le défaut ``"fra"`` doit toujours passer la
    validation (sinon TesseractAdapter() planterait sans
    arguments).
    """
    adapter = TesseractAdapter()
    assert adapter.lang == "fra"
