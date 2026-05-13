"""Phase 1.1 du plan d'audit — l'adapter eScriptorium passe
désormais par ``validate_http_url`` pour les fetch GET/POST et par
``download_url`` pour les téléchargements d'images.

Audit code-quality (2026-05) : ``escriptorium._get/_post`` et le
``urllib.request.urlretrieve(part.image_url)`` ligne 410 fetchaient
sans valider l'URL — un manifeste pointant
``http://169.254.169.254/...`` exfiltrait les métadonnées cloud,
``http://127.0.0.1:6379/...`` parlait au Redis local, etc.  Le
helper ``validate_http_url`` existait déjà pour IIIF/Gallica/
HTR-United mais n'était pas branché pour eScriptorium.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from picarones.adapters.corpus.escriptorium import EScriptoriumClient


@pytest.fixture
def client() -> EScriptoriumClient:
    """Client eScriptorium configuré sur un hôte fictif valide.

    Le constructeur n'effectue aucun fetch — on peut donc fabriquer
    un client avec une URL publique fictive et tester les méthodes
    individuellement.
    """
    return EScriptoriumClient("https://escriptorium.example.org", token="dummy")


# --------------------------------------------------------------------------
# _get / _post : hostnames bloqués
# --------------------------------------------------------------------------


class TestGetBlocksDangerousHosts:
    """``_get`` doit refuser les hostnames internes avant tout fetch."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://169.254.169.254",          # AWS metadata
            "http://metadata.google.internal",  # GCP metadata
            "http://10.0.0.42",                 # RFC 1918
            "http://192.168.1.1",               # RFC 1918
            "http://172.16.0.5",                # RFC 1918
            "http://0.0.0.0",                   # unspecified
        ],
    )
    def test_get_refuses_internal_host(self, base_url: str) -> None:
        """Chaque IP/host interne fait lever RuntimeError sans fetch."""
        client = EScriptoriumClient(base_url, token="dummy")
        with patch("urllib.request.urlopen") as mock_urlopen:
            with pytest.raises(RuntimeError, match="(anti-SSRF|refusé|Schéma)"):
                client._get("projects/")
            # Le fetch ne doit jamais avoir lieu.
            mock_urlopen.assert_not_called()

    def test_get_refuses_file_scheme(self) -> None:
        """Le schéma ``file://`` est refusé avant fetch."""
        client = EScriptoriumClient("file:///etc/passwd", token="dummy")
        with patch("urllib.request.urlopen") as mock_urlopen:
            with pytest.raises(RuntimeError):
                client._get("anything")
            mock_urlopen.assert_not_called()


class TestPostBlocksDangerousHosts:
    """``_post`` (création de couche OCR) doit aussi valider."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://169.254.169.254",
            "http://localhost",
            "http://10.0.0.1",
        ],
    )
    def test_post_refuses_internal_host(self, base_url: str) -> None:
        client = EScriptoriumClient(base_url, token="dummy")
        with patch("urllib.request.urlopen") as mock_urlopen:
            with pytest.raises(RuntimeError, match="(anti-SSRF|refusé|Schéma)"):
                client._post("documents/1/parts/2/transcriptions/", {"key": "value"})
            mock_urlopen.assert_not_called()


# --------------------------------------------------------------------------
# Image download via download_url (Phase 1.1) — anti-SSRF
# --------------------------------------------------------------------------


class TestImageDownloadValidatesURL:
    """``import_document`` doit refuser de fetch une image dont
    l'``image_url`` pointe vers un hôte interne.

    On teste ici uniquement la sous-routine qui télécharge l'image
    (le helper ``download_url`` lève ``ValueError`` validate_http_url).
    """

    def test_download_url_rejects_metadata_host(self) -> None:
        """Vérification directe de l'invariant : download_url
        ne fetch pas une URL metadata cloud."""
        from picarones.adapters.corpus._http import download_url

        with patch("urllib.request.urlopen") as mock_urlopen:
            with pytest.raises(ValueError, match="(anti-SSRF|refusé)"):
                download_url("http://169.254.169.254/latest/meta-data/")
            mock_urlopen.assert_not_called()


# --------------------------------------------------------------------------
# Garde-fou — l'import du module ne plante pas
# --------------------------------------------------------------------------


def test_module_imports_validate_http_url() -> None:
    """Le module ``escriptorium`` doit avoir importé ``validate_http_url``
    au top-level — protection contre une régression d'import lazy
    qui contournerait la vérification.
    """
    import picarones.adapters.corpus.escriptorium as mod

    assert hasattr(mod, "validate_http_url"), (
        "escriptorium.py n'importe plus validate_http_url — "
        "régression Phase 1.1 de l'audit code-quality."
    )
    assert hasattr(mod, "download_url"), (
        "escriptorium.py n'importe plus download_url — "
        "régression Phase 1.1 de l'audit code-quality."
    )
