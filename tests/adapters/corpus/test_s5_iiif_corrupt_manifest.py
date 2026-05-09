"""Sprint S5 — Tests de manifestes IIIF corrompus / malicieux.

Cas couverts :

- JSON tronqué (5 bytes seulement)
- JSON valide mais champs IIIF requis absents (``@context``,
  ``sequences``…)
- Manifeste qui pointe vers une URL d'image loopback (rejeté par
  validate_http_url côté téléchargement)
- Manifeste géant (> 10 Mo) — ne doit pas tout charger en mémoire
  sans limite explicite (xfail si la limite n'existe pas).
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest


# --------------------------------------------------------------------------
# 1. JSON tronqué
# --------------------------------------------------------------------------


class TestIIIFTruncatedJson:
    """Un manifeste tronqué doit lever ``ValueError`` avec un message
    explicite, pas une JSONDecodeError nue."""

    def test_5_bytes_truncated_raises_value_error(self):
        from picarones.adapters.corpus.iiif import _fetch_manifest

        # 5 bytes de JSON mal formé
        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"@co'
            mock_resp.__enter__ = lambda self: self
            mock_resp.__exit__ = lambda self, *a: None
            mock_urlopen.return_value = mock_resp

            with pytest.raises(ValueError) as exc_info:
                _fetch_manifest("https://example.org/manifest.json")
            # Doit mentionner JSON ou manifeste
            msg = str(exc_info.value).lower()
            assert "json" in msg or "manifeste" in msg or "manifest" in msg

    def test_empty_response_raises_value_error(self):
        from picarones.adapters.corpus.iiif import _fetch_manifest

        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b""
            mock_resp.__enter__ = lambda self: self
            mock_resp.__exit__ = lambda self, *a: None
            mock_urlopen.return_value = mock_resp

            with pytest.raises(ValueError):
                _fetch_manifest("https://example.org/manifest.json")


# --------------------------------------------------------------------------
# 2. JSON valide mais champs IIIF requis absents
# --------------------------------------------------------------------------


class TestIIIFMissingFields:
    """Un manifeste sans ``@context`` ni ``items``/``sequences`` doit
    pouvoir être détecté comme invalide par le parseur (ou produire 0
    canvases sans plantage)."""

    def test_no_context_no_sequences_yields_empty_canvases(self):
        from picarones.adapters.corpus.iiif import IIIFManifestParser

        # Manifeste valide JSON mais vide de toute donnée IIIF
        empty = {}
        parser = IIIFManifestParser(empty)
        canvases = parser.canvases()
        # Le parser ne doit pas planter sur un manifeste vide.
        # Acceptable : retour vide.
        assert canvases == []

    def test_missing_sequences_v2_yields_empty(self):
        from picarones.adapters.corpus.iiif import IIIFManifestParser

        # Manifeste v2-like sans sequences
        manifest = {
            "@context": "http://iiif.io/api/presentation/2/context.json",
            "@type": "sc:Manifest",
            "label": "doc sans pages",
        }
        parser = IIIFManifestParser(manifest)
        canvases = parser.canvases()
        assert canvases == []


# --------------------------------------------------------------------------
# 3. Manifeste avec URL d'image loopback
# --------------------------------------------------------------------------


class TestIIIFLoopbackImageURL:
    """Si le manifeste pointe une image vers ``http://127.0.0.1/...``,
    le téléchargement doit être bloqué par validate_http_url (anti-SSRF)."""

    def test_download_loopback_image_rejected(self):
        from picarones.adapters.corpus._http import download_url

        # Une URL d'image qui pointe vers loopback doit être refusée
        # avant la résolution réseau.
        with pytest.raises(ValueError) as exc_info:
            download_url("http://127.0.0.1/iiif/image/full/max/0/default.jpg")
        msg = str(exc_info.value).lower()
        assert "loopback" in msg or "ssrf" in msg or "interne" in msg or "127" in msg

    def test_fetch_manifest_loopback_url_rejected(self):
        from picarones.adapters.corpus.iiif import _fetch_manifest

        # Manifeste hébergé sur loopback : refus immédiat (anti-SSRF
        # statique côté validate_http_url).
        with pytest.raises(ValueError):
            _fetch_manifest("http://127.0.0.1/manifest.json")


# --------------------------------------------------------------------------
# 4. Manifeste géant (> 10 Mo)
# --------------------------------------------------------------------------


class TestIIIFOversizedManifest:
    """Un manifeste de plusieurs dizaines de Mo doit avoir une borne
    de taille pour éviter un DoS mémoire.

    Si la borne n'existe pas dans le code actuel, ce test est marqué
    ``xfail`` pour signaler explicitement l'absence de la fonctionnalité
    (sans casser la suite ni masquer le problème).
    """

    def test_oversized_manifest_should_have_size_limit(self):
        from picarones.adapters.corpus.iiif import _fetch_manifest

        # Manifeste valide mais artificiellement gonflé à ~12 Mo
        # par un padding du label.
        big_label = "x" * (12 * 1024 * 1024)
        big_manifest = {
            "@context": "http://iiif.io/api/presentation/2/context.json",
            "@type": "sc:Manifest",
            "label": big_label,
            "sequences": [],
        }
        big_bytes = json.dumps(big_manifest).encode("utf-8")

        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = big_bytes
            mock_resp.__enter__ = lambda self: self
            mock_resp.__exit__ = lambda self, *a: None
            mock_urlopen.return_value = mock_resp

            # Si une limite existe, on s'attend à une exception (ValueError
            # ou OSError ou MemoryError selon implémentation). Sinon le
            # manifeste est chargé entièrement — révélateur de l'absence
            # de garde.
            try:
                manifest = _fetch_manifest("https://example.org/big.json")
                # Pas de garde-fou : on charge tout. C'est la vérité du
                # code actuel — on signale via xfail.
                assert isinstance(manifest, dict)
                pytest.xfail(
                    "S5 — IIIFImporter._fetch_manifest accepte sans broncher "
                    "un manifeste de >10 Mo : pas de borne de taille. "
                    "À durcir : ajouter une lecture par chunks avec MAX_MANIFEST_SIZE."
                )
            except (ValueError, MemoryError, OSError):
                # Une garde existe — comportement souhaité.
                pass


# --------------------------------------------------------------------------
# 5. Manifeste avec contenu malformé (clés bizarres)
# --------------------------------------------------------------------------


class TestIIIFMalformedFields:
    """Un canvas avec des champs ``label``/``image_url`` de types
    inattendus doit être absorbé par le parseur sans crash."""

    def test_canvas_with_int_label_does_not_crash(self):
        from picarones.adapters.corpus.iiif import IIIFManifestParser

        manifest = {
            "@context": "http://iiif.io/api/presentation/2/context.json",
            "@type": "sc:Manifest",
            "sequences": [{
                "canvases": [
                    {"label": 12345, "images": []},
                ],
            }],
        }
        parser = IIIFManifestParser(manifest)
        canvases = parser.canvases()
        # Un canvas, pas de plantage
        assert len(canvases) == 1
        assert isinstance(canvases[0].label, str)
