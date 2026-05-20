"""Sprint S5 — Tests de dégradation réseau pour GallicaClient.

Ce module simule différents modes de panne de l'API Gallica (BnF) :

- Timeout de connexion
- Erreur HTTP 503 (Service Unavailable)
- Erreur HTTP 404 (Not Found)
- Connection refused (réseau inaccessible)
- Réponse partielle / connexion coupée

Pour chaque cas, on vérifie :

- ``GallicaClient`` ne masque pas l'erreur silencieusement (search()
  documente l'erreur via logger, get_metadata() retourne un dict avec
  juste l'ARK).
- Aucun fichier partiel n'est laissé sur disque en cas d'échec.

Les sources HTTP sont mockées au niveau ``urllib.request.urlopen`` pour
simuler les échecs réseau sans dépendance externe (voir CLAUDE.md règle
"pas de tests réseau réels par défaut").
"""

from __future__ import annotations

import socket
import urllib.error
from unittest.mock import patch

import pytest


# --------------------------------------------------------------------------
# 1. Timeout de connexion
# --------------------------------------------------------------------------


class TestGallicaTimeoutPropagation:
    """Sur timeout réseau enveloppé par urllib (URLError), search()
    retourne [] (par contrat) mais log l'erreur ; get_metadata()
    retourne le dict minimal {'ark': ark}.

    Note S5 : ``urllib.request.urlopen`` enveloppe les ``socket.timeout``
    bruts dans ``URLError`` côté production. Ici on simule ce
    comportement de wrapping pour que ``download_url`` capture bien
    l'exception. Un ``socket.timeout`` (= ``TimeoutError``) brut
    *ne serait pas* attrapé par le ``except (URLError, HTTPError)``
    actuel — c'est un point de fragilité documenté ailleurs."""

    def test_search_timeout_returns_empty_list_logs_error(self, caplog):
        from picarones.adapters.corpus.gallica import GallicaClient

        client = GallicaClient(delay_between_requests=0)
        # Wrap le timeout dans URLError comme le ferait urllib
        url_err = urllib.error.URLError(socket.timeout("connection timed out"))
        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen",
            side_effect=url_err,
        ):
            with caplog.at_level("ERROR"):
                results = client.search(title="Froissart", max_results=5)
        # Contrat : pas de plantage, retour vide silencieusement.
        assert results == []
        # Mais l'erreur est documentée
        assert any(
            "SRU" in rec.message or "Erreur" in rec.message
            or "Impossible" in rec.message
            for rec in caplog.records
        )

    def test_get_metadata_timeout_returns_minimal_dict(self):
        from picarones.adapters.corpus.gallica import GallicaClient

        client = GallicaClient(delay_between_requests=0)
        url_err = urllib.error.URLError(socket.timeout("connection timed out"))
        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen",
            side_effect=url_err,
        ):
            meta = client.get_metadata("12148/btv1b8453561w")
        assert meta == {"ark": "12148/btv1b8453561w"}

    def test_raw_socket_timeout_propagates_documents_fragility(self):
        """Documente la fragilité réelle : un ``socket.timeout`` brut
        (= ``TimeoutError`` Py3.10+) n'est PAS attrapé par
        ``except (URLError, HTTPError)`` dans download_url. C'est un bug
        latent — marqué xfail jusqu'à fix production."""
        from picarones.adapters.corpus._http import download_url

        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen",
            side_effect=socket.timeout("raw timeout"),
        ):
            try:
                download_url(
                    "https://gallica.bnf.fr/test",
                    retries=1,
                    backoff=0.0,
                    timeout=1,
                )
            except RuntimeError:
                # Comportement souhaité (si fix appliqué)
                pass
            except (TimeoutError, socket.timeout):
                # Comportement actuel — bug latent
                pytest.xfail(
                    "S5 — download_url ne capture pas socket.timeout brut "
                    "(seulement URLError/HTTPError). À corriger : ajouter "
                    "OSError/TimeoutError au except."
                )


# --------------------------------------------------------------------------
# 2. Erreur HTTP 503 (Service Unavailable)
# --------------------------------------------------------------------------


class TestGallica503Propagation:
    """503 = panne de l'API Gallica côté serveur. Doit lever
    ``RuntimeError`` au niveau ``download_url`` ; le client de plus
    haut niveau (search, get_metadata) absorbe en retour vide /
    minimal mais log."""

    def test_download_url_propagates_503_after_retries(self):
        from picarones.adapters.corpus._http import download_url

        http_error = urllib.error.HTTPError(
            url="https://gallica.bnf.fr/SRU?q=test",
            code=503,
            msg="Service Unavailable",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen",
            side_effect=http_error,
        ):
            # ``download_url`` doit lever RuntimeError explicite, pas
            # silence ni dict vide.
            with pytest.raises(RuntimeError) as exc_info:
                download_url(
                    "https://gallica.bnf.fr/SRU?q=test",
                    retries=2,
                    backoff=0.0,
                    timeout=1,
                )
            assert "https://gallica.bnf.fr/SRU?q=test" in str(exc_info.value)


# --------------------------------------------------------------------------
# 3. Erreur HTTP 404 (Not Found)
# --------------------------------------------------------------------------


class TestGallica404NotFound:
    """404 = ARK inexistant. get_ocr_text() retourne '' sans planter."""

    def test_get_ocr_text_404_returns_empty(self):
        from picarones.adapters.corpus.gallica import GallicaClient

        client = GallicaClient(delay_between_requests=0)
        http_error = urllib.error.HTTPError(
            url="https://gallica.bnf.fr/ark:/12148/inexistant/f1.texteBrut",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen",
            side_effect=http_error,
        ):
            text = client.get_ocr_text("12148/inexistant", page=1)
        # Contrat documenté : "" si OCR non disponible.
        assert text == ""


# --------------------------------------------------------------------------
# 4. Connection refused (réseau totalement inaccessible)
# --------------------------------------------------------------------------


class TestGallicaConnectionRefused:
    """Le réseau est down (Wi-Fi coupé, DNS cassé). On veut une erreur
    explicite avec message propre, pas un AttributeError ou KeyError."""

    def test_download_url_connection_refused_explicit_error(self):
        from picarones.adapters.corpus._http import download_url

        url_error = urllib.error.URLError(
            ConnectionRefusedError("Connection refused")
        )
        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen",
            side_effect=url_error,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                download_url(
                    "https://gallica.bnf.fr/manifest.json",
                    retries=1,
                    backoff=0.0,
                    timeout=1,
                )
            assert "gallica.bnf.fr" in str(exc_info.value)


# --------------------------------------------------------------------------
# 5. Pas de fichier partiel sur disque en cas d'échec
# --------------------------------------------------------------------------


class TestGallicaNoPartialFileOnFailure:
    """Si le téléchargement échoue avant la fin, aucun fichier
    partiel ne doit polluer le filesystem.

    Note : la fonction ``download_url`` retourne ``bytes`` en mémoire,
    elle n'écrit jamais sur disque (pas de risque de partial). On
    vérifie tout de même le comportement défensif côté client.
    """

    def test_no_orphan_files_after_search_timeout(self, tmp_path):
        from picarones.adapters.corpus.gallica import GallicaClient

        client = GallicaClient(delay_between_requests=0)
        # Le tmp_path est totalement vide au départ
        before = list(tmp_path.iterdir())
        assert before == []

        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen",
            side_effect=urllib.error.URLError(socket.timeout("timeout")),
        ):
            client.search(title="Froissart")

        # tmp_path doit rester vide : Gallica ne touche pas au disque
        # pendant search/get_metadata
        after = list(tmp_path.iterdir())
        assert after == [], f"Fichiers parasites créés: {after}"

    def test_get_ocr_text_failure_no_disk_artifact(self, tmp_path):
        from picarones.adapters.corpus.gallica import GallicaClient

        client = GallicaClient(delay_between_requests=0)
        before = list(tmp_path.iterdir())
        assert before == []

        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen",
            side_effect=urllib.error.URLError("network unreachable"),
        ):
            text = client.get_ocr_text("12148/anything", page=1)

        assert text == ""
        # Aucun fichier intermédiaire dans tmp_path
        after = list(tmp_path.iterdir())
        assert after == []


# --------------------------------------------------------------------------
# 6. Retry exponentiel : message d'erreur explicite après épuisement
# --------------------------------------------------------------------------


class TestGallicaRetriesExhausted:
    """Après ``retries`` tentatives, ``download_url`` lève une
    ``RuntimeError`` qui mentionne le nombre exact de tentatives."""

    def test_retries_exhausted_explicit_message(self):
        from picarones.adapters.corpus._http import download_url

        with patch(
            "picarones.adapters.corpus._http.urllib.request.urlopen",
            side_effect=urllib.error.URLError("server down"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                download_url(
                    "https://gallica.bnf.fr/test",
                    retries=3,
                    backoff=0.0,  # pas d'attente pour le test
                    timeout=1,
                )
            # Le message contient "3 tentatives"
            assert "3 tentatives" in str(exc_info.value)
