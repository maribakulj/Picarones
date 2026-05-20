"""Sprint S1.6 — Tests d'attaque SSRF (Server-Side Request Forgery).

``picarones.adapters.corpus._http.validate_http_url`` est censé
empêcher un import IIIF/Gallica/HuggingFace malicieux de faire
fetcher des ressources internes.

Vecteurs couverts
-----------------
1. **Schémas non-HTTP** (``file://``, ``ftp://``, ``data:``) —
   défense déjà annoncée.
2. **Localhost / loopback** (``http://127.0.0.1``, ``http://localhost``,
   ``http://[::1]``).
3. **Métadonnées cloud** (``http://169.254.169.254`` — AWS,
   ``http://metadata.google.internal`` — GCP).
4. **Réseaux privés RFC 1918** (``10.0.0.0/8``, ``172.16.0.0/12``,
   ``192.168.0.0/16``).
5. **Lien local** (``169.254.0.0/16``).

NB : la défense statique ne suffit pas contre les attaques DNS
rebinding ou les redirections HTTP qui pointent ensuite sur du
loopback.  Mais l'absence de défense statique est un signal clair.
"""

from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────
# 1. Schémas non-HTTP (déjà censé être bloqué)
# ──────────────────────────────────────────────────────────────────────


class TestNonHttpSchemes:
    """``file://``, ``ftp://``, ``data:`` doivent lever
    ``ValueError``."""

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://internal.corp/secrets.txt",
            "data:text/html,<script>alert(1)</script>",
            "javascript:alert(1)",
            "gopher://internal:11211/_set%20foo%20bar",
        ],
    )
    def test_non_http_scheme_rejected(self, url: str) -> None:
        from picarones.adapters.corpus._http import validate_http_url

        with pytest.raises(ValueError):
            validate_http_url(url)


# ──────────────────────────────────────────────────────────────────────
# 2. Loopback / localhost
# ──────────────────────────────────────────────────────────────────────


class TestLoopbackBlocked:
    """Un import IIIF qui pointe ``http://127.0.0.1:6379`` peut
    parler au Redis interne."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/manifest.json",
            "http://127.0.0.1:8080/admin",
            "http://localhost/",
            "http://localhost:5432/",
            "http://[::1]/",
            "http://0.0.0.0/",
        ],
    )
    def test_loopback_rejected(self, url: str) -> None:
        from picarones.adapters.corpus._http import validate_http_url

        with pytest.raises(ValueError):
            validate_http_url(url)


# ──────────────────────────────────────────────────────────────────────
# 3. Métadonnées cloud (AWS / GCP)
# ──────────────────────────────────────────────────────────────────────


class TestCloudMetadataBlocked:
    """169.254.169.254 expose les credentials AWS IAM, GCP project
    metadata, etc.  Doit être refusé."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
            "http://metadata.google.internal/",
        ],
    )
    def test_cloud_metadata_rejected(self, url: str) -> None:
        from picarones.adapters.corpus._http import validate_http_url

        with pytest.raises(ValueError):
            validate_http_url(url)


# ──────────────────────────────────────────────────────────────────────
# 4. Réseaux privés RFC 1918
# ──────────────────────────────────────────────────────────────────────


class TestPrivateNetworksBlocked:
    """Dans une institution, l'app peut être derrière un reverse-proxy
    avec accès au réseau interne.  Bloquer les IP privées par défaut."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://10.0.0.1/",
            "http://10.255.255.254/admin",
            "http://172.16.0.1/",
            "http://172.31.255.254/",
            "http://192.168.1.1/",
            "http://192.168.255.254/",
        ],
    )
    def test_private_ipv4_rejected(self, url: str) -> None:
        from picarones.adapters.corpus._http import validate_http_url

        with pytest.raises(ValueError):
            validate_http_url(url)


# ──────────────────────────────────────────────────────────────────────
# 5. Lien-local IPv4 (hors AWS metadata, autres usages)
# ──────────────────────────────────────────────────────────────────────


class TestLinkLocalBlocked:
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.1.1/",
            "http://169.254.255.254/",
        ],
    )
    def test_link_local_rejected(self, url: str) -> None:
        from picarones.adapters.corpus._http import validate_http_url

        with pytest.raises(ValueError):
            validate_http_url(url)


# ──────────────────────────────────────────────────────────────────────
# 6. Sanity : URLs publiques légitimes acceptées
# ──────────────────────────────────────────────────────────────────────


class TestPublicURLsAccepted:
    @pytest.mark.parametrize(
        "url",
        [
            "https://gallica.bnf.fr/iiif/ark:/12148/btv1b8451639t/manifest.json",
            "https://huggingface.co/datasets/test/resolve/main/file.json",
            "https://github.com/PRImA-Research-Lab/PAGE-XML/raw/master/sample.xml",
            "http://images.example.org/iiif/abc/manifest.json",
        ],
    )
    def test_public_url_accepted(self, url: str) -> None:
        from picarones.adapters.corpus._http import validate_http_url

        # Ne lève pas — URL valide.
        validate_http_url(url)
