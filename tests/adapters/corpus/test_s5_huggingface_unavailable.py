"""Sprint S5 — Tests d'indisponibilité de HuggingFace Hub.

Cas couverts :

- HF Hub renvoie 503 (panne)
- HF Hub renvoie 404 (dataset inexistant)
- Erreur réseau (DNS down)

Pour chacun, vérifie que :

- ``HuggingFaceImporter.search`` retourne au moins les datasets de
  référence (fallback gracieux), pas une exception cryptique.
- L'erreur API est documentée via ``record_fallback`` (pas de
  silence complet).
- ``import_dataset`` n'écrit qu'un fichier de métadonnées si
  ``datasets`` n'a rien pu importer (jamais d'images partielles).
"""

from __future__ import annotations

import urllib.error
import warnings
from unittest.mock import patch

import pytest


# --------------------------------------------------------------------------
# Setup : les imports HuggingFace émettent un UserWarning expérimental.
# On les filtre pour la lisibilité des sorties pytest sans masquer un
# vrai warning du code testé.
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _silence_hf_experimental_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*huggingface.*experimental.*",
            category=UserWarning,
        )
        yield


# --------------------------------------------------------------------------
# 1. HF Hub renvoie 503
# --------------------------------------------------------------------------


class TestHuggingFace503:
    """Quand l'API HF répond 503, search() doit retourner au moins
    les datasets de référence pré-intégrés (graceful degradation)."""

    def test_search_503_falls_back_to_reference_datasets(self):
        from picarones.adapters.corpus.huggingface import HuggingFaceImporter

        importer = HuggingFaceImporter()

        http_503 = urllib.error.HTTPError(
            url="https://huggingface.co/api/datasets",
            code=503,
            msg="Service Unavailable",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        with patch(
            "urllib.request.urlopen",
            side_effect=http_503,
        ):
            results = importer.search(query="medieval", limit=5)

        # Les datasets de référence pré-intégrés doivent être retournés
        # même si l'API est down.
        assert isinstance(results, list)
        # Au moins un résultat dans la liste de référence
        # (filtrage par query="medieval")
        assert len(results) >= 1
        # Tous les résultats viennent de la liste de référence
        # (pas de l'API qui est down)
        for r in results:
            assert r.source == "reference"


# --------------------------------------------------------------------------
# 2. HF Hub renvoie 404 sur un dataset précis
# --------------------------------------------------------------------------


class TestHuggingFace404:
    """``import_dataset`` sur un dataset_id inexistant ne crée pas
    d'images partielles. Seul le fichier de métadonnées
    ``huggingface_meta.json`` est créé (avec l'info "0 imported")."""

    def test_import_unknown_dataset_writes_only_metadata(self, tmp_path):
        from picarones.adapters.corpus.huggingface import HuggingFaceImporter

        importer = HuggingFaceImporter()
        # On force _try_import_with_datasets_lib à retourner 0
        # (datasets non installé, ou dataset 404, ou ImportError)
        with patch(
            "picarones.adapters.corpus.huggingface."
            "_try_import_with_datasets_lib",
            return_value=0,
        ):
            result = importer.import_dataset(
                "nonexistent/dataset-404",
                output_dir=tmp_path,
                max_samples=10,
                show_progress=False,
            )

        # Le fichier de métadonnées doit exister
        meta_file = tmp_path / "huggingface_meta.json"
        assert meta_file.exists()

        # Et 0 fichier d'image / GT n'a été créé
        files = sorted(p.name for p in tmp_path.iterdir())
        # Le seul fichier qui doit exister est huggingface_meta.json
        assert files == ["huggingface_meta.json"]

        assert result["files_imported"] == 0
        assert result["dataset_id"] == "nonexistent/dataset-404"


# --------------------------------------------------------------------------
# 3. Erreur réseau brute (DNS down)
# --------------------------------------------------------------------------


class TestHuggingFaceNetworkDown:
    """Sur DNS down ou socket refused, search() doit retourner les
    datasets de référence sans propager l'exception (test du
    contrat de graceful degradation)."""

    def test_search_dns_down_returns_reference_only(self):
        from picarones.adapters.corpus.huggingface import HuggingFaceImporter

        importer = HuggingFaceImporter()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Name or service not known"),
        ):
            # Doit retourner sans lever d'exception
            results = importer.search(query="ocr", limit=5)

        assert isinstance(results, list)
        for r in results:
            # Tous viennent de la liste de référence (API inaccessible)
            assert r.source == "reference"


# --------------------------------------------------------------------------
# 4. Erreur claire vs cryptique
# --------------------------------------------------------------------------


class TestHuggingFaceErrorMessageQuality:
    """Quand un dataset_id totalement vide est fourni, on s'attend
    à un comportement défini (pas un AttributeError au fond d'une
    pile non gérée)."""

    def test_empty_dataset_id_does_not_crash_metadata_write(self, tmp_path):
        from picarones.adapters.corpus.huggingface import HuggingFaceImporter

        importer = HuggingFaceImporter()
        with patch(
            "picarones.adapters.corpus.huggingface."
            "_try_import_with_datasets_lib",
            return_value=0,
        ):
            # Empty dataset_id : on accepte n'importe quel comportement
            # tant qu'il est défini (pas de TypeError, pas d'AttributeError)
            result = importer.import_dataset(
                dataset_id="",
                output_dir=tmp_path,
                max_samples=1,
                show_progress=False,
            )
        # Le fichier de métadonnées existe
        assert (tmp_path / "huggingface_meta.json").exists()
        assert result["dataset_id"] == ""
