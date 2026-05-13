"""Phase 3.3 audit code-quality — ``NormalizationProfile.from_yaml``
exposé en CLI et via ``POST /api/normalization/profiles/preview``.

Avant la Phase 3.3 :

- ``NormalizationProfile.from_yaml`` était écrit (formats/text/normalization.py:191)
  mais aucun caller ne l'utilisait : 0 hit dans ``grep -rn from_yaml``,
  0 test associé.
- L'API web exposait seulement les 11 profils builtin via
  ``GET /api/normalization/profiles``.
- La CLI ``picarones run`` n'avait aucune option de normalisation.

Phase 3.3 :

- Option CLI ``--normalization-profile <ID-OR-PATH>`` (identifiant
  builtin ou chemin .yaml versionné).
- Helper ``picarones.interfaces.cli._normalization_arg.resolve_normalization_profile``
  unifiant les deux chemins.
- Endpoint ``POST /api/normalization/profiles/preview`` (validation
  + sérialisation, pas de persistance).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


_VALID_YAML = textwrap.dedent(
    """\
    name: medieval_custom
    description: Français médiéval BnF (test)
    caseless: false
    nfc: true
    exclude_chars: ".,;:!?"
    diplomatic:
      ſ: s
      u: v
      v: u
    """
)


# --------------------------------------------------------------------------
# 1. NormalizationProfile.from_yaml — round-trip
# --------------------------------------------------------------------------


class TestFromYAMLDirect:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        from picarones.formats.text.normalization import NormalizationProfile

        path = tmp_path / "medieval.yaml"
        path.write_text(_VALID_YAML, encoding="utf-8")

        profile = NormalizationProfile.from_yaml(path)
        assert profile.name == "medieval_custom"
        assert profile.description == "Français médiéval BnF (test)"
        assert profile.caseless is False
        assert profile.nfc is True
        assert profile.diplomatic_table["ſ"] == "s"
        assert "." in profile.exclude_chars
        assert "!" in profile.exclude_chars

    def test_missing_name_uses_filename_stem(self, tmp_path: Path) -> None:
        """Si ``name`` n'est pas dans le YAML, le stem du fichier
        fait office de défaut (cf. docstring de la fonction)."""
        from picarones.formats.text.normalization import NormalizationProfile

        path = tmp_path / "my_corpus.yml"
        path.write_text("caseless: true\n", encoding="utf-8")

        profile = NormalizationProfile.from_yaml(path)
        assert profile.name == "my_corpus"
        assert profile.caseless is True


# --------------------------------------------------------------------------
# 2. CLI helper resolve_normalization_profile
# --------------------------------------------------------------------------


class TestResolveCLIArg:
    def test_none_returns_none(self) -> None:
        from picarones.interfaces.cli._normalization_arg import (
            resolve_normalization_profile,
        )
        assert resolve_normalization_profile(None) is None
        assert resolve_normalization_profile("") is None

    def test_builtin_id_resolves(self) -> None:
        from picarones.evaluation.metrics.normalization import (
            NORMALIZATION_PROFILES,
        )
        from picarones.interfaces.cli._normalization_arg import (
            resolve_normalization_profile,
        )

        profile = resolve_normalization_profile("nfc")
        assert profile is NORMALIZATION_PROFILES["nfc"]

    def test_yaml_path_resolves(self, tmp_path: Path) -> None:
        from picarones.interfaces.cli._normalization_arg import (
            resolve_normalization_profile,
        )

        path = tmp_path / "custom.yaml"
        path.write_text(_VALID_YAML, encoding="utf-8")

        profile = resolve_normalization_profile(str(path))
        assert profile is not None
        assert profile.name == "medieval_custom"

    def test_yaml_path_missing_raises(self, tmp_path: Path) -> None:
        from picarones.interfaces.cli._normalization_arg import (
            resolve_normalization_profile,
        )

        with pytest.raises(FileNotFoundError, match="introuvable"):
            resolve_normalization_profile(str(tmp_path / "absent.yaml"))

    def test_unknown_id_raises_with_help(self) -> None:
        from picarones.interfaces.cli._normalization_arg import (
            resolve_normalization_profile,
        )

        with pytest.raises(ValueError, match="inconnu") as exc_info:
            resolve_normalization_profile("not_a_real_profile")
        # Le message doit citer les identifiants disponibles pour
        # aider l'utilisateur à se corriger sans aller lire la doc.
        assert "nfc" in str(exc_info.value)


# --------------------------------------------------------------------------
# 3. Option CLI --normalization-profile branchée à ``picarones run``
# --------------------------------------------------------------------------


class TestCLIIntegration:
    def test_run_cmd_accepts_normalization_profile_option(self) -> None:
        """L'option ``--normalization-profile`` doit être déclarée
        sur la commande ``run`` (Click)."""
        from click.testing import CliRunner

        from picarones.interfaces.cli._workflows import run_cmd

        runner = CliRunner()
        result = runner.invoke(run_cmd, ["--help"])
        assert result.exit_code == 0, result.output
        assert "--normalization-profile" in result.output
        assert "ID-OR-PATH" in result.output

    def test_run_cmd_rejects_invalid_profile_with_clean_error(
        self, tmp_path: Path,
    ) -> None:
        """Un identifiant inconnu doit produire un exit code != 0 et
        un message d'erreur lisible (pas un stacktrace Python).
        Vérifie que la résolution est bien faite **avant** le
        chargement du corpus (rejet précoce)."""
        from click.testing import CliRunner

        from picarones.interfaces.cli._workflows import run_cmd

        runner = CliRunner()
        result = runner.invoke(
            run_cmd,
            [
                "--corpus", str(tmp_path),  # corpus vide — peu importe
                "--engines", "tesseract",
                "--output", str(tmp_path / "out.json"),
                "--normalization-profile", "not_a_real_profile",
            ],
        )
        assert result.exit_code != 0
        assert "profil normalisation" in result.output.lower() or "normalization" in result.output.lower()


# --------------------------------------------------------------------------
# 4. Endpoint POST /api/normalization/profiles/preview
# --------------------------------------------------------------------------


@pytest.fixture
def web_client():
    """Client FastAPI minimal pour tester l'endpoint preview."""
    from fastapi.testclient import TestClient

    from picarones.interfaces.web.routers.normalization import router

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestPreviewEndpoint:
    def test_valid_yaml_returns_serialized_profile(self, web_client) -> None:
        resp = web_client.post(
            "/api/normalization/profiles/preview",
            json={"yaml": _VALID_YAML},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["name"] == "medieval_custom"
        assert data["caseless"] is False
        assert data["nfc"] is True
        assert data["diplomatic_rules"] == 3
        assert data["diplomatic_table"]["ſ"] == "s"
        assert "." in data["exclude_chars"]

    def test_invalid_yaml_returns_400(self, web_client) -> None:
        # YAML syntaxiquement invalide.
        resp = web_client.post(
            "/api/normalization/profiles/preview",
            json={"yaml": "name: [unclosed_list"},
        )
        assert resp.status_code == 400
        assert "invalide" in resp.json()["detail"].lower()

    def test_yaml_too_large_rejected_by_pydantic(self, web_client) -> None:
        """Le ``max_length`` Pydantic doit refuser un YAML > 64 KiB
        au niveau de la validation request, avant tout parsing."""
        oversized = "x: y\n" * 20000  # ~100 KiB
        resp = web_client.post(
            "/api/normalization/profiles/preview",
            json={"yaml": oversized},
        )
        # Pydantic renvoie 422 sur max_length, ou 400 si notre check
        # interne se déclenche d'abord — les deux sont OK.
        assert resp.status_code in (400, 422)

    def test_preview_does_not_register_profile(self, web_client) -> None:
        """Le profil prévisualisé ne doit PAS apparaître dans la
        liste ``GET /api/normalization/profiles`` — c'est un preview,
        pas une persistance."""
        # On envoie un profil avec un nom unique.
        yaml = textwrap.dedent("""\
            name: zzz_unique_test_profile_xyz
            caseless: true
        """)
        resp = web_client.post(
            "/api/normalization/profiles/preview",
            json={"yaml": yaml},
        )
        assert resp.status_code == 200

        # GET doit toujours retourner uniquement les builtins.
        list_resp = web_client.get("/api/normalization/profiles")
        assert list_resp.status_code == 200
        ids = {p["id"] for p in list_resp.json()["profiles"]}
        assert "zzz_unique_test_profile_xyz" not in ids
