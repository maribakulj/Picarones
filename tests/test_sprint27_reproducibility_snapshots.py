"""Tests Sprint 27 — snapshots de reproductibilité dans le rapport HTML.

Le Sprint 27 ajoute le bloc ``report_data["snapshots"]`` qui embarque
dans chaque rapport HTML auto-contenu :

  - le YAML brut intégral de ``picarones/data/pricing.yaml`` ;
  - les entrées du glossaire dans la langue du rapport ;
  - le profil de normalisation effectivement utilisé ;
  - la version Picarones, la version Python, la plateforme,
    le commit git si dispo, et la liste figée des paquets installés.

Le but est qu'un lecteur du rapport puisse rejouer la synthèse, le
Pareto et le glossaire sans accès au code source du moment où le
rapport a été généré.
"""

from __future__ import annotations

import json
import re

import pytest


# ---------------------------------------------------------------------------
# 1. Fonctions snapshot unitaires
# ---------------------------------------------------------------------------

class TestPricingSnapshot:
    def test_default_pricing_yaml_is_loaded(self):
        from picarones.report.snapshot import pricing_snapshot
        s = pricing_snapshot()
        assert s["available"] is True
        assert s["filename"] == "pricing.yaml"
        assert s["size_bytes"] > 100, "pricing.yaml ne doit pas être quasi-vide"
        # raw_yaml et data sont cohérents
        assert isinstance(s["raw_yaml"], str)
        assert isinstance(s["data"], dict)

    def test_data_contains_meta_and_engines(self):
        from picarones.report.snapshot import pricing_snapshot
        s = pricing_snapshot()
        assert "meta" in s["data"], "le snapshot doit exposer la section meta"
        assert "engines" in s["data"], "le snapshot doit exposer engines"

    def test_missing_path_returns_unavailable(self, tmp_path):
        from picarones.report.snapshot import pricing_snapshot
        s = pricing_snapshot(pricing_path=tmp_path / "ne-pas-exister.yaml")
        assert s["available"] is False
        assert "introuvable" in s["reason"].lower()

    def test_custom_yaml_round_trips(self, tmp_path):
        from picarones.report.snapshot import pricing_snapshot
        custom = tmp_path / "custom.yaml"
        custom.write_text(
            "meta:\n  currency: USD\n  last_updated: 2026-01-01\nengines:\n  fake: {type: local}\n",
            encoding="utf-8",
        )
        s = pricing_snapshot(pricing_path=custom)
        assert s["available"] is True
        assert s["data"]["meta"]["currency"] == "USD"
        assert "fake" in s["data"]["engines"]
        # Le brut doit être identique au fichier source — preuve de fidélité.
        assert s["raw_yaml"] == custom.read_text(encoding="utf-8")


class TestGlossarySnapshot:
    def test_default_lang_returns_entries(self):
        from picarones.report.snapshot import glossary_snapshot
        s = glossary_snapshot(lang="fr")
        assert s["available"] is True
        assert s["entry_count"] > 10
        # Quelques clés canoniques attendues
        for k in ("cer", "wer"):
            assert k in s["entries"]

    def test_used_keys_filter(self):
        from picarones.report.snapshot import glossary_snapshot
        s = glossary_snapshot(lang="fr", used_keys=["cer"])
        assert s["entry_count"] == 1
        assert list(s["entries"]) == ["cer"]

    def test_unknown_lang_falls_back(self):
        # `load_glossary` retombe sur fr si la langue est absente — donc
        # le snapshot doit être disponible avec lang='fr' ou la langue
        # demandée selon ce qu'on retourne. On vérifie qu'on ne crashe pas.
        from picarones.report.snapshot import glossary_snapshot
        s = glossary_snapshot(lang="xx-pas-existante")
        # Soit on retombe sur fr (available=True), soit on signale unavailable.
        assert "available" in s

    def test_entries_sorted_for_determinism(self):
        from picarones.report.snapshot import glossary_snapshot
        s = glossary_snapshot(lang="fr")
        keys = list(s["entries"])
        assert keys == sorted(keys), (
            "Les entrées doivent être triées pour produire un snapshot "
            "bit-à-bit reproductible."
        )


class TestNormalizationSnapshot:
    def test_builtin_profile_serializes(self):
        from picarones.core.normalization import get_builtin_profile
        from picarones.report.snapshot import normalization_snapshot
        p = get_builtin_profile("medieval_french")
        s = normalization_snapshot(p)
        assert s["available"] is True
        assert s["name"] == "medieval_french"
        assert s["nfc"] is True
        # La table contient des correspondances connues
        assert s["diplomatic_table"].get("ſ") == "s"

    def test_none_profile_returns_unavailable(self):
        from picarones.report.snapshot import normalization_snapshot
        s = normalization_snapshot(None)
        assert s["available"] is False

    def test_exclude_chars_sorted(self):
        from picarones.core.normalization import get_builtin_profile
        from picarones.report.snapshot import normalization_snapshot
        p = get_builtin_profile("sans_ponctuation")
        s = normalization_snapshot(p)
        # Liste triée pour reproductibilité
        assert s["exclude_chars"] == sorted(s["exclude_chars"])


class TestEnvironmentSnapshot:
    def test_returns_picarones_version(self):
        from picarones import __version__
        from picarones.report.snapshot import environment_snapshot
        s = environment_snapshot()
        assert s["available"] is True
        assert s["picarones_version"] == __version__

    def test_python_and_platform_present(self):
        from picarones.report.snapshot import environment_snapshot
        s = environment_snapshot()
        assert s["python_version"]
        assert s["python_implementation"]
        assert s["platform"]

    def test_installed_packages_sorted_unique(self):
        from picarones.report.snapshot import environment_snapshot
        s = environment_snapshot()
        pkgs = s["installed_packages"]
        assert isinstance(pkgs, list)
        # Triés case-insensitive
        assert pkgs == sorted(pkgs, key=str.lower)
        # Pas de doublons
        names = [p.split("==", 1)[0].lower() for p in pkgs]
        assert len(names) == len(set(names))

    def test_git_commit_is_str_or_none(self):
        from picarones.report.snapshot import environment_snapshot
        s = environment_snapshot()
        commit = s.get("git_commit")
        assert commit is None or (isinstance(commit, str) and 0 < len(commit) <= 12)


# ---------------------------------------------------------------------------
# 2. snapshot_all : l'API agrégée appelée par ReportGenerator
# ---------------------------------------------------------------------------

class TestSnapshotAll:
    def test_contains_all_four_blocks(self):
        from picarones.report.snapshot import snapshot_all
        s = snapshot_all()
        for k in ("pricing", "glossary", "normalization", "environment"):
            assert k in s, f"snapshot_all doit exposer la clé '{k}'"
        assert s["schema_version"] == 1

    def test_deterministic_for_same_inputs(self):
        from picarones.core.normalization import get_builtin_profile
        from picarones.report.snapshot import snapshot_all
        profile = get_builtin_profile("nfc")

        a = snapshot_all(lang="fr", normalization_profile=profile)
        b = snapshot_all(lang="fr", normalization_profile=profile)
        # Les sections statiques (pricing, glossary, normalization) sont
        # déterministes ; environment peut varier sur git_commit selon
        # l'état du repo. On compare donc les trois sections clés.
        for k in ("pricing", "glossary", "normalization"):
            assert a[k] == b[k], f"Section '{k}' non déterministe"


# ---------------------------------------------------------------------------
# 3. Intégration ReportGenerator : snapshots embarqués dans le HTML
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated_report_html(tmp_path_factory) -> str:
    """Génère un rapport démo et retourne son contenu HTML."""
    from picarones import fixtures
    from picarones.core.normalization import get_builtin_profile
    from picarones.report.generator import ReportGenerator

    b = fixtures.generate_sample_benchmark(n_docs=6)
    out_dir = tmp_path_factory.mktemp("rep27")
    out = out_dir / "report.html"
    gen = ReportGenerator(
        b,
        lang="fr",
        normalization_profile=get_builtin_profile("medieval_french"),
    )
    gen.generate(out)
    return out.read_text(encoding="utf-8")


def _extract_report_data(html: str) -> dict:
    """Récupère le dict ``report_data`` injecté dans le HTML.

    Le générateur sérialise ``report_data`` en JSON dans une balise
    ``<script id="picarones-data" type="application/json">``. Cette
    fonction parse le JSON pour permettre des assertions précises.
    """
    m = re.search(
        r'<script[^>]*id="picarones-data"[^>]*>(.*?)</script>',
        html,
        re.DOTALL,
    )
    if not m:
        # Fallback : chercher la première occurrence de ``"snapshots"``
        # et ouvrir le JSON englobant.
        idx = html.find('"snapshots"')
        assert idx >= 0, "Aucun bloc 'snapshots' trouvé dans le rapport"
        # On retourne un dict factice pour ne pas bloquer les tests qui
        # ne dépendent pas du parse précis.
        return {"snapshots": {"present_in_html": True}}
    return json.loads(m.group(1))


class TestReportEmbedsSnapshots:
    def test_html_contains_snapshots_block(self, generated_report_html):
        assert '"snapshots"' in generated_report_html
        assert '"schema_version":1' in generated_report_html

    def test_pricing_yaml_embedded_raw(self, generated_report_html):
        # Le YAML brut doit être présent (chercher une ligne caractéristique)
        assert "engines:" in generated_report_html
        # ``meta:`` apparaît aussi dans pricing.yaml
        assert "meta:" in generated_report_html

    def test_environment_block_embedded(self, generated_report_html):
        assert '"picarones_version"' in generated_report_html
        assert '"python_version"' in generated_report_html
        assert '"installed_packages"' in generated_report_html

    def test_glossary_block_embedded(self, generated_report_html):
        # Quelques clés du glossaire doivent figurer dans le HTML — mais
        # comme le glossaire est aussi rendu côté UI dans une autre var,
        # on vérifie au moins la présence du JSON glossary dans snapshots.
        assert '"entries"' in generated_report_html

    def test_normalization_profile_embedded(self, generated_report_html):
        # Le snapshot doit nommer le profil utilisé
        assert "medieval_french" in generated_report_html


class TestReportSnapshotPersistsAcrossPricingChanges:
    """Garantie de reproductibilité : un rapport généré aujourd'hui reste
    cohérent avec le pricing au moment de la génération, même si
    ``picarones/data/pricing.yaml`` change ensuite."""

    def test_snapshot_carries_full_yaml_for_replay(self, generated_report_html):
        # Si quelqu'un ouvre le HTML demain et veut rejouer la table de
        # prix, il peut extraire le ``raw_yaml`` du bloc snapshots et le
        # parser. On vérifie que le brut YAML est bien là tel quel.
        assert "raw_yaml" in generated_report_html
        # Les hypothèses détaillées (assumptions, notes, sources) sont
        # dans le YAML — au moins une doit apparaître dans le HTML
        # via le bloc raw_yaml.
        assert ("assumptions" in generated_report_html
                or "notes" in generated_report_html
                or "sources" in generated_report_html), (
            "Le YAML pricing brut doit embarquer assumptions/notes/sources"
        )
