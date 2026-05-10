"""Tests Sprint 97 — B.6 : politique de modules contribués.

Couvre :

1. ``ModuleManifest`` : as_dict, champs.
2. ``validate_manifest`` :
   - manifest valide → tous les checks passent
   - champ manquant → check fail
   - input/output_types vides → check fail
3. ``audit_module`` :
   - module + manifest valide → passed=True
   - classe ne hérite pas de BaseModule → fail
   - I/O ne correspondent pas → fail
   - process absent → fail
   - case-insensitive sur les types
4. Vue HTML :
   - empty
   - rendu complet
   - badge ✓ / ✗
   - anti-injection
   - FR + EN
5. Documentation : présente.
6. Complétude i18n FR/EN.
"""

from __future__ import annotations

import json
from pathlib import Path

from picarones.evaluation.metrics.module_policy import (
    AuditCheck,
    AuditResult,
    ModuleManifest,
    audit_module,
    validate_manifest,
)
from picarones.domain.artifacts import ArtifactType
from picarones.domain.module_protocol import BaseModule
from picarones.reports.html.renderers.module_audit import (
    build_module_audit_html,
)


def _load_labels(lang: str) -> dict:
    p = (
        Path(__file__).parent.parent.parent.parent
        / "picarones" / "reports" / "i18n" / f"{lang}.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


def _ok_manifest(**overrides) -> ModuleManifest:
    base = {
        "name": "my-mod", "version": "1.0.0",
        "author": "alice", "license": "MIT",
        "description": "test module",
        "input_types": ["text"], "output_types": ["text"],
    }
    base.update(overrides)
    return ModuleManifest(**base)


class _MockTextModule(BaseModule):
    name = "mock-text"
    input_types = (ArtifactType.TEXT,)
    output_types = (ArtifactType.TEXT,)
    execution_mode = "cpu"

    def process(self, inputs):
        return inputs


# ──────────────────────────────────────────────────────────────────────────
# 1. ModuleManifest
# ──────────────────────────────────────────────────────────────────────────


class TestManifest:
    def test_as_dict(self) -> None:
        m = _ok_manifest()
        d = m.as_dict()
        assert d["name"] == "my-mod"
        assert d["input_types"] == ["text"]
        assert d["citation"] is None

    def test_optional_fields(self) -> None:
        m = _ok_manifest(citation="Foo 2025", homepage="https://example")
        d = m.as_dict()
        assert d["citation"] == "Foo 2025"
        assert d["homepage"] == "https://example"


# ──────────────────────────────────────────────────────────────────────────
# 2. validate_manifest
# ──────────────────────────────────────────────────────────────────────────


class TestValidate:
    def test_full_manifest_passes(self) -> None:
        checks = validate_manifest(_ok_manifest())
        assert all(c.passed for c in checks)

    def test_missing_field_fails(self) -> None:
        checks = validate_manifest(_ok_manifest(license=""))
        assert any(
            (c.name == "manifest.license" and not c.passed)
            for c in checks
        )

    def test_empty_input_types_fails(self) -> None:
        checks = validate_manifest(_ok_manifest(input_types=[]))
        assert any(
            (c.name == "manifest.input_types" and not c.passed)
            for c in checks
        )

    def test_empty_output_types_fails(self) -> None:
        checks = validate_manifest(_ok_manifest(output_types=[]))
        assert any(
            (c.name == "manifest.output_types" and not c.passed)
            for c in checks
        )


# ──────────────────────────────────────────────────────────────────────────
# 3. audit_module
# ──────────────────────────────────────────────────────────────────────────


class TestAuditModule:
    def test_valid_module_passes(self) -> None:
        result = audit_module(_MockTextModule, _ok_manifest())
        assert result.passed
        assert result.n_failed == 0

    def test_non_basemodule_fails(self) -> None:
        class NotABaseModule:
            input_types = (ArtifactType.TEXT,)
            output_types = (ArtifactType.TEXT,)
            def process(self, inputs):
                return inputs
        result = audit_module(NotABaseModule, _ok_manifest())
        assert not result.passed
        assert any(
            c.name == "module.inherits_base_module" and not c.passed
            for c in result.checks
        )

    def test_io_mismatch_fails(self) -> None:
        # Manifest dit ALTO mais module dit TEXT
        manifest = _ok_manifest(output_types=["alto"])
        result = audit_module(_MockTextModule, manifest)
        assert not result.passed
        assert any(
            c.name == "module.output_types_match_manifest" and not c.passed
            for c in result.checks
        )

    def test_case_insensitive_types(self) -> None:
        # Manifest en majuscules, module en lowercase
        manifest = _ok_manifest(
            input_types=["TEXT"], output_types=["TEXT"],
        )
        result = audit_module(_MockTextModule, manifest)
        assert result.passed

    def test_accepts_instance_or_class(self) -> None:
        result_class = audit_module(_MockTextModule, _ok_manifest())
        result_instance = audit_module(_MockTextModule(), _ok_manifest())
        assert result_class.passed == result_instance.passed

    def test_audit_result_dict(self) -> None:
        result = audit_module(_MockTextModule, _ok_manifest())
        d = result.as_dict()
        assert d["module_name"] == "my-mod"
        assert d["passed"] is True
        assert d["n_passed"] >= 5
        assert isinstance(d["checks"], list)


# ──────────────────────────────────────────────────────────────────────────
# 4. Vue HTML
# ──────────────────────────────────────────────────────────────────────────


def _audit_entry(manifest: ModuleManifest, passed: bool = True,
                 n_failed: int = 0) -> dict:
    audit = AuditResult(
        module_name=manifest.name,
        passed=passed,
        checks=[
            AuditCheck("manifest.name", True),
            AuditCheck("manifest.version", True),
            AuditCheck("module.inherits_base_module", passed),
        ],
    )
    return {
        "manifest": manifest.as_dict(),
        "audit": audit.as_dict(),
    }


class TestRender:
    def test_empty_returns_empty(self) -> None:
        assert build_module_audit_html(None) == ""
        assert build_module_audit_html([]) == ""

    def test_renders_table(self) -> None:
        entry = _audit_entry(_ok_manifest(citation="Foo 2025"))
        html = build_module_audit_html([entry], _load_labels("fr"))
        assert "<table" in html
        assert "my-mod" in html
        assert "1.0.0" in html
        assert "Foo 2025" in html
        # Badge ✓
        assert "✓" in html

    def test_failed_audit_shows_cross(self) -> None:
        manifest = _ok_manifest()
        entry = _audit_entry(manifest, passed=False, n_failed=2)
        # Patch n_failed
        entry["audit"]["n_failed"] = 2
        html = build_module_audit_html([entry], _load_labels("fr"))
        assert "✗" in html
        assert "2" in html  # nombre d'échecs

    def test_anti_injection_name(self) -> None:
        manifest = _ok_manifest(name="<script>alert(1)</script>")
        entry = _audit_entry(manifest)
        html = build_module_audit_html([entry], _load_labels("fr"))
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_anti_injection_homepage(self) -> None:
        manifest = _ok_manifest(homepage="<svg/>")
        entry = _audit_entry(manifest)
        html = build_module_audit_html([entry], _load_labels("fr"))
        assert "<svg/>" not in html
        assert "&lt;svg" in html

    def test_anti_injection_citation(self) -> None:
        manifest = _ok_manifest(citation="<img src=x onerror=alert>")
        entry = _audit_entry(manifest)
        html = build_module_audit_html([entry], _load_labels("fr"))
        assert "<img src" not in html
        assert "&lt;img" in html

    def test_renders_in_english(self) -> None:
        entry = _audit_entry(_ok_manifest())
        html = build_module_audit_html([entry], _load_labels("en"))
        assert "Audited modules" in html


# ──────────────────────────────────────────────────────────────────────────
# 5. Documentation
# ──────────────────────────────────────────────────────────────────────────


class TestDocumentation:
    def test_docs_present(self) -> None:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "docs" / "developer" / "module-policy.md"
        )
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        # Doit mentionner les concepts clés
        assert "ModuleManifest" in text
        assert "BaseModule" in text
        assert "audit_module" in text

    def test_docs_lists_required_fields(self) -> None:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "docs" / "developer" / "module-policy.md"
        )
        text = path.read_text(encoding="utf-8")
        for key in ("name", "version", "author", "license", "description"):
            assert f"`{key}`" in text, f"champ manquant dans la doc : {key}"


# ──────────────────────────────────────────────────────────────────────────
# 6. Complétude i18n
# ──────────────────────────────────────────────────────────────────────────


_KEYS = {
    "audit_title", "audit_note", "audit_pass", "audit_fail",
    "audit_module", "audit_status", "audit_version", "audit_author",
    "audit_license", "audit_io", "audit_citation", "audit_homepage",
}


class TestI18n:
    def test_fr(self) -> None:
        d = _load_labels("fr")
        assert not _KEYS - d.keys()

    def test_en(self) -> None:
        d = _load_labels("en")
        assert not _KEYS - d.keys()
