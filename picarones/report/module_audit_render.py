"""Rendu HTML « Modules audités » — Sprint 97 (B.6).

Suite directe ``picarones/core/module_policy.py``.  Pattern
identique aux autres rendus : server-side, pas de JS, anti-
injection systématique.

Vue
---
Tableau récapitulatif des modules utilisés dans une pipeline
composée, chacun avec :

- Statut d'audit (✓ vert si tous les checks passent, ✗ rouge
  sinon, avec compte des échecs) ;
- Métadonnées : version, auteur, licence ;
- Citation académique si fournie ;
- Lien vers la homepage si fourni.

Adaptive : ``""`` si la liste est vide.

Note d'intégration
------------------
Module pur — l'utilisateur compose la liste depuis sa
``PipelineSpec`` augmentée des ``ModuleManifest`` :

.. code-block:: python

    from picarones.measurements.module_policy import audit_module
    from picarones.report.module_audit_render import build_module_audit_html

    audits = []
    for step in pipeline.steps:
        manifest = step.module.manifest  # convention applicative
        result = audit_module(step.module, manifest)
        audits.append({
            "manifest": manifest.as_dict(),
            "audit": result.as_dict(),
        })
    html = build_module_audit_html(audits, labels)
"""

from __future__ import annotations

from html import escape as _e
from typing import Optional


def _passed_badge(passed: bool, n_failed: int, label_pass: str,
                  label_fail: str) -> str:
    if passed:
        return (
            f'<span style="color:#16a34a;font-weight:700">'
            f'✓ {_e(label_pass)}</span>'
        )
    return (
        f'<span style="color:#dc2626;font-weight:700">'
        f'✗ {_e(label_fail)} ({n_failed})</span>'
    )


def build_module_audit_html(
    audits: Optional[list],
    labels: Optional[dict[str, str]] = None,
) -> str:
    """Construit la vue HTML « Modules audités ».

    Parameters
    ----------
    audits:
        Liste de dicts ``{"manifest": ManifestDict, "audit":
        AuditResultDict}``.  Si vide ou ``None``, retourne ``""``.
    labels:
        Dict i18n.  Clés sous le préfixe ``audit_*``.
    """
    if not audits:
        return ""
    rows = [
        a for a in audits
        if isinstance(a, dict)
        and isinstance(a.get("manifest"), dict)
        and isinstance(a.get("audit"), dict)
    ]
    if not rows:
        return ""
    labels = labels or {}
    title = labels.get("audit_title", "Modules audités")
    note = labels.get(
        "audit_note",
        "Récapitulatif des modules utilisés dans la pipeline "
        "composée. Un module qui ne passe pas l'audit n'est "
        "pas exécutable. Métadonnées issues du manifest fourni "
        "par le contributeur (auteur, licence, citation).",
    )
    label_pass = labels.get("audit_pass", "audit OK")
    label_fail = labels.get("audit_fail", "checks échoués")
    h_module = labels.get("audit_module", "Module")
    h_status = labels.get("audit_status", "Audit")
    h_version = labels.get("audit_version", "Version")
    h_author = labels.get("audit_author", "Auteur")
    h_license = labels.get("audit_license", "Licence")
    h_io = labels.get("audit_io", "Entrée → sortie")
    h_citation = labels.get("audit_citation", "Citation")
    h_homepage = labels.get("audit_homepage", "Page projet")

    parts = [
        '<section class="audit-section" style="margin:1rem 0">',
        f'<h3 style="margin:0 0 .3rem 0">{_e(title)}</h3>',
        f'<div style="font-size:.85rem;opacity:.75;margin-bottom:.5rem">'
        f'{_e(note)}</div>',
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:.9rem">',
        '<thead><tr>',
    ]
    for col in (h_module, h_status, h_version, h_author,
                h_license, h_io, h_citation, h_homepage):
        parts.append(
            f'<th scope=\"col\" style="padding:.4rem .6rem;text-align:left;'
            f'border-bottom:1px solid #ccc;font-weight:600">'
            f'{_e(col)}</th>'
        )
    parts.append("</tr></thead><tbody>")

    for entry in rows:
        manifest = entry["manifest"]
        audit = entry["audit"]
        name = str(manifest.get("name") or "?")
        version = str(manifest.get("version") or "—")
        author = str(manifest.get("author") or "—")
        license_ = str(manifest.get("license") or "—")
        in_types = ", ".join(manifest.get("input_types") or []) or "—"
        out_types = ", ".join(manifest.get("output_types") or []) or "—"
        citation = manifest.get("citation") or ""
        homepage = manifest.get("homepage") or ""
        passed = bool(audit.get("passed"))
        n_failed = int(audit.get("n_failed") or 0)
        status_cell = _passed_badge(
            passed, n_failed, label_pass, label_fail,
        )
        # Citation : tronqué si trop long
        citation_str = str(citation)[:120]
        if len(str(citation)) > 120:
            citation_str += "…"
        citation_cell = (
            _e(citation_str) if citation_str.strip() else "—"
        )
        # Homepage : on n'auto-link **pas** (anti-injection +
        # honnêteté : l'URL peut pointer ailleurs).  On affiche
        # le texte échappé tel quel.
        homepage_cell = (
            _e(str(homepage))[:80] + ("…" if len(str(homepage)) > 80 else "")
        ) if str(homepage).strip() else "—"
        parts.append(
            f'<tr>'
            f'<td style="padding:.4rem .6rem;font-family:monospace">'
            f'{_e(name)}</td>'
            f'<td style="padding:.4rem .6rem">{status_cell}</td>'
            f'<td style="padding:.4rem .6rem;font-family:monospace">'
            f'{_e(version)}</td>'
            f'<td style="padding:.4rem .6rem">{_e(author)}</td>'
            f'<td style="padding:.4rem .6rem;font-family:monospace">'
            f'{_e(license_)}</td>'
            f'<td style="padding:.4rem .6rem;font-family:monospace;'
            f'font-size:.8rem">{_e(in_types)} → {_e(out_types)}</td>'
            f'<td style="padding:.4rem .6rem;font-size:.8rem;'
            f'opacity:.85">{citation_cell}</td>'
            f'<td style="padding:.4rem .6rem;font-family:monospace;'
            f'font-size:.8rem">{homepage_cell}</td>'
            f'</tr>'
        )
    parts.append("</tbody></table></section>")
    return "".join(parts)


__all__ = ["build_module_audit_html"]
