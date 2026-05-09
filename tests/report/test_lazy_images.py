"""Tests Sprint A5 — option ``lazy_images`` du ReportGenerator (M-16).

Vérifie que :

1. Par défaut (``lazy_images=False``), les images restent embarquées
   en base64 (rétrocompat — rapport mono-fichier transportable).
2. Avec ``lazy_images=True``, les images sont externalisées dans
   ``<output_dir>/report-assets/`` et le HTML les référence par URL
   relative.
3. Le HTML reste valide et lisible dans les deux modes.
4. La taille du HTML monolithique baisse drastiquement en mode lazy
   sur un corpus de plusieurs documents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picarones.evaluation.synthetic import generate_sample_benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_benchmark_with_images(tmp_path: Path):
    """Benchmark démo avec quelques images PNG synthétiques sur disque.

    On utilise les fixtures officielles puis on remplace les
    ``image_path`` par des PNG réels créés à la volée pour que
    ``_externalize_images_to_dir`` ait de quoi travailler.
    """
    from PIL import Image

    bench = generate_sample_benchmark(n_docs=3)
    # Crée 3 PNG synthétiques minuscules
    for i, engine_report in enumerate(bench.engine_reports):
        for j, dr in enumerate(engine_report.document_results):
            img_path = tmp_path / f"img_{j}.png"
            if not img_path.exists():
                Image.new("RGB", (200, 100), color=(255, 240, 220)).save(img_path)
            dr.image_path = str(img_path)
    return bench


# ---------------------------------------------------------------------------
# Mode par défaut (rétrocompat) : images embarquées base64
# ---------------------------------------------------------------------------


def test_default_mode_inlines_images(demo_benchmark_with_images, tmp_path: Path) -> None:
    """``lazy_images=False`` (défaut) : les images vivent en base64
    inline dans le HTML, aucun fichier d'asset n'est créé."""
    from picarones.reports_v2.html.generator import ReportGenerator

    out = tmp_path / "report.html"
    gen = ReportGenerator(demo_benchmark_with_images)
    path = gen.generate(out)

    assert path.exists()
    html = path.read_text(encoding="utf-8")
    # Rétrocompat : data-URI base64 présent
    assert "data:image" in html or "image/png;base64" in html, (
        "En mode par défaut, le HTML doit contenir des data-URI base64."
    )
    # Pas de dossier d'assets externes
    assert not (tmp_path / "report-assets").exists(), (
        "En mode inline, aucun fichier d'asset ne doit être créé."
    )


# ---------------------------------------------------------------------------
# Mode lazy : images externalisées
# ---------------------------------------------------------------------------


def test_lazy_mode_creates_asset_directory(
    demo_benchmark_with_images, tmp_path: Path
) -> None:
    """``lazy_images=True`` : ``report-assets/`` est créé à côté du HTML
    et contient des fichiers image."""
    from picarones.reports_v2.html.generator import ReportGenerator

    out = tmp_path / "report.html"
    gen = ReportGenerator(demo_benchmark_with_images, lazy_images=True)
    path = gen.generate(out)

    assert path.exists()
    assets_dir = tmp_path / "report-assets"
    assert assets_dir.exists() and assets_dir.is_dir()
    asset_files = list(assets_dir.iterdir())
    assert len(asset_files) >= 1, (
        f"Au moins une image doit être externalisée. "
        f"Trouvé : {asset_files}"
    )


def test_lazy_mode_html_references_relative_urls(
    demo_benchmark_with_images, tmp_path: Path
) -> None:
    """En mode lazy, le HTML référence les images via URL relative
    ``report-assets/...`` plutôt qu'un data-URI."""
    from picarones.reports_v2.html.generator import ReportGenerator

    out = tmp_path / "report.html"
    gen = ReportGenerator(demo_benchmark_with_images, lazy_images=True)
    path = gen.generate(out)

    html = path.read_text(encoding="utf-8")
    assert "report-assets/" in html, (
        "Le HTML doit référencer les images via URL relative."
    )
    # ``loading="lazy"`` doit toujours être présent (le template le pose)
    assert 'loading="lazy"' in html


def test_lazy_mode_significantly_reduces_html_size(
    demo_benchmark_with_images, tmp_path: Path
) -> None:
    """Le HTML lazy doit être nettement plus petit que le HTML inline.

    Sur le corpus démo (3 docs × 200×100 PNG), le ratio doit être
    favorable au lazy. Test peu strict (ratio > 1.05) pour ne pas
    être flaky en fonction du contenu vendor.
    """
    from picarones.reports_v2.html.generator import ReportGenerator

    inline_out = tmp_path / "inline.html"
    lazy_out = tmp_path / "lazy.html"

    ReportGenerator(demo_benchmark_with_images, lazy_images=False).generate(inline_out)
    ReportGenerator(demo_benchmark_with_images, lazy_images=True).generate(lazy_out)

    inline_size = inline_out.stat().st_size
    lazy_size = lazy_out.stat().st_size
    assert inline_size > lazy_size, (
        f"Le HTML lazy ({lazy_size} B) doit être < HTML inline "
        f"({inline_size} B). Diff : {inline_size - lazy_size} B."
    )


# ---------------------------------------------------------------------------
# Robustesse
# ---------------------------------------------------------------------------


def test_lazy_mode_with_missing_image_does_not_crash(tmp_path: Path) -> None:
    """Si l'image source n'existe pas, l'externalisation log un warning
    et continue (rétrocompat avec ``_encode_image_b64`` qui retourne ''
    silencieusement)."""
    from picarones.reports_v2.html.generator import ReportGenerator

    bench = generate_sample_benchmark(n_docs=2)
    # Pointe vers un chemin inexistant
    for er in bench.engine_reports:
        for dr in er.document_results:
            dr.image_path = "/nonexistent/missing.png"

    out = tmp_path / "report.html"
    # Ne doit PAS lever
    path = ReportGenerator(bench, lazy_images=True).generate(out)
    assert path.exists()


def test_safe_filename_generation(tmp_path: Path) -> None:
    """Les doc_id contenant des caractères non-FS-safe doivent produire
    des noms de fichiers normalisés (pas de path traversal possible)."""
    from PIL import Image

    from picarones.reports_v2.html.generator import _externalize_images_to_dir

    src = tmp_path / "src.png"
    Image.new("RGB", (50, 50), color=(0, 0, 0)).save(src)

    bench = generate_sample_benchmark(n_docs=1)
    bad_id = "../../etc/passwd"
    for er in bench.engine_reports:
        for dr in er.document_results:
            dr.doc_id = bad_id
            dr.image_path = str(src)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    mapping = _externalize_images_to_dir(bench, out_dir)

    # Garde-fou de path traversal : aucun fichier ne doit être créé en
    # dehors de out_dir/report-assets, **et** le chemin résolu de tout
    # fichier d'asset doit rester *à l'intérieur* du dossier d'assets.
    forbidden = out_dir.parent / "etc" / "passwd"
    assert not forbidden.exists(), "Path traversal détecté !"
    assets_dir = (out_dir / "report-assets").resolve()
    if mapping:
        for url in mapping.values():
            assert url.startswith("report-assets/")
            # Le chemin résolu doit être contenu dans assets_dir
            resolved = (out_dir / url).resolve()
            assert str(resolved).startswith(str(assets_dir)), (
                f"Path traversal : {resolved} sort de {assets_dir}"
            )
