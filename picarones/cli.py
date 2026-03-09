"""Interface en ligne de commande Picarones (Click).

Commandes disponibles
---------------------
picarones run         — Lance un benchmark complet
picarones report      — Génère le rapport HTML depuis un JSON de résultats
picarones demo        — Génère un rapport de démonstration avec données fictives
picarones metrics     — Calcule CER/WER entre deux fichiers texte
picarones engines     — Liste les moteurs disponibles
picarones info        — Informations de version
picarones history     — Consulte l'historique des benchmarks (suivi longitudinal)
picarones robustness  — Lance une analyse de robustesse sur un corpus

Exemples d'usage
----------------
    picarones run --corpus ./corpus/ --engines tesseract --output results.json
    picarones metrics --reference gt.txt --hypothesis ocr.txt
    picarones history --engine tesseract
    picarones robustness --corpus ./gt/ --engine tesseract
    picarones engines
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from picarones import __version__

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _engine_from_name(engine_name: str, lang: str, psm: int) -> "BaseOCREngine":
    """Instancie un moteur par son nom."""
    from picarones.engines.tesseract import TesseractEngine

    if engine_name in {"tesseract", "tess"}:
        return TesseractEngine(config={"lang": lang, "psm": psm})

    try:
        from picarones.engines.pero_ocr import PeroOCREngine

        if engine_name in {"pero_ocr", "pero"}:
            return PeroOCREngine(config={"name": "pero_ocr"})
    except ImportError:
        pass

    raise click.BadParameter(
        f"Moteur inconnu ou non disponible : '{engine_name}'. "
        "Moteurs supportés : tesseract, pero_ocr"
    )


# ---------------------------------------------------------------------------
# Groupe principal
# ---------------------------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="picarones")
def cli() -> None:
    """Picarones — Plateforme de comparaison de moteurs OCR pour documents patrimoniaux."""


# ---------------------------------------------------------------------------
# picarones run
# ---------------------------------------------------------------------------

@cli.command("run")
@click.option(
    "--corpus", "-c",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Dossier contenant les paires image / .gt.txt",
)
@click.option(
    "--engines", "-e",
    default="tesseract",
    show_default=True,
    help="Liste de moteurs séparés par des virgules (ex : tesseract,pero_ocr)",
)
@click.option(
    "--output", "-o",
    default="results.json",
    show_default=True,
    type=click.Path(resolve_path=True),
    help="Fichier JSON de sortie",
)
@click.option(
    "--lang", "-l",
    default="fra",
    show_default=True,
    help="Code langue Tesseract (fra, lat, eng…)",
)
@click.option("--psm", default=6, show_default=True, help="Page Segmentation Mode Tesseract (0-13)")
@click.option("--no-progress", is_flag=True, default=False, help="Désactive la barre de progression")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
@click.option(
    "--fail-if-cer-above",
    default=None,
    type=float,
    metavar="THRESHOLD",
    help="Quitte avec code 1 si CER moyen > THRESHOLD (usage CI/CD)",
)
def run_cmd(
    corpus: str,
    engines: str,
    output: str,
    lang: str,
    psm: int,
    no_progress: bool,
    verbose: bool,
    fail_if_cer_above: float | None,
) -> None:
    """Lance un benchmark OCR sur un corpus de documents.

    Le corpus doit être un dossier contenant des paires
    <image>.<ext> + <image>.gt.txt (vérité terrain).
    """
    _setup_logging(verbose)

    from picarones.core.corpus import load_corpus_from_directory
    from picarones.core.runner import run_benchmark

    # Chargement du corpus
    try:
        corp = load_corpus_from_directory(corpus)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Erreur corpus : {exc}", err=True)
        sys.exit(1)

    click.echo(f"Corpus '{corp.name}' — {len(corp)} documents chargés.")

    # Instanciation des moteurs
    engine_names = [e.strip() for e in engines.split(",") if e.strip()]
    ocr_engines = []
    for name in engine_names:
        try:
            engine = _engine_from_name(name, lang=lang, psm=psm)
            ocr_engines.append(engine)
        except click.BadParameter as exc:
            click.echo(f"Erreur moteur : {exc}", err=True)
            sys.exit(1)

    if not ocr_engines:
        click.echo("Aucun moteur valide spécifié.", err=True)
        sys.exit(1)

    click.echo(f"Moteurs : {', '.join(e.name for e in ocr_engines)}")

    # Lancement du benchmark
    result = run_benchmark(
        corpus=corp,
        engines=ocr_engines,
        output_json=output,
        show_progress=not no_progress,
    )

    # Affichage du classement
    click.echo("\n── Classement ──────────────────────────────────")
    for rank, entry in enumerate(result.ranking(), 1):
        cer_pct = f"{entry['mean_cer'] * 100:.2f}%" if entry["mean_cer"] is not None else "N/A"
        wer_pct = f"{entry['mean_wer'] * 100:.2f}%" if entry["mean_wer"] is not None else "N/A"
        failed = entry["failed"]
        failed_str = f" ({failed} erreur(s))" if failed else ""
        click.echo(f"  {rank}. {entry['engine']:<20} CER={cer_pct:<8} WER={wer_pct}{failed_str}")

    click.echo(f"\nRésultats écrits dans : {output}")

    # Mode CI/CD : exit code non-zero si CER > seuil
    if fail_if_cer_above is not None:
        for entry in result.ranking():
            if entry["mean_cer"] is not None and entry["mean_cer"] * 100 > fail_if_cer_above:
                click.echo(
                    f"\nECHEC : {entry['engine']} CER={entry['mean_cer']*100:.2f}% "
                    f"> seuil {fail_if_cer_above:.2f}%",
                    err=True,
                )
                sys.exit(1)


# ---------------------------------------------------------------------------
# picarones metrics
# ---------------------------------------------------------------------------

@cli.command("metrics")
@click.option(
    "--reference", "-r",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Fichier vérité terrain (texte brut UTF-8)",
)
@click.option(
    "--hypothesis", "-H",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Fichier transcription OCR (texte brut UTF-8)",
)
@click.option("--json-output", is_flag=True, default=False, help="Sortie en JSON")
def metrics_cmd(reference: str, hypothesis: str, json_output: bool) -> None:
    """Calcule CER et WER entre deux fichiers texte."""
    from picarones.core.metrics import compute_metrics

    ref_text = Path(reference).read_text(encoding="utf-8").strip()
    hyp_text = Path(hypothesis).read_text(encoding="utf-8").strip()

    result = compute_metrics(ref_text, hyp_text)

    if json_output:
        click.echo(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        click.echo(f"CER            : {result.cer_percent:.2f}%")
        click.echo(f"CER (NFC)      : {result.cer_nfc * 100:.2f}%")
        click.echo(f"CER (caseless) : {result.cer_caseless * 100:.2f}%")
        click.echo(f"WER            : {result.wer_percent:.2f}%")
        click.echo(f"WER (normalisé): {result.wer_normalized * 100:.2f}%")
        click.echo(f"MER            : {result.mer * 100:.2f}%")
        click.echo(f"WIL            : {result.wil * 100:.2f}%")
        click.echo(f"Longueur GT    : {result.reference_length} chars")
        click.echo(f"Longueur OCR   : {result.hypothesis_length} chars")
        if result.error:
            click.echo(f"Erreur         : {result.error}", err=True)


# ---------------------------------------------------------------------------
# picarones engines
# ---------------------------------------------------------------------------

@cli.command("engines")
def engines_cmd() -> None:
    """Liste les moteurs OCR disponibles et vérifie leur installation."""
    engines = [
        ("tesseract", "Tesseract 5 (pytesseract)", "pytesseract"),
        ("pero_ocr", "Pero OCR", "pero_ocr"),
    ]

    click.echo("Moteurs OCR disponibles :\n")
    for engine_id, label, module in engines:
        try:
            __import__(module)
            status = click.style("✓ disponible", fg="green")
        except ImportError:
            status = click.style("✗ non installé", fg="red")
        click.echo(f"  {engine_id:<15} {label:<35} {status}")

    click.echo(
        "\nPour installer un moteur manquant :\n"
        "  pip install pytesseract\n"
        "  pip install pero-ocr"
    )


# ---------------------------------------------------------------------------
# picarones info
# ---------------------------------------------------------------------------

@cli.command("info")
def info_cmd() -> None:
    """Affiche les informations de version de Picarones et de ses dépendances."""
    click.echo(f"Picarones v{__version__}")
    click.echo("")

    deps = [
        ("click", "click"),
        ("jiwer", "jiwer"),
        ("Pillow", "PIL"),
        ("pytesseract", "pytesseract"),
        ("tqdm", "tqdm"),
        ("numpy", "numpy"),
        ("pyyaml", "yaml"),
    ]

    click.echo("Dépendances :")
    for name, module in deps:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installé")
            status = click.style(f"v{version}", fg="green")
        except ImportError:
            status = click.style("non installé", fg="red")
        click.echo(f"  {name:<15} {status}")


# ---------------------------------------------------------------------------
# picarones report
# ---------------------------------------------------------------------------

@cli.command("report")
@click.option(
    "--results", "-r",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Fichier JSON de résultats produit par 'picarones run'",
)
@click.option(
    "--output", "-o",
    default="rapport.html",
    show_default=True,
    type=click.Path(resolve_path=True),
    help="Fichier HTML de sortie",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def report_cmd(results: str, output: str, verbose: bool) -> None:
    """Génère le rapport HTML interactif depuis un fichier JSON de résultats.

    Le rapport est un fichier HTML auto-contenu, lisible hors-ligne,
    avec tableau de classement, galerie, vue document et graphiques.
    """
    _setup_logging(verbose)

    from picarones.report.generator import ReportGenerator

    click.echo(f"Chargement des résultats : {results}")
    try:
        gen = ReportGenerator.from_json(results)
    except Exception as exc:
        click.echo(f"Erreur lors du chargement : {exc}", err=True)
        sys.exit(1)

    click.echo(f"Génération du rapport HTML…")
    path = gen.generate(output)
    click.echo(f"Rapport généré : {path}")
    click.echo(f"Ouvrez-le dans un navigateur : file://{path}")


# ---------------------------------------------------------------------------
# picarones demo
# ---------------------------------------------------------------------------

@cli.command("demo")
@click.option(
    "--output", "-o",
    default="rapport_demo.html",
    show_default=True,
    type=click.Path(resolve_path=True),
    help="Fichier HTML de sortie",
)
@click.option(
    "--docs", "-n",
    default=12,
    show_default=True,
    type=click.IntRange(1, 12),
    help="Nombre de documents fictifs (1–12)",
)
@click.option(
    "--json-output", "-j",
    default=None,
    type=click.Path(resolve_path=True),
    help="Exporte aussi les résultats JSON",
)
@click.option(
    "--with-history",
    is_flag=True,
    default=False,
    help="Inclut une démonstration du suivi longitudinal (8 runs fictifs)",
)
@click.option(
    "--with-robustness",
    is_flag=True,
    default=False,
    help="Inclut une démonstration de l'analyse de robustesse",
)
@click.option(
    "--lang",
    default="fr",
    show_default=True,
    type=click.Choice(["fr", "en"], case_sensitive=False),
    help="Langue du rapport HTML généré (fr = français, en = anglais patrimonial)",
)
def demo_cmd(
    output: str,
    docs: int,
    json_output: str | None,
    with_history: bool,
    with_robustness: bool,
    lang: str,
) -> None:
    """Génère un rapport de démonstration avec des données fictives réalistes.

    Utile pour tester le rendu HTML sans installer Tesseract ni Pero OCR.

    \b
    Exemples :
        picarones demo
        picarones demo --lang en
        picarones demo --with-history
        picarones demo --with-robustness
        picarones demo --with-history --with-robustness --docs 8
    """
    from picarones.fixtures import generate_sample_benchmark
    from picarones.report.generator import ReportGenerator

    click.echo(f"Génération des données fictives ({docs} documents, 3 moteurs)…")
    benchmark = generate_sample_benchmark(n_docs=docs)

    if json_output:
        bm_path = benchmark.to_json(json_output)
        click.echo(f"Résultats JSON : {bm_path}")

    gen = ReportGenerator(benchmark, lang=lang)
    path = gen.generate(output)
    click.echo(f"Rapport de démonstration : {path}")
    click.echo(f"Ouvrez-le dans un navigateur : file://{path}")

    # Suivi longitudinal
    if with_history:
        click.echo("\n── Démonstration suivi longitudinal ──────────────")
        from picarones.core.history import BenchmarkHistory, generate_demo_history
        history = BenchmarkHistory(":memory:")
        generate_demo_history(history, n_runs=8)
        entries = history.query(engine="tesseract")
        click.echo(f"  {history.count()} entrées générées (8 runs, 3 moteurs).")
        click.echo("\n  Évolution du CER — tesseract :")
        for e in entries:
            cer_str = f"{e.cer_percent:.2f}%" if e.cer_percent is not None else "N/A"
            bar = "█" * int((e.cer_percent or 0) * 2)
            click.echo(f"    {e.timestamp[:10]}  {cer_str:<8}  {bar}")
        regression = history.detect_regression("tesseract", threshold=0.01)
        if regression and regression.is_regression:
            click.echo(
                click.style(
                    f"\n  RÉGRESSION détectée ! delta CER = +{regression.delta_cer * 100:.2f}%",
                    fg="red",
                )
            )
        else:
            click.echo(click.style("\n  Aucune régression détectée.", fg="green"))

    # Analyse de robustesse
    if with_robustness:
        click.echo("\n── Démonstration analyse de robustesse ───────────")
        from picarones.core.robustness import generate_demo_robustness_report
        report = generate_demo_robustness_report(
            engine_names=["tesseract", "pero_ocr"]
        )
        for curve in report.curves:
            if curve.degradation_type == "noise":
                click.echo(f"\n  {curve.engine_name} / bruit gaussien :")
                for label, cer in zip(curve.labels, curve.cer_values):
                    cer_pct = f"{(cer or 0) * 100:.1f}%"
                    bar = "█" * int((cer or 0) * 40)
                    click.echo(f"    {label:<12} {cer_pct:<8} {bar}")
                if curve.critical_threshold_level is not None:
                    click.echo(
                        click.style(
                            f"    Niveau critique (CER>20%) : σ={curve.critical_threshold_level}",
                            fg="yellow",
                        )
                    )


# ---------------------------------------------------------------------------
# picarones import (groupe de sous-commandes)
# ---------------------------------------------------------------------------

@cli.group("import")
def import_group() -> None:
    """Importe un corpus depuis une source distante (IIIF, HuggingFace…)."""


@import_group.command("iiif")
@click.argument("manifest_url")
@click.option(
    "--pages", "-p",
    default="all",
    show_default=True,
    help=(
        "Pages à importer. Formats : '1-10', '1,3,5', '1-5,10,15-20', 'all'. "
        "Les numéros sont 1-based (1 = première page du manifeste)."
    ),
)
@click.option(
    "--output", "-o",
    default="./corpus_iiif/",
    show_default=True,
    type=click.Path(resolve_path=True),
    help="Dossier de destination pour les images et les fichiers .gt.txt",
)
@click.option(
    "--max-resolution",
    default=0,
    type=int,
    show_default=True,
    help="Résolution maximale des images téléchargées (largeur en pixels). 0 = max disponible.",
)
@click.option("--no-progress", is_flag=True, default=False, help="Désactive la barre de progression")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def import_iiif_cmd(
    manifest_url: str,
    pages: str,
    output: str,
    max_resolution: int,
    no_progress: bool,
    verbose: bool,
) -> None:
    """Importe un corpus depuis un manifeste IIIF (v2 ou v3).

    MANIFEST_URL : URL du manifeste IIIF (Gallica, Bodleian, BL, BSB…)

    Exemples :

    \b
        picarones import iiif https://gallica.bnf.fr/ark:/12148/xxx/manifest.json
        picarones import iiif https://gallica.bnf.fr/ark:/12148/xxx/manifest.json --pages 1-10
        picarones import iiif https://gallica.bnf.fr/ark:/12148/xxx/manifest.json --pages 1,3,5-8 --output ./mon_corpus/

    Les images sont téléchargées dans le dossier de sortie.
    Des fichiers .gt.txt vides (ou remplis si le manifeste contient des annotations
    de transcription) sont créés à côté de chaque image.
    """
    _setup_logging(verbose)

    from picarones.importers.iiif import IIIFImporter

    click.echo(f"Manifeste IIIF : {manifest_url}")

    try:
        importer = IIIFImporter(manifest_url, max_resolution=max_resolution)
        importer.load()

        all_canvases = importer.parser.canvases()
        click.echo(
            f"Manifeste IIIF v{importer.parser.version} — "
            f"titre : {importer.parser.label} — "
            f"{len(all_canvases)} canvas disponibles"
        )

        selected = importer.list_canvases(pages)
        click.echo(f"Pages sélectionnées : {len(selected)} sur {len(all_canvases)}")

        corpus = importer.import_corpus(
            pages=pages,
            output_dir=output,
            show_progress=not no_progress,
        )

    except (ValueError, RuntimeError) as exc:
        click.echo(f"Erreur import IIIF : {exc}", err=True)
        sys.exit(1)

    click.echo(f"\n{len(corpus)} documents importés dans : {output}")

    # Résumé
    gt_filled = sum(1 for d in corpus.documents if d.ground_truth.strip())
    if gt_filled:
        click.echo(f"Transcriptions trouvées dans le manifeste : {gt_filled}/{len(corpus)}")
    else:
        click.echo(
            "Aucune transcription dans le manifeste — "
            "les fichiers .gt.txt sont vides (à remplir manuellement ou via OCR)."
        )

    click.echo(f"\nPour lancer un benchmark sur ce corpus :")
    click.echo(f"  picarones run --corpus {output} --engines tesseract")


# ---------------------------------------------------------------------------
# picarones serve
# ---------------------------------------------------------------------------

@cli.command("serve")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Adresse d'écoute du serveur web",
)
@click.option(
    "--port", "-p",
    default=8000,
    show_default=True,
    type=click.IntRange(1, 65535),
    help="Port d'écoute du serveur web",
)
@click.option("--reload", is_flag=True, default=False, help="Mode rechargement automatique (développement)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def serve_cmd(host: str, port: int, reload: bool, verbose: bool) -> None:
    """Lance l'interface web locale Picarones sur localhost.

    Accessible dans le navigateur à l'adresse : http://HOST:PORT

    \b
    Exemples :
        picarones serve
        picarones serve --port 8080
        picarones serve --host 0.0.0.0 --port 8000
    """
    _setup_logging(verbose)

    try:
        import uvicorn
    except ImportError:
        click.echo(
            "uvicorn n'est pas installé. Installez-le avec :\n"
            "  pip install uvicorn[standard]\n"
            "ou :\n"
            "  pip install picarones[web]",
            err=True,
        )
        sys.exit(1)

    url = f"http://{host}:{port}"
    click.echo(f"Picarones — Interface web locale")
    click.echo(f"Démarrage du serveur sur {url}")
    click.echo(f"Appuyez sur Ctrl+C pour arrêter.\n")

    log_level = "debug" if verbose else "info"
    uvicorn.run(
        "picarones.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


# ---------------------------------------------------------------------------
# picarones history
# ---------------------------------------------------------------------------

@cli.command("history")
@click.option(
    "--db",
    default="~/.picarones/history.db",
    show_default=True,
    type=click.Path(resolve_path=False),
    help="Chemin vers la base SQLite d'historique",
)
@click.option(
    "--engine", "-e",
    default=None,
    help="Filtre sur le nom du moteur",
)
@click.option(
    "--corpus", "-c",
    default=None,
    help="Filtre sur le nom du corpus",
)
@click.option(
    "--since",
    default=None,
    metavar="DATE",
    help="Date minimale ISO 8601 (ex: 2025-01-01)",
)
@click.option(
    "--limit", "-n",
    default=50,
    show_default=True,
    type=click.IntRange(1, 10000),
    help="Nombre maximum d'entrées à afficher",
)
@click.option(
    "--regression",
    is_flag=True,
    default=False,
    help="Détecter automatiquement les régressions (compare au run précédent)",
)
@click.option(
    "--regression-threshold",
    default=0.01,
    show_default=True,
    type=float,
    metavar="DELTA",
    help="Seuil de régression en points de CER absolus (ex: 0.01 = 1%)",
)
@click.option(
    "--export-json",
    default=None,
    type=click.Path(resolve_path=True),
    help="Exporte l'historique complet en JSON",
)
@click.option(
    "--demo",
    is_flag=True,
    default=False,
    help="Pré-remplir la base avec des données fictives de démonstration",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def history_cmd(
    db: str,
    engine: str | None,
    corpus: str | None,
    since: str | None,
    limit: int,
    regression: bool,
    regression_threshold: float,
    export_json: str | None,
    demo: bool,
    verbose: bool,
) -> None:
    """Consulte l'historique des benchmarks (suivi longitudinal).

    Affiche l'évolution du CER dans le temps pour chaque moteur et corpus.
    Permet de détecter automatiquement les régressions entre deux runs.

    \b
    Exemples :
        picarones history
        picarones history --engine tesseract --corpus "Chroniques BnF"
        picarones history --regression --regression-threshold 0.02
        picarones history --demo   # données fictives de démonstration
        picarones history --export-json historique.json
    """
    _setup_logging(verbose)

    from picarones.core.history import BenchmarkHistory, generate_demo_history

    history = BenchmarkHistory(db)

    if demo:
        click.echo("Insertion de données fictives de démonstration dans l'historique…")
        generate_demo_history(history, n_runs=8)
        click.echo(f"  {history.count()} entrées insérées.")

    if export_json:
        path = history.export_json(export_json)
        click.echo(f"Historique exporté : {path}")
        return

    entries = history.query(engine=engine, corpus=corpus, since=since, limit=limit)

    if not entries:
        click.echo("Aucun benchmark dans l'historique.")
        click.echo(
            "\nPour enregistrer automatiquement les runs, utilisez :\n"
            "  picarones run --corpus ./gt/ --engines tesseract --save-history\n"
            "\nOu pour tester avec des données fictives :\n"
            "  picarones history --demo"
        )
        return

    # Regrouper par moteur
    by_engine: dict[str, list] = {}
    for entry in entries:
        by_engine.setdefault(entry.engine_name, []).append(entry)

    click.echo(f"\n── Historique des benchmarks ({'filtré' if engine or corpus else 'tous'}) ──")
    click.echo(f"  Base : {history.db_path}")
    click.echo(f"  Total entrées : {len(entries)}\n")

    for eng_name, eng_entries in by_engine.items():
        click.echo(click.style(f"  Moteur : {eng_name}", bold=True))
        for e in eng_entries:
            cer_str = f"{e.cer_percent:.2f}%" if e.cer_percent is not None else "N/A"
            wer_str = f"{e.wer_mean * 100:.2f}%" if e.wer_mean is not None else "N/A"
            ts = e.timestamp[:10]  # date uniquement
            click.echo(f"    {ts}  CER={cer_str:<8} WER={wer_str:<8} docs={e.doc_count}  corpus={e.corpus_name}")
        click.echo()

    # Détection de régression
    if regression:
        click.echo("── Détection de régressions ──────────────────────")
        regressions = history.detect_all_regressions(threshold=regression_threshold)
        if not regressions:
            click.echo(
                click.style(
                    f"  Aucune régression détectée (seuil={regression_threshold*100:.1f}%)",
                    fg="green",
                )
            )
        else:
            for r in regressions:
                delta_str = f"+{r.delta_cer * 100:.2f}%" if r.delta_cer else "N/A"
                click.echo(
                    click.style(
                        f"  RÉGRESSION {r.engine_name} / {r.corpus_name} : "
                        f"delta CER={delta_str} "
                        f"({r.baseline_timestamp[:10]} → {r.current_timestamp[:10]})",
                        fg="red",
                    )
                )


# ---------------------------------------------------------------------------
# picarones robustness
# ---------------------------------------------------------------------------

@cli.command("robustness")
@click.option(
    "--corpus", "-c",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Dossier contenant les paires image / .gt.txt",
)
@click.option(
    "--engine", "-e",
    default="tesseract",
    show_default=True,
    help="Moteur OCR à tester (tesseract, pero_ocr…)",
)
@click.option(
    "--degradations", "-d",
    default="noise,blur,rotation,resolution,binarization",
    show_default=True,
    help="Types de dégradation séparés par des virgules",
)
@click.option(
    "--cer-threshold",
    default=0.20,
    show_default=True,
    type=float,
    metavar="THRESHOLD",
    help="Seuil CER pour définir le niveau critique (0-1)",
)
@click.option(
    "--max-docs",
    default=10,
    show_default=True,
    type=click.IntRange(1, 1000),
    help="Nombre maximum de documents à traiter",
)
@click.option(
    "--output-json", "-o",
    default=None,
    type=click.Path(resolve_path=True),
    help="Exporte le rapport de robustesse en JSON",
)
@click.option(
    "--lang", "-l",
    default="fra",
    show_default=True,
    help="Code langue Tesseract",
)
@click.option("--no-progress", is_flag=True, default=False, help="Désactive la barre de progression")
@click.option("--demo", is_flag=True, default=False, help="Mode démo avec données fictives (sans OCR réel)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def robustness_cmd(
    corpus: str,
    engine: str,
    degradations: str,
    cer_threshold: float,
    max_docs: int,
    output_json: str | None,
    lang: str,
    no_progress: bool,
    demo: bool,
    verbose: bool,
) -> None:
    """Lance une analyse de robustesse d'un moteur OCR face aux dégradations d'image.

    Génère des versions dégradées des images (bruit, flou, rotation,
    réduction de résolution, binarisation) et mesure le CER à chaque niveau.

    \b
    Exemples :
        picarones robustness --corpus ./gt/ --engine tesseract
        picarones robustness --corpus ./gt/ --engine pero_ocr --degradations noise,blur
        picarones robustness --corpus ./gt/ --engine tesseract --output-json robustness.json
        picarones robustness --corpus ./gt/ --engine tesseract --demo
    """
    _setup_logging(verbose)

    import json as _json

    deg_types = [d.strip() for d in degradations.split(",") if d.strip()]

    from picarones.core.robustness import (
        RobustnessAnalyzer, ALL_DEGRADATION_TYPES, generate_demo_robustness_report
    )

    # Valider les types de dégradation
    invalid = [d for d in deg_types if d not in ALL_DEGRADATION_TYPES]
    if invalid:
        click.echo(
            f"Types de dégradation invalides : {', '.join(invalid)}\n"
            f"Types valides : {', '.join(ALL_DEGRADATION_TYPES)}",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Corpus       : {corpus}")
    click.echo(f"Moteur       : {engine}")
    click.echo(f"Dégradations : {', '.join(deg_types)}")
    click.echo(f"Seuil CER    : {cer_threshold * 100:.0f}%")

    if demo:
        click.echo("\nMode démo : génération d'un rapport fictif réaliste…")
        report = generate_demo_robustness_report(engine_names=[engine])
    else:
        # Charger le corpus
        from picarones.core.corpus import load_corpus_from_directory
        try:
            corp = load_corpus_from_directory(corpus)
        except (FileNotFoundError, ValueError) as exc:
            click.echo(f"Erreur corpus : {exc}", err=True)
            sys.exit(1)

        click.echo(f"\n{len(corp)} documents chargés. Début de l'analyse…\n")

        # Instancier le moteur
        try:
            ocr_engine = _engine_from_name(engine, lang=lang, psm=6)
        except click.BadParameter as exc:
            click.echo(f"Erreur moteur : {exc}", err=True)
            sys.exit(1)

        from picarones.core.robustness import RobustnessAnalyzer
        analyzer = RobustnessAnalyzer(
            engines=[ocr_engine],
            degradation_types=deg_types,
            cer_threshold=cer_threshold,
        )
        report = analyzer.analyze(
            corpus=corp,
            show_progress=not no_progress,
            max_docs=max_docs,
        )

    # Affichage des résultats
    click.echo("\n── Résultats de robustesse ──────────────────────────")
    for curve in report.curves:
        click.echo(f"\n  {curve.engine_name} / {curve.degradation_type}")
        for label, cer in zip(curve.labels, curve.cer_values):
            if cer is not None:
                bar_len = int(cer * 40)
                bar = "█" * bar_len
                cer_pct = f"{cer * 100:.1f}%"
                threshold_marker = " ← CRITIQUE" if curve.critical_threshold_level is not None and \
                    curve.levels[curve.labels.index(label)] == curve.critical_threshold_level else ""
                click.echo(f"    {label:<12} {cer_pct:<8} {bar}{threshold_marker}")
        if curve.critical_threshold_level is not None:
            click.echo(
                click.style(
                    f"    Niveau critique (CER>{cer_threshold*100:.0f}%) : {curve.critical_threshold_level}",
                    fg="yellow",
                )
            )
        else:
            click.echo(click.style(f"    Robuste jusqu'au niveau max.", fg="green"))

    # Résumé
    click.echo("\n── Résumé ──────────────────────────────────────────")
    for key, val in report.summary.items():
        if key.startswith("most_robust_"):
            deg = key.replace("most_robust_", "")
            click.echo(f"  Moteur le plus robuste ({deg}) : {val}")

    # Export JSON
    if output_json:
        report_dict = report.as_dict()
        Path(output_json).write_text(
            _json.dumps(report_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.echo(f"\nRapport JSON exporté : {output_json}")


# ---------------------------------------------------------------------------
# Mise à jour de picarones demo pour illustrer suivi longitudinal + robustesse
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cli()
