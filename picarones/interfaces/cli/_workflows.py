"""Commandes workflows benchmark : run, diagnose, economics, edition, compare.

Sous-module CLI extrait de l'ancien ``picarones/cli.py`` (1519 lignes)
lors du chantier 5 post-Sprint 97.  Les commandes ici s'enregistrent
automatiquement sur le groupe ``cli`` à l'import.

Comportement et signatures inchangés — uniquement de la modularisation.
"""

from __future__ import annotations

import json
import sys

import click

from picarones.interfaces.cli import cli, _engine_from_name, _setup_logging


def _validate_cer_threshold(
    ctx: click.Context, param: click.Parameter, value: float | None,
) -> float | None:
    """Callback Click qui valide ``--fail-if-cer-above`` à l'analyse.

    Sémantique : fraction ∈ [0, 1] (ex : 0.15 = 15 %), cohérent avec
    ``BenchmarkResult.ranking()[i]["mean_cer"]`` qui est aussi en
    fraction.

    Garde-fou migration : avant le fix de sémantique, le seuil était
    interprété comme un pourcentage (15.0 = 15 %).  Tout caller qui
    passe encore une valeur > 1 vient de l'ancienne sémantique — on
    échoue bruyamment plutôt que de muter silencieusement le
    comportement (un seuil de 1500 % ne se déclencherait jamais et
    l'utilisateur croirait que son CI est sain).
    """
    if value is None:
        return None
    if value < 0:
        raise click.BadParameter(
            f"doit être ≥ 0, reçu {value}.",
        )
    if value > 1.0:
        raise click.BadParameter(
            f"doit être une fraction ∈ [0, 1] (ex : 0.15 = 15 %), "
            f"reçu {value}. Si vous utilisiez l'ancienne sémantique "
            "pourcentage, divisez par 100 (ex : 15.0 → 0.15).",
        )
    return value


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
    callback=_validate_cer_threshold,
    help=(
        "Quitte avec code 1 si CER moyen > THRESHOLD (usage CI/CD). "
        "THRESHOLD est une fraction ∈ [0, 1] (ex : 0.15 = 15 %)."
    ),
)
@click.option(
    "--profile",
    default="standard",
    show_default=True,
    type=click.Choice([
        "minimal", "standard", "philological", "diagnostics",
        "economics", "pipeline", "full",
    ]),
    help=(
        "Profil de calcul des métriques (chantier 2 post-Sprint 97). "
        "'minimal' calcule uniquement CER/WER (rapide, bench massif). "
        "'standard' active les 12 hooks historiques (défaut, rétrocompat). "
        "Voir docs/profiles/ pour le détail."
    ),
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
    profile: str,
) -> None:
    """Lance un benchmark OCR sur un corpus de documents.

    Le corpus doit être un dossier contenant des paires
    <image>.<ext> + <image>.gt.txt (vérité terrain).

    ``--fail-if-cer-above`` est validé à l'analyse Click (cf.
    ``_validate_cer_threshold``) — une valeur invalide est rejetée
    avant toute opération coûteuse.
    """
    _setup_logging(verbose)

    from picarones.evaluation.corpus import load_corpus_from_directory
    from picarones.app.services.benchmark_runner import run_benchmark_via_service

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
    click.echo(f"Profil de métriques : {profile}")

    # Lancement du benchmark
    result = run_benchmark_via_service(
        corpus=corp,
        engines=ocr_engines,
        output_json=output,
        show_progress=not no_progress,
        profile=profile,
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

    # Mode CI/CD : exit code non-zero si CER > seuil.
    # ``fail_if_cer_above`` est déjà validé en tête de fonction (∈ [0, 1]).
    if fail_if_cer_above is not None:
        for entry in result.ranking():
            if (
                entry["mean_cer"] is not None
                and entry["mean_cer"] > fail_if_cer_above
            ):
                click.echo(
                    f"\nECHEC : {entry['engine']} "
                    f"CER={entry['mean_cer']*100:.2f}% "
                    f"> seuil {fail_if_cer_above*100:.2f}%",
                    err=True,
                )
                sys.exit(1)


# ---------------------------------------------------------------------------
# Workflows CLI dédiés (chantier 4 post-Sprint 97)
# ---------------------------------------------------------------------------
#
# Chaque commande spécialisée fixe un profil de calcul (chantier 2) et
# émet un message identifiant la famille avant de déléguer au runner.
# L'option ``--profile`` reste disponible mais le défaut change pour
# chaque commande.

def _html_path_from_json(json_path: str) -> str:
    """Convertit un chemin ``results.json`` en chemin ``results.html``.

    Utilisé par les workflows pour générer automatiquement le rapport
    HTML à côté du JSON (Phase 4.5 du chantier post-rewrite — auparavant
    chaque workflow imprimait juste le chemin JSON et l'utilisateur
    devait relancer ``picarones report --results …`` manuellement,
    contre-intuitif vu que le workflow vendait un livrable HTML).
    """
    from pathlib import Path
    p = Path(json_path)
    return str(p.with_suffix(".html"))


def _run_workflow(
    *,
    corpus: str,
    engines: str,
    output: str,
    lang: str,
    psm: int,
    no_progress: bool,
    verbose: bool,
    profile: str,
    workflow_label: str,
    generate_html: bool = True,
    html_lang: str = "fr",
) -> None:
    """Implémentation commune des commandes ``run``, ``diagnose``,
    ``economics`` et ``edition``.

    Les 4 commandes partagent le squelette : chargement corpus →
    instanciation moteurs → ``run_benchmark_via_service(profile=...)`` → affichage
    classement → génération automatique du rapport HTML.  Seul le profil
    par défaut et le message d'en-tête diffèrent.

    Phase 4.5 du chantier post-rewrite : ``generate_html=True`` par
    défaut.  Auparavant les workflows ne produisaient que du JSON, ce
    qui forçait l'utilisateur à ré-exécuter ``picarones report``
    manuellement — contre-intuitif (les docstrings vendaient une vue
    HTML "Diagnostic", "Coût et performance", "Taxonomie avancée"
    qui n'était jamais générée).  Passer ``generate_html=False``
    permet de désactiver pour les usages CI/scripts qui ne veulent
    que le JSON.
    """
    _setup_logging(verbose)

    from picarones.evaluation.corpus import load_corpus_from_directory
    from picarones.app.services.benchmark_runner import run_benchmark_via_service

    try:
        corp = load_corpus_from_directory(corpus)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Erreur corpus : {exc}", err=True)
        sys.exit(1)

    click.echo(f"[{workflow_label}] Corpus '{corp.name}' — "
               f"{len(corp)} documents chargés.")

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
    click.echo(f"Profil de métriques : {profile}")

    result = run_benchmark_via_service(
        corpus=corp,
        engines=ocr_engines,
        output_json=output,
        show_progress=not no_progress,
        profile=profile,
    )

    click.echo("\n── Classement ──────────────────────────────────")
    for rank, entry in enumerate(result.ranking(), 1):
        cer_pct = (
            f"{entry['mean_cer'] * 100:.2f}%"
            if entry["mean_cer"] is not None else "N/A"
        )
        wer_pct = (
            f"{entry['mean_wer'] * 100:.2f}%"
            if entry["mean_wer"] is not None else "N/A"
        )
        failed = entry["failed"]
        failed_str = f" ({failed} erreur(s))" if failed else ""
        click.echo(
            f"  {rank}. {entry['engine']:<20} "
            f"CER={cer_pct:<8} WER={wer_pct}{failed_str}"
        )

    click.echo(f"\nRésultats JSON écrits dans : {output}")

    if generate_html:
        html_output = _html_path_from_json(output)
        try:
            from picarones.reports.html.generator import ReportGenerator
            gen = ReportGenerator(result, lang=html_lang)
            gen.generate(html_output)
            click.echo(f"Rapport HTML généré    : {html_output}")
        except Exception as exc:  # noqa: BLE001
            # Le JSON est déjà écrit ; on logue l'échec HTML sans
            # quitter avec un code d'erreur (l'utilisateur peut
            # relancer ``picarones report`` manuellement).
            click.echo(
                f"Avertissement : génération HTML échouée ({exc}).  "
                f"Relancer ``picarones report --results {output}`` "
                "pour réessayer.",
                err=True,
            )


@cli.command("diagnose")
@click.option(
    "--corpus", "-c", required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Dossier contenant les paires image / .gt.txt",
)
@click.option(
    "--engines", "-e", default="tesseract", show_default=True,
    help="Liste de moteurs séparés par des virgules",
)
@click.option(
    "--output", "-o", default="results_diagnose.json", show_default=True,
    type=click.Path(resolve_path=True),
    help="Fichier JSON de sortie",
)
@click.option("--lang", "-l", default="fra", show_default=True,
              help="Code langue Tesseract")
@click.option("--psm", default=6, show_default=True,
              help="Page Segmentation Mode Tesseract")
@click.option("--no-progress", is_flag=True, default=False,
              help="Désactive la barre de progression")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Mode verbeux")
@click.option("--no-html", is_flag=True, default=False,
              help="N'écrit que le JSON, pas le rapport HTML")
@click.option("--html-lang", default="fr", show_default=True,
              type=click.Choice(["fr", "en"]),
              help="Langue du rapport HTML")
def diagnose_cmd(
    corpus: str, engines: str, output: str, lang: str, psm: int,
    no_progress: bool, verbose: bool, no_html: bool, html_lang: str,
) -> None:
    """Workflow diagnostic : bench + leviers d'amélioration + image_predictive.

    Active le profil ``diagnostics`` (chantier 2) qui calcule les
    métriques nécessaires à la vue HTML « Diagnostic approfondi »
    (chantier 3) : leviers, profil d'image, baseline, longitudinal.
    Idéal pour comprendre *pourquoi* un moteur produit ces résultats
    sur ce corpus, pas seulement *quel CER*.

    Phase 4.5 du chantier post-rewrite : génère désormais le HTML
    automatiquement à côté du JSON (``--no-html`` pour skipper).
    """
    _run_workflow(
        corpus=corpus, engines=engines, output=output,
        lang=lang, psm=psm,
        no_progress=no_progress, verbose=verbose,
        profile="diagnostics",
        workflow_label="diagnose",
        generate_html=not no_html,
        html_lang=html_lang,
    )


@cli.command("economics")
@click.option(
    "--corpus", "-c", required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Dossier contenant les paires image / .gt.txt",
)
@click.option(
    "--engines", "-e", default="tesseract", show_default=True,
    help="Liste de moteurs séparés par des virgules",
)
@click.option(
    "--output", "-o", default="results_economics.json", show_default=True,
    type=click.Path(resolve_path=True),
    help="Fichier JSON de sortie",
)
@click.option("--lang", "-l", default="fra", show_default=True,
              help="Code langue Tesseract")
@click.option("--psm", default=6, show_default=True,
              help="Page Segmentation Mode Tesseract")
@click.option("--no-progress", is_flag=True, default=False,
              help="Désactive la barre de progression")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Mode verbeux")
@click.option("--no-html", is_flag=True, default=False,
              help="N'écrit que le JSON, pas le rapport HTML")
@click.option("--html-lang", default="fr", show_default=True,
              type=click.Choice(["fr", "en"]),
              help="Langue du rapport HTML")
def economics_cmd(
    corpus: str, engines: str, output: str, lang: str, psm: int,
    no_progress: bool, verbose: bool, no_html: bool, html_lang: str,
) -> None:
    """Workflow économique : bench + throughput effectif + (cost projection).

    Active le profil ``economics`` (chantier 2) qui se concentre sur
    les métriques de décision budget : pages/h utilisable (intégrant
    la correction humaine HTR-United à 5 s/erreur), coût marginal par
    erreur évitée. La vue HTML « Coût et performance » (chantier 3)
    est désormais générée automatiquement (Phase 4.5 chantier
    post-rewrite — ``--no-html`` pour skipper).
    """
    _run_workflow(
        corpus=corpus, engines=engines, output=output,
        lang=lang, psm=psm,
        no_progress=no_progress, verbose=verbose,
        profile="economics",
        workflow_label="economics",
        generate_html=not no_html,
        html_lang=html_lang,
    )


@cli.command("edition")
@click.option(
    "--corpus", "-c", required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Dossier contenant les paires image / .gt.txt",
)
@click.option(
    "--engines", "-e", default="tesseract", show_default=True,
    help="Liste de moteurs séparés par des virgules",
)
@click.option(
    "--output", "-o", default="results_edition.json", show_default=True,
    type=click.Path(resolve_path=True),
    help="Fichier JSON de sortie",
)
@click.option("--lang", "-l", default="fra", show_default=True,
              help="Code langue Tesseract")
@click.option("--psm", default=6, show_default=True,
              help="Page Segmentation Mode Tesseract")
@click.option("--no-progress", is_flag=True, default=False,
              help="Désactive la barre de progression")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Mode verbeux")
@click.option("--no-html", is_flag=True, default=False,
              help="N'écrit que le JSON, pas le rapport HTML")
@click.option("--html-lang", default="fr", show_default=True,
              type=click.Choice(["fr", "en"]),
              help="Langue du rapport HTML")
def edition_cmd(
    corpus: str, engines: str, output: str, lang: str, psm: int,
    no_progress: bool, verbose: bool, no_html: bool, html_lang: str,
) -> None:
    """Workflow édition critique : bench + métriques philologiques.

    Active le profil ``philological`` (chantier 2) qui inclut les
    modules philologiques (unicode_blocks, abbreviations, MUFI,
    early_modern_typography, modern_archives, roman_numerals) et la
    vue HTML « Taxonomie avancée » (chantier 3) avec comparaison
    miroir leader vs runner-up. Cible : éditeurs de chartes,
    paléographes, archivistes.

    Phase 4.5 du chantier post-rewrite : génère le HTML
    automatiquement (``--no-html`` pour skipper).
    """
    _run_workflow(
        corpus=corpus, engines=engines, output=output,
        lang=lang, psm=psm,
        no_progress=no_progress, verbose=verbose,
        profile="philological",
        workflow_label="edition",
        generate_html=not no_html,
        html_lang=html_lang,
    )


# ---------------------------------------------------------------------------
# picarones compare (Sprint 28)
# ---------------------------------------------------------------------------

@cli.command("compare")
@click.argument("run_a", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument("run_b", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option(
    "--output", "-o",
    default="comparaison.html",
    show_default=True,
    type=click.Path(resolve_path=True),
    help="Fichier HTML de sortie pour le rapport de comparaison",
)
@click.option(
    "--threshold",
    default=0.005,
    show_default=True,
    type=float,
    help="Seuil régression / amélioration (CER absolu, ex. 0.005 = 0,5 pp)",
)
@click.option(
    "--label-a",
    default="A",
    show_default=True,
    help="Étiquette du premier run dans le rapport",
)
@click.option(
    "--label-b",
    default="B",
    show_default=True,
    help="Étiquette du second run dans le rapport",
)
@click.option(
    "--json", "json_only", is_flag=True, default=False,
    help="Sortie JSON sur stdout au lieu du rapport HTML",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Mode verbeux")
def compare_cmd(
    run_a: str,
    run_b: str,
    output: str,
    threshold: float,
    label_a: str,
    label_b: str,
    json_only: bool,
    verbose: bool,
) -> None:
    """Compare deux runs de benchmark JSON et signale les régressions.

    Convention : un Δ CER positif signifie que ``B`` est moins bon que
    ``A``. Un moteur dont |Δ CER| > ``--threshold`` est marqué comme
    régression ou amélioration.

    \b
    Exemples :
        picarones compare run_v1.json run_v2.json -o diff.html
        picarones compare run_v1.json run_v2.json --json
        picarones compare run_v1.json run_v2.json --threshold 0.01 --label-a v1 --label-b v2
    """
    _setup_logging(verbose)

    from picarones.reports.html.comparison import (
        compare_benchmarks,
        detect_regressions,
        render_comparison_html,
    )

    diff = compare_benchmarks(
        run_a, run_b,
        threshold=threshold,
        label_a=label_a,
        label_b=label_b,
    )
    regressions = detect_regressions(diff)

    if json_only:
        click.echo(json.dumps(diff.as_dict(), ensure_ascii=False, indent=2))
        if regressions:
            sys.exit(2)  # exit code 2 → régression détectée (utile en CI)
        return

    out = render_comparison_html(diff, output)
    click.echo(f"Rapport de comparaison : {out}")
    click.echo(f"Moteurs comparés : {len(diff.deltas)}")
    click.echo(f"Régressions     : {len(regressions)}")
    click.echo(f"Améliorations   : {sum(1 for d in diff.deltas if d.is_improvement)}")
    if regressions:
        click.echo("\n— Régressions détectées —")
        for d in regressions:
            click.echo(
                f"  ⚠ {d.engine} : "
                f"{d.cer_a:.3f} → {d.cer_b:.3f} (Δ +{d.delta_cer:.3f})"
            )
        sys.exit(2)
