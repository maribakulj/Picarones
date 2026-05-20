"""Commandes workflows benchmark : run, diagnose, economics, edition, compare.

Sous-module CLI extrait de l'ancien ``picarones/cli.py`` (1519 lignes)
lors du chantier 5 post-Sprint 97.  Les commandes ici s'enregistrent
automatiquement sur le groupe ``cli`` à l'import.

Comportement et signatures inchangés — uniquement de la modularisation.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import click

if TYPE_CHECKING:
    from picarones.evaluation.benchmark_result import BenchmarkResult
    from picarones.evaluation.corpus import Corpus


def _run_orchestrator_for_cli(
    corpus: "Corpus",
    engines: list[Any],
    *,
    views: tuple[str, ...] = ("text_final",),
    profile: str = "standard",
    normalization_profile: Any | None = None,
    char_exclude: Any | None = None,
    partial_dir: str | Path | None = None,
    entity_extractor: str | None = None,
    output_json: str | Path | None = None,
    progress_callback: Callable[[str, int, str], None] | None = None,
) -> "BenchmarkResult":
    """Helper local CLI — pattern 3 étapes vers ``RunOrchestrator``.

    Factorise le pattern ``prepare_preset_args → execute_preset →
    run_result_to_benchmark_result`` pour les 2 commandes CLI qui en
    ont besoin (``run`` et ``_run_workflow``).  Helper interne à la
    couche CLI, pas un service global.

    Phase B3-final corr-A/B/C (mai 2026) — propage les params
    ``views``, ``partial_dir``, ``entity_extractor`` exposés en CLI.

    Le ``BenchmarkResult`` retourné est consommé par les renderers
    historiques (ranking, sortie HTML legacy, etc.).
    """
    from picarones.app.services import (
        RunOrchestrator,
        prepare_preset_args,
        run_result_to_benchmark_result,
    )

    with tempfile.TemporaryDirectory(prefix="picarones_cli_") as ws:
        ws_path = Path(ws)
        run_dir = ws_path / "run"
        args = prepare_preset_args(
            corpus, engines,
            workspace_dir=ws_path / "gt",
            output_dir=run_dir,
            views=views,
            profile=profile,
            normalization_profile=normalization_profile,
            char_exclude=char_exclude,
            partial_dir=partial_dir,
            entity_extractor=entity_extractor,
            output_json=output_json,
        )
        orch_result = RunOrchestrator(run_dir).execute_preset(
            spec=args.spec,
            corpus_spec=args.corpus_spec,
            extracted_dir=args.extracted_dir,
            pipeline_specs=args.pipeline_specs,
            adapter_resolver=args.adapter_resolver,
            adapter_kwargs=args.adapter_kwargs,
            progress_callback=progress_callback,
            # Phase B3-final hotfix (mai 2026) — passer le corpus
            # mémoire évite que ``_persist_legacy_benchmark_json``
            # essaie de reloader depuis ``workspace_dir`` (qui ne
            # contient que les .gt.txt, pas les images).
            corpus_legacy=corpus,
        )
        return run_result_to_benchmark_result(
            orch_result.run_result,
            corpus=corpus, engines=engines,
            char_exclude=char_exclude,
            normalization_profile=normalization_profile,
            profile=profile,
        )

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
@click.option(
    "--normalization-profile",
    "normalization_profile",
    default=None,
    metavar="ID-OR-PATH",
    help=(
        "Profil de normalisation Unicode à appliquer à GT et hypothèses "
        "avant calcul des métriques.  Soit un identifiant builtin "
        "(nfc, caseless, medieval_french, early_modern_english, …), "
        "soit un chemin vers un fichier .yaml/.yml versionné dans git. "
        "Phase 3.3 audit code-quality — voir "
        "``docs/how-to/custom-normalization-profile.md`` pour le schéma."
    ),
)
@click.option(
    "--views",
    default="text_final",
    show_default=True,
    metavar="VIEW1,VIEW2,…",
    help=(
        "Liste des vues d'évaluation à appliquer, séparées par des "
        "virgules.  Valeurs canoniques : 'text_final' (CER/WER sur "
        "texte plat), 'alto_documentary' (validité + métriques ALTO), "
        "'searchability' (rappel fuzzy + séquences numériques).  "
        "Phase B6 — débloque le rapport HTML multi-vues.  Requiert "
        "que les pipelines produisent les artefacts éligibles (ex : "
        "alto_documentary nécessite ALTO_XML, activé par "
        "``--expose-alto`` côté Tesseract)."
    ),
)
@click.option(
    "--expose-alto",
    "expose_alto",
    is_flag=True,
    default=False,
    help=(
        "Active la production native d'ALTO XML par Tesseract via "
        "``pytesseract.image_to_alto_xml``.  Sans ce flag, Tesseract "
        "ne produit que du RAW_TEXT.  Phase B5 — combiné avec "
        "``--views alto_documentary``, débloque les métriques "
        "structurelles ALTO dans le rapport HTML."
    ),
)
@click.option(
    "--char-exclude",
    "char_exclude",
    default=None,
    metavar="CHARS",
    help=(
        "Caractères à exclure du calcul CER/WER (ex : '!?.,;:' pour "
        "ignorer la ponctuation).  Phase B2.5 — propagé à "
        "``compute_metrics``."
    ),
)
@click.option(
    "--partial-dir",
    "partial_dir",
    default=None,
    type=click.Path(resolve_path=True),
    help=(
        "Répertoire pour la reprise sur interruption.  Si fourni, "
        "chaque pipeline est persisté en JSONL et reprenable après "
        "crash.  Phase B2.3."
    ),
)
@click.option(
    "--entity-extractor",
    "entity_extractor",
    default=None,
    metavar="DOTTED_PATH",
    help=(
        "Dotted path Python vers une factory d'extracteur d'entités "
        "(ex : ``mypkg.ner:SpacyExtractor``).  Si fourni, les "
        "métriques NER (precision/recall/F1) sont attachées au "
        "``BenchmarkResult``.  Phase B2.4 — requiert que la GT du "
        "corpus contienne un niveau ENTITIES."
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
    normalization_profile: str | None,
    views: str,
    expose_alto: bool,
    char_exclude: str | None,
    partial_dir: str | None,
    entity_extractor: str | None,
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
    from picarones.interfaces.cli._normalization_arg import (
        resolve_normalization_profile,
    )

    # Résolution du profil de normalisation (Phase 3.3 — identifiant
    # builtin ou chemin YAML versionné).  ``None`` si l'option est
    # absente — comportement historique (aucune normalisation
    # explicite côté caller, défauts de la couche text_metrics).
    try:
        resolved_norm_profile = resolve_normalization_profile(
            normalization_profile,
        )
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Erreur profil normalisation : {exc}", err=True)
        sys.exit(1)

    # Chargement du corpus
    try:
        corp = load_corpus_from_directory(corpus)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Erreur corpus : {exc}", err=True)
        sys.exit(1)

    click.echo(f"Corpus '{corp.name}' — {len(corp)} documents chargés.")
    if resolved_norm_profile is not None:
        click.echo(
            f"Profil normalisation : {resolved_norm_profile.name} "
            f"({len(resolved_norm_profile.diplomatic_table)} règles diplomatiques)"
        )

    # Phase B3-final corr-B (mai 2026) — ``expose_alto`` propagé à
    # Tesseract uniquement.  Les autres adapters ignorent le flag.
    # Instanciation des moteurs
    engine_names = [e.strip() for e in engines.split(",") if e.strip()]
    ocr_engines = []
    for name in engine_names:
        try:
            engine = _engine_from_name(
                name, lang=lang, psm=psm, expose_alto=expose_alto,
            )
            ocr_engines.append(engine)
        except click.BadParameter as exc:
            click.echo(f"Erreur moteur : {exc}", err=True)
            sys.exit(1)

    if not ocr_engines:
        click.echo("Aucun moteur valide spécifié.", err=True)
        sys.exit(1)

    click.echo(f"Moteurs : {', '.join(e.name for e in ocr_engines)}")
    click.echo(f"Profil de métriques : {profile}")
    if expose_alto:
        click.echo("ALTO XML activé pour Tesseract (--expose-alto).")

    # Phase B3-final corr-A : parsing des vues canoniques.
    views_tuple = tuple(v.strip() for v in views.split(",") if v.strip())
    click.echo(f"Vues : {', '.join(views_tuple)}")

    # Lancement du benchmark — pattern 3 étapes via helper local
    # (Phase B3-final, Option 10 : prepare → execute_preset → converter).
    result = _run_orchestrator_for_cli(
        corp, ocr_engines,
        profile=profile,
        normalization_profile=resolved_norm_profile,
        char_exclude=char_exclude,
        partial_dir=partial_dir,
        entity_extractor=entity_extractor,
        views=views_tuple,
        output_json=output,
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

def _b3_final_options(func: Callable) -> Callable:
    """Decorator Click qui ajoute les 5 options B3-final à une commande.

    Phase D1 (audit B3-final, mai 2026) — mutualise les options
    ``--views`` / ``--expose-alto`` / ``--char-exclude`` /
    ``--partial-dir`` / ``--entity-extractor`` ajoutées aux commandes
    ``run`` + ``diagnose`` + ``economics`` + ``edition`` + ``compare``
    + ``robustness``.

    Avant cette correction, seule ``run`` exposait les options ; les
    workflows secondaires ignoraient silencieusement la valeur métier
    B5/B6 (utilisateur invoquant ``picarones diagnose`` ne pouvait
    pas activer AltoView).
    """
    func = click.option(
        "--entity-extractor",
        "entity_extractor",
        default=None,
        metavar="DOTTED_PATH",
        help=(
            "Dotted path Python vers une factory d'extracteur "
            "d'entités (ex : ``mypkg.ner:SpacyExtractor``).  "
            "Phase B2.4."
        ),
    )(func)
    func = click.option(
        "--partial-dir",
        "partial_dir",
        default=None,
        type=click.Path(resolve_path=True),
        help=(
            "Répertoire pour la reprise sur interruption.  "
            "Phase B2.3."
        ),
    )(func)
    func = click.option(
        "--char-exclude",
        "char_exclude",
        default=None,
        metavar="CHARS",
        help=(
            "Caractères à exclure du calcul CER/WER.  Phase B2.5."
        ),
    )(func)
    func = click.option(
        "--expose-alto",
        "expose_alto",
        is_flag=True,
        default=False,
        help=(
            "Active la production native d'ALTO XML par Tesseract.  "
            "Phase B5 — combiné avec ``--views alto_documentary``, "
            "débloque les métriques structurelles ALTO."
        ),
    )(func)
    func = click.option(
        "--views",
        default="text_final",
        show_default=True,
        metavar="VIEW1,VIEW2,…",
        help=(
            "Liste des vues d'évaluation : 'text_final', "
            "'alto_documentary', 'searchability'.  Phase B6."
        ),
    )(func)
    return func


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
    # Phase D1 (audit B3-final, mai 2026) — propagation des options
    # B3-final aux workflows secondaires (diagnose/economics/edition/
    # compare/robustness).  Avant cette correction, ces commandes
    # ignoraient silencieusement les nouvelles features.
    views: tuple[str, ...] = ("text_final",),
    expose_alto: bool = False,
    char_exclude: str | None = None,
    partial_dir: str | None = None,
    entity_extractor: str | None = None,
    normalization_profile: Any | None = None,
) -> None:
    """Implémentation commune des commandes ``run``, ``diagnose``,
    ``economics`` et ``edition``.

    Les 4 commandes partagent le squelette : chargement corpus →
    instanciation moteurs → ``_run_orchestrator_for_cli(profile=...)`` → affichage
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

    Phase D1 (audit B3-final) : tous les paramètres B3-final
    (``views``, ``expose_alto``, ``char_exclude``, ``partial_dir``,
    ``entity_extractor``, ``normalization_profile``) sont désormais
    propagés.  Les wrappers Click (diagnose/economics/etc.) doivent
    exposer les options correspondantes.
    """
    _setup_logging(verbose)

    from picarones.evaluation.corpus import load_corpus_from_directory

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
            engine = _engine_from_name(
                name, lang=lang, psm=psm, expose_alto=expose_alto,
            )
            ocr_engines.append(engine)
        except click.BadParameter as exc:
            click.echo(f"Erreur moteur : {exc}", err=True)
            sys.exit(1)

    if not ocr_engines:
        click.echo("Aucun moteur valide spécifié.", err=True)
        sys.exit(1)

    click.echo(f"Moteurs : {', '.join(e.name for e in ocr_engines)}")
    click.echo(f"Profil de métriques : {profile}")
    if expose_alto:
        click.echo("ALTO XML activé pour Tesseract (--expose-alto).")
    click.echo(f"Vues : {', '.join(views)}")

    result = _run_orchestrator_for_cli(
        corp, ocr_engines,
        views=views,
        profile=profile,
        normalization_profile=normalization_profile,
        char_exclude=char_exclude,
        partial_dir=partial_dir,
        entity_extractor=entity_extractor,
        output_json=output,
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
@_b3_final_options
def diagnose_cmd(
    corpus: str, engines: str, output: str, lang: str, psm: int,
    no_progress: bool, verbose: bool, no_html: bool, html_lang: str,
    views: str, expose_alto: bool, char_exclude: str | None,
    partial_dir: str | None, entity_extractor: str | None,
) -> None:
    """Workflow diagnostic : bench + leviers d'amélioration + image_predictive.

    Active le profil ``diagnostics`` (chantier 2) qui calcule les
    métriques nécessaires à la vue HTML « Diagnostic approfondi »
    (chantier 3) : leviers, profil d'image, baseline, longitudinal.
    Idéal pour comprendre *pourquoi* un moteur produit ces résultats
    sur ce corpus, pas seulement *quel CER*.

    Phase 4.5 du chantier post-rewrite : génère désormais le HTML
    automatiquement à côté du JSON (``--no-html`` pour skipper).

    Phase D1 (audit B3-final) : accepte les options B3-final
    (--views, --expose-alto, --char-exclude, --partial-dir,
    --entity-extractor).
    """
    views_tuple = tuple(v.strip() for v in views.split(",") if v.strip())
    _run_workflow(
        corpus=corpus, engines=engines, output=output,
        lang=lang, psm=psm,
        no_progress=no_progress, verbose=verbose,
        profile="diagnostics",
        workflow_label="diagnose",
        generate_html=not no_html,
        html_lang=html_lang,
        views=views_tuple,
        expose_alto=expose_alto,
        char_exclude=char_exclude,
        partial_dir=partial_dir,
        entity_extractor=entity_extractor,
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
@_b3_final_options
def economics_cmd(
    corpus: str, engines: str, output: str, lang: str, psm: int,
    no_progress: bool, verbose: bool, no_html: bool, html_lang: str,
    views: str, expose_alto: bool, char_exclude: str | None,
    partial_dir: str | None, entity_extractor: str | None,
) -> None:
    """Workflow économique : bench + throughput effectif + (cost projection)."""
    views_tuple = tuple(v.strip() for v in views.split(",") if v.strip())
    _run_workflow(
        corpus=corpus, engines=engines, output=output,
        lang=lang, psm=psm,
        no_progress=no_progress, verbose=verbose,
        profile="economics",
        workflow_label="economics",
        generate_html=not no_html,
        html_lang=html_lang,
        views=views_tuple,
        expose_alto=expose_alto,
        char_exclude=char_exclude,
        partial_dir=partial_dir,
        entity_extractor=entity_extractor,
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
@_b3_final_options
def edition_cmd(
    corpus: str, engines: str, output: str, lang: str, psm: int,
    no_progress: bool, verbose: bool, no_html: bool, html_lang: str,
    views: str, expose_alto: bool, char_exclude: str | None,
    partial_dir: str | None, entity_extractor: str | None,
) -> None:
    """Workflow édition critique : bench + métriques philologiques."""
    views_tuple = tuple(v.strip() for v in views.split(",") if v.strip())
    _run_workflow(
        corpus=corpus, engines=engines, output=output,
        lang=lang, psm=psm,
        no_progress=no_progress, verbose=verbose,
        profile="philological",
        workflow_label="edition",
        generate_html=not no_html,
        html_lang=html_lang,
        views=views_tuple,
        expose_alto=expose_alto,
        char_exclude=char_exclude,
        partial_dir=partial_dir,
        entity_extractor=entity_extractor,
    )


# ---------------------------------------------------------------------------
# picarones compare
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
