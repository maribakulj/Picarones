"""Garde-fous structurels de la documentation (Phase 1 D7).

Ces tests verrouillent la posture documentaire visée après Phase 1 :

- **Racine sobre** : ≤ 9 fichiers .md à la racine du repo.
- **Fichiers actifs courts** : aucun fichier actif > 600 lignes, sauf
  allowlist explicite (CHANGELOG, spécification produit).
- **Archive explicite** : tout fichier sous ``docs/archive/`` doit
  être signalé comme archivé (header « Archived document » dans le
  fichier ou indexation depuis ``docs/archive/README.md``).
- **Nav mkdocs sobre** : la nav ne référence pas directement les
  sous-dossiers d'archive — seule l'entrée
  ``docs/archive/README.md`` est autorisée.
- **Pas de narration sprint dans la doc active** : les fichiers
  actifs ne contiennent pas de phrases narratives type « Sprint
  S2.1 — rééquilibrage » qui décrivent l'histoire du chantier
  plutôt que le contrat actuel.  Les chemins/identifiants
  techniques (ex. ``rewrite-status-s46.md`` dans un lien) restent
  autorisés — c'est de la référence, pas de la narration.

Toute violation = échec CI.  Le but n'est pas d'être maximaliste ;
c'est d'éviter la dérive lente vers le mode « chantier en cours »
qui avait amené le repo à 70+ fichiers .md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────────────────────────────
# Constantes — politique chiffrée
# ─────────────────────────────────────────────────────────────────────

#: Budget de fichiers .md à la racine.  Cible 7 ; 9 toléré pendant
#: la phase de transition (CONTRIBUTING.en.md et SECURITY.en.md sont
#: gardés à la racine par convention GitHub pour la visibilité
#: institutionnelle — sinon le badge "code of conduct" et autres
#: index communautaires ne les trouvent pas).
ROOT_MD_BUDGET = 9

#: Plafond de lignes pour un fichier de doc actif.  Au-delà, le
#: fichier devrait être découpé ou élagué.  L'allowlist explicite
#: ci-dessous contient les exceptions justifiées (spec produit,
#: changelog).
ACTIVE_DOC_LINE_BUDGET = 600

#: Fichiers de doc autorisés à dépasser le budget.  Toute addition
#: doit être justifiée en revue — ce n'est pas une zone tampon, c'est
#: un registre explicite des exceptions.
ACTIVE_DOC_LINE_BUDGET_ALLOWLIST: frozenset[str] = frozenset({
    # Spec produit : grandit naturellement avec les fonctionnalités ;
    # déjà découpée en sections, le tout en un fichier reste utile
    # comme contrat unique consultable d'un coup.
    "docs/reference/specification.md",
    # CHANGELOG actif : période v2.0 et après.  L'historique pré-v2.0
    # vit dans docs/archive/changelog-pre-v2.md.
    "CHANGELOG.md",
})

#: Préfixes des chemins considérés comme "actifs" (non archivés).
ACTIVE_DOC_AREAS: tuple[str, ...] = (
    "docs/",
)

#: Préfixes EXCLUS de la zone active (= archives, hors périmètre des
#: garde-fous structurels).
ARCHIVE_PATH_PREFIXES: tuple[str, ...] = (
    "docs/archive/",
)

#: Header obligatoire dans chaque fichier d'archive (ou alternative :
#: indexation depuis docs/archive/README.md).
ARCHIVE_HEADER_MARKERS: tuple[str, ...] = (
    "Archived document",
    "Archived",
    "**Archived**",
    "Archive historique",
)

#: Patterns narratifs interdits dans la doc active.  Ces motifs
#: décrivent l'histoire du chantier ; ils n'ont pas leur place dans
#: une doc qui décrit le contrat actuel.  Les références techniques
#: (chemins de fichiers archivés, identifiants de releases) sont
#: gérées par une exclusion contextuelle (mot dans un lien
#: markdown ou backtick — laissé passer).
FORBIDDEN_NARRATIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # « Sprint XX » en prose (pas dans un code-span, pas dans un lien)
    (
        "Sprint XX en prose",
        re.compile(
            r"(?<![`/\-])Sprint\s+[A-Z]?\d+(?:\.\d+)*(?![`/\.\-\w])",
        ),
    ),
    # « handover » dans le texte (les liens vers session-handover.md
    # sont OK car le mot est entre slashes ou backticks)
    (
        "narration handover",
        re.compile(
            r"(?<![`/\-])\bhandover\b(?![`/\-])",
            re.IGNORECASE,
        ),
    ),
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _all_md_files() -> list[Path]:
    return sorted(REPO_ROOT.glob("*.md")) + sorted(
        REPO_ROOT.glob("docs/**/*.md")
    )


def _is_archive(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return any(rel.startswith(p) for p in ARCHIVE_PATH_PREFIXES)


def _is_active_doc(path: Path) -> bool:
    """Une doc « active » = sous ``docs/`` mais pas sous archive,
    ou à la racine du repo."""
    rel = path.relative_to(REPO_ROOT).as_posix()
    if rel.startswith("docs/"):
        return not any(rel.startswith(p) for p in ARCHIVE_PATH_PREFIXES)
    # Racine : un seul niveau de profondeur, donc on regarde si ça
    # ressemble à un .md racine.
    return "/" not in rel and rel.endswith(".md")


def _strip_code_and_links(text: str) -> str:
    """Retire les blocs code, les inline-code et les liens markdown
    avant de chercher des motifs narratifs.  Les références techniques
    (``Sprint 78`` dans un code-span ou un nom de fichier) ne sont
    pas de la narration."""
    # Blocs de code triple-backtick
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Inline code single-backtick
    text = re.sub(r"`[^`]*`", "", text)
    # Liens markdown ``[label](url)`` : on garde le label, on retire
    # l'URL où les noms de fichiers archivés vivent.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


# ─────────────────────────────────────────────────────────────────────
# 1. Budget racine
# ─────────────────────────────────────────────────────────────────────


def test_root_md_budget() -> None:
    """La racine du repo doit contenir ≤ ROOT_MD_BUDGET fichiers .md.

    Cibler une racine sobre : un nouveau contributeur voit en quelques
    lignes ce qu'est le projet.  Tout fichier de référence ou
    d'opération vit dans ``docs/``."""
    root_mds = sorted(REPO_ROOT.glob("*.md"))
    rel = [p.name for p in root_mds]
    assert len(root_mds) <= ROOT_MD_BUDGET, (
        f"Racine contient {len(root_mds)} fichiers .md (budget = "
        f"{ROOT_MD_BUDGET}) : {rel}.\n"
        "Déplacer les fichiers non-institutionnels vers docs/."
    )


# ─────────────────────────────────────────────────────────────────────
# 2. Budget de lignes par fichier actif
# ─────────────────────────────────────────────────────────────────────


def test_no_active_doc_exceeds_line_budget() -> None:
    """Aucun fichier de doc active ne doit dépasser
    ACTIVE_DOC_LINE_BUDGET lignes, sauf allowlist explicite."""
    offenders: list[str] = []
    for path in _all_md_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if not _is_active_doc(path):
            continue
        if rel in ACTIVE_DOC_LINE_BUDGET_ALLOWLIST:
            continue
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        if n_lines > ACTIVE_DOC_LINE_BUDGET:
            offenders.append(f"  {rel} : {n_lines} lignes")

    assert not offenders, (
        f"Fichiers actifs au-dessus du budget "
        f"{ACTIVE_DOC_LINE_BUDGET} lignes :\n"
        + "\n".join(offenders)
        + "\n\n→ Soit découper/élaguer, soit ajouter à "
        "ACTIVE_DOC_LINE_BUDGET_ALLOWLIST avec justification."
    )


# ─────────────────────────────────────────────────────────────────────
# 3. Fichiers d'archive : header ou indexation
# ─────────────────────────────────────────────────────────────────────


def _archive_index_text() -> str:
    """Contenu de docs/archive/README.md, qui est l'index des
    archives.  Un fichier archivé sans header peut être « couvert »
    par une mention dans cet index."""
    index = REPO_ROOT / "docs" / "archive" / "README.md"
    if not index.exists():
        return ""
    return index.read_text(encoding="utf-8")


def test_all_archive_files_have_header_or_are_indexed() -> None:
    """Tout fichier sous ``docs/archive/`` doit soit contenir un
    marqueur « Archived » en tête (lecteur direct), soit être
    référencé depuis ``docs/archive/README.md`` (lecteur via index).

    Sans ce test, un fichier déplacé en archive sans bandeau pourrait
    être lu comme de la doc active."""
    archive_dir = REPO_ROOT / "docs" / "archive"
    if not archive_dir.exists():
        pytest.skip("docs/archive/ absent")

    index_text = _archive_index_text()
    offenders: list[str] = []

    for path in sorted(archive_dir.rglob("*.md")):
        if path.name == "README.md":
            continue  # l'index lui-même
        # Lit les 30 premières lignes (= la zone "tête de fichier")
        head = "\n".join(
            path.read_text(encoding="utf-8").splitlines()[:30]
        )
        has_header = any(m in head for m in ARCHIVE_HEADER_MARKERS)
        rel = path.relative_to(REPO_ROOT).as_posix()
        # « indexé » = chemin relatif au répertoire archive cité dans
        # README.md de l'archive
        rel_to_archive = path.relative_to(archive_dir).as_posix()
        is_indexed = (
            rel_to_archive in index_text
            or path.name in index_text
            or rel_to_archive.rsplit("/", 1)[0] + "/" in index_text
        )
        if not (has_header or is_indexed):
            offenders.append(f"  {rel}")

    assert not offenders, (
        f"Fichiers d'archive sans bandeau « Archived » ni "
        f"indexation depuis docs/archive/README.md :\n"
        + "\n".join(offenders)
        + "\n\n→ Ajouter ``> **Archived document.**`` en tête du "
        "fichier OU ajouter une mention explicite dans "
        "docs/archive/README.md."
    )


# ─────────────────────────────────────────────────────────────────────
# 4. mkdocs nav exclut les sous-dossiers d'archive
# ─────────────────────────────────────────────────────────────────────


def _mkdocs_nav() -> list:
    mkdocs = REPO_ROOT / "mkdocs.yml"
    if not mkdocs.exists():
        return []
    # mkdocs utilise des balises spéciales — safe_load suffit pour
    # lire la structure de nav.  En cas de balises personnalisées,
    # on tombe sur un parse_error explicite.
    try:
        doc = yaml.safe_load(mkdocs.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        pytest.fail(f"mkdocs.yml YAML invalide : {e}")
    return doc.get("nav") or []


def _flatten_nav_paths(nav: list, acc: list[str] | None = None) -> list[str]:
    """Aplatit la nav mkdocs en liste de chemins de fichiers
    référencés."""
    if acc is None:
        acc = []
    for entry in nav:
        if isinstance(entry, str):
            acc.append(entry)
        elif isinstance(entry, dict):
            for v in entry.values():
                if isinstance(v, str):
                    acc.append(v)
                elif isinstance(v, list):
                    _flatten_nav_paths(v, acc)
    return acc


def test_mkdocs_nav_excludes_archive_subdirs() -> None:
    """La nav mkdocs ne doit référencer aucun fichier sous
    ``docs/archive/`` SAUF ``docs/archive/README.md`` (l'index).

    Sans cette discipline, les fichiers archivés réapparaissent dans
    la navigation utilisateur — exactement le problème que l'archive
    devait résoudre."""
    nav_paths = _flatten_nav_paths(_mkdocs_nav())
    offenders: list[str] = []
    for p in nav_paths:
        if p.startswith("archive/") and p != "archive/README.md":
            offenders.append(p)
        # Couvre aussi les chemins fully-qualified si présents
        if "docs/archive/" in p:
            if not p.endswith("docs/archive/README.md"):
                offenders.append(p)

    assert not offenders, (
        "mkdocs.yml référence des fichiers d'archive dans la nav "
        f"active : {offenders}.\n"
        "→ Ne garder que ``archive/README.md`` (point d'entrée vers "
        "les archives)."
    )


# ─────────────────────────────────────────────────────────────────────
# 5. Pas de narration sprint dans la doc active
# ─────────────────────────────────────────────────────────────────────


#: Cible atteinte (Phase 2 — convergence narrative, juin 2026).
#: La doc active ne contient plus aucune mention narrative
#: « Sprint XX » : passage de 88 (baseline initial) → 0 par
#: reformulation par intention dans 24 fichiers, sans perte de
#: contenu technique.
#:
#: Ratchet maintenu = 0.  Toute PR qui réintroduirait de la
#: narration sprint en prose dans la doc active fait échouer
#: ``test_no_active_doc_contains_sprint_narrative``.  La double
#: assertion (≥ et ≤ 0) impose qu'on garde le compteur à 0 — pas
#: de tolérance.
ACTIVE_NARRATIVE_BASELINE = 0


def test_no_active_doc_contains_sprint_narrative() -> None:
    """La doc active ne doit pas contenir de narration de chantier
    type « Sprint A2 — refonte X » en prose.

    Les références techniques (chemin de fichier archivé, identifiant
    de release) restent autorisées car elles sont dans des liens ou
    des code-spans, retirés du texte avant le scan."""
    offenders: list[tuple[str, str]] = []
    for path in _all_md_files():
        if not _is_active_doc(path):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        # On exclut le CHANGELOG actif et la spec : ils peuvent citer
        # des sprints dans leur historique structuré (sections
        # taggées) sans que ce soit de la narration insidieuse.
        if rel in ACTIVE_DOC_LINE_BUDGET_ALLOWLIST:
            continue
        # On exclut aussi le fichier de gouvernance documentaire
        # (CLAUDE.md) qui décrit le code et peut référencer des
        # sprints comme repères historiques.
        if rel == "CLAUDE.md":
            continue
        text = _strip_code_and_links(path.read_text(encoding="utf-8"))
        for label, pattern in FORBIDDEN_NARRATIVE_PATTERNS:
            for match in pattern.finditer(text):
                offenders.append((rel, f"{label} : « {match.group(0)} »"))

    # Filtrer les doublons exacts (un même match listé une seule fois)
    offenders = sorted(set(offenders))

    assert len(offenders) <= ACTIVE_NARRATIVE_BASELINE, (
        f"Doc active contient {len(offenders)} mentions narratives "
        f"de chantier (baseline ratchet = {ACTIVE_NARRATIVE_BASELINE}) :\n"
        + "\n".join(f"  {f} : {m}" for f, m in offenders[:30])
        + ("\n  …" if len(offenders) > 30 else "")
        + "\n\n→ Reformuler les phrases qui décrivent l'histoire du "
        "chantier en phrases qui décrivent le contrat actuel.  Ou "
        "baisser ACTIVE_NARRATIVE_BASELINE si on accepte cette dette."
    )


def _count_active_narrative_mentions() -> int:
    """Compte les mentions narratives dans la doc active (utilisé par
    les deux tests de ratchet pour garantir la même métrique)."""
    offenders: list[tuple[str, str]] = []
    for path in _all_md_files():
        if not _is_active_doc(path):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ACTIVE_DOC_LINE_BUDGET_ALLOWLIST or rel == "CLAUDE.md":
            continue
        text = _strip_code_and_links(path.read_text(encoding="utf-8"))
        for label, pattern in FORBIDDEN_NARRATIVE_PATTERNS:
            for match in pattern.finditer(text):
                offenders.append((rel, f"{label} : {match.group(0)}"))
    return len(sorted(set(offenders)))


def test_active_narrative_baseline_decreases() -> None:
    """Ratchet : si le compteur descend en-dessous du baseline, il
    faut mettre à jour la constante dans le même commit pour
    verrouiller le gain.  Pattern classique des tests d'architecture
    (cf. test_doc_paths.py)."""
    actual = _count_active_narrative_mentions()
    assert actual >= ACTIVE_NARRATIVE_BASELINE, (
        f"Excellent : {actual} mentions narratives vs "
        f"baseline {ACTIVE_NARRATIVE_BASELINE}.\n"
        f"Mets à jour ACTIVE_NARRATIVE_BASELINE = {actual} "
        "pour verrouiller le gain."
    )
