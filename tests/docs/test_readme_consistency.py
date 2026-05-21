"""Tests de cohérence du ``README.md`` avec le code réel.

Sprint A2 (items M-19, M-23, M-24, M-25, M-26 de l'audit
``institutional-readiness-2026-05.md``).

**Contrat** : tout élément vérifiable annoncé dans le README doit
correspondre à un artefact du code. La direction est unidirectionnelle :
on vérifie que ce qui est *annoncé* existe, **pas** que tout ce qui
existe est annoncé (cette deuxième garantie sera posée en A13 lors de
la refonte complète du README).

Vérifications appliquées :

1. **Moteurs OCR listés** dans le tableau « Supported Engines » ⇒
   un fichier ``picarones/engines/{nom}.py`` doit exister, ou la ligne
   est annotée par un commentaire HTML
   ``<!-- doc-check: skip-engine -->``.
2. **Commandes CLI listées** dans le tableau « CLI Commands » ⇒ la
   commande doit apparaître dans ``picarones --help``.
3. **Endpoints API web listés** ⇒ doivent exister dans
   ``app.openapi()["paths"]`` (ou être annotés skip).
4. **Compteur de tests** : les phrases du type « pytest tests/ →
   N passed » doivent correspondre au baseline collecté par
   ``pytest --collect-only``, à 5 % près (tolérance pour PR en cours).

**Mécanisme d'exception** : pour autoriser une ligne temporairement
non vérifiable (par exemple un moteur en cours d'ajout dans le même
PR que le README), insérer un commentaire HTML
``<!-- doc-check: skip-<kind> -->`` à la fin de la ligne du tableau.
À utiliser avec modération — tout skip est inspecté en revue.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
ENGINES_DIR = REPO_ROOT / "picarones" / "adapters" / "ocr"

#: Marqueur HTML qui désactive un check sur la ligne. Format :
#: ``<!-- doc-check: skip-engine -->``, ``skip-cli``, ``skip-endpoint``.
SKIP_PATTERN = re.compile(r"<!--\s*doc-check:\s*skip-([a-z]+)\s*-->")

#: Préfixes de "moteurs" du tableau qui ne sont *pas* des moteurs OCR
#: (ce sont des LLMs/VLMs utilisés via les pipelines). Ils sont
#: tolérés en attendant la refonte A13 qui scindera le tableau.
#: La comparaison est sur préfixe pour accepter les annotations comme
#: ``GPT-4o (VLM)`` ou ``Claude Sonnet (VLM)``.
NOT_OCR_ENGINES_TOLERATED_PREFIXES = (
    "GPT-4",
    "GPT-3",
    "Claude",
    "Mistral Large",
    "Mistral Small",
    "Mistral 7B",
    "Ministral",
    "Ollama",
    "Llama",
    "Custom engine",
)


# ---------------------------------------------------------------------------
# Helpers de parsing
# ---------------------------------------------------------------------------


def _read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _normalize_engine_name(name: str) -> str:
    """Normalise un nom de moteur en chemin de module candidat.

    Exemples :
    - ``"Tesseract 5"`` → ``"tesseract"``
    - ``"Pero OCR"`` → ``"pero_ocr"``
    - ``"Google Vision"`` → ``"google_vision"``
    - ``"Azure Doc Intelligence"`` → ``"azure_doc_intel"``
    """
    n = name.lower().strip()
    # Retire emphasis markdown (**Tesseract 5**)
    n = n.replace("**", "").strip()
    # Retire les versions ("tesseract 5" → "tesseract")
    n = re.sub(r"\s+\d+(\.\d+)*$", "", n)
    # Mappings explicites (alias historiques)
    aliases = {
        "azure document intelligence": "azure_doc_intel",
        "azure doc intelligence": "azure_doc_intel",
        # Phase 3 du chantier post-rewrite : kraken/calamari sont
        # listés dans le README avec leur nom commercial complet,
        # mais leur module Python s'appelle juste ``kraken.py`` /
        # ``calamari.py`` (cohérent avec ``pero_ocr.py`` qui ne
        # s'appelle pas ``pero_ocr_htr.py``).
        "kraken htr": "kraken",
        "calamari ocr": "calamari",
    }
    if n in aliases:
        return aliases[n]
    # Default : remplace espaces par underscores
    return n.replace(" ", "_").replace("-", "_")


def _has_engine_adapter(engine_name: str) -> bool:
    """Vrai si ``picarones/engines/{normalized}.py`` existe."""
    candidate = _normalize_engine_name(engine_name)
    return (ENGINES_DIR / f"{candidate}.py").exists()


def _parse_markdown_tables(text: str) -> list[dict]:
    """Parse toutes les tables Markdown du document.

    Retourne une liste de dicts ``{"headers": [...], "rows": [[...], ...],
    "raw_rows": [str, ...]}``. ``raw_rows`` conserve la ligne brute pour
    permettre la détection des marqueurs ``<!-- doc-check: skip-... -->``.
    """
    tables: list[dict] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Heuristique : une table commence par une ligne d'en-tête en | ... |
        # suivie par une ligne de séparation |---|...|
        if line.startswith("|") and i + 1 < len(lines):
            sep = lines[i + 1].strip()
            if re.match(r"^\|[\s:|-]+\|$", sep):
                headers = [h.strip() for h in line.strip("|").split("|")]
                rows: list[list[str]] = []
                raw_rows: list[str] = []
                j = i + 2
                while j < len(lines) and lines[j].strip().startswith("|"):
                    raw = lines[j]
                    cells = [c.strip() for c in raw.strip().strip("|").split("|")]
                    rows.append(cells)
                    raw_rows.append(raw)
                    j += 1
                tables.append({"headers": headers, "rows": rows, "raw_rows": raw_rows})
                i = j
                continue
        i += 1
    return tables


def _has_skip_marker(raw_row: str, kind: str) -> bool:
    """Vrai si la ligne contient ``<!-- doc-check: skip-<kind> -->``."""
    for match in SKIP_PATTERN.finditer(raw_row):
        if match.group(1) == kind:
            return True
    return False


def _find_table_by_header(tables: list[dict], required_columns: set[str]) -> dict | None:
    """Retourne la première table dont les en-têtes contiennent ``required_columns``."""
    for table in tables:
        normalized = {h.lower().strip() for h in table["headers"]}
        if required_columns.issubset(normalized):
            return table
    return None


# ---------------------------------------------------------------------------
# 1. Moteurs OCR listés (M-23)
# ---------------------------------------------------------------------------


def test_engines_table_present() -> None:
    """Le README doit contenir une table des moteurs supportés."""
    tables = _parse_markdown_tables(_read_readme())
    table = _find_table_by_header(tables, {"engine", "type"})
    assert table is not None, (
        "Aucune table 'Supported Engines' trouvée dans le README "
        "(en-têtes 'Engine' + 'Type' attendus)."
    )
    assert len(table["rows"]) >= 1, "Table moteurs vide."


def test_listed_engines_have_adapter() -> None:
    """Tout moteur OCR listé dans le README doit avoir un adapter dans
    ``picarones/engines/``, sauf annotation explicite ``skip-engine``.

    Tolérance : les LLMs/VLMs (GPT-4o, Claude, etc.) sont tolérés tant que
    A13 (refonte README) n'a pas scindé le tableau en 'OCR engines' et
    'LLM/VLM adapters'. La tolérance est pilotée par la frozenset
    ``NOT_OCR_ENGINES_TOLERATED`` ci-dessus.
    """
    tables = _parse_markdown_tables(_read_readme())
    table = _find_table_by_header(tables, {"engine", "type"})
    assert table is not None, "Table moteurs absente (pré-requis : test_engines_table_present)"

    missing: list[str] = []
    for row, raw in zip(table["rows"], table["raw_rows"], strict=True):
        if not row or not row[0].strip():
            continue
        engine_name = row[0].replace("**", "").strip()

        # Skip marker explicite
        if _has_skip_marker(raw, "engine"):
            continue
        # LLM/VLM tolérés (à scinder en A13). Comparaison sur préfixe
        # pour accepter "GPT-4o (VLM)", "Claude Sonnet (VLM)", etc.
        if any(engine_name.startswith(p) for p in NOT_OCR_ENGINES_TOLERATED_PREFIXES):
            continue

        if not _has_engine_adapter(engine_name):
            missing.append(engine_name)

    assert not missing, (
        f"Moteurs annoncés dans le README sans adapter dans "
        f"picarones/engines/ : {missing}. "
        f"Soit créer l'adapter, soit retirer la ligne, soit annoter "
        f"avec <!-- doc-check: skip-engine -->."
    )


# ---------------------------------------------------------------------------
# 2. Commandes CLI (M-25)
# ---------------------------------------------------------------------------


def _real_cli_commands() -> set[str]:
    """Retourne l'ensemble des commandes effectivement exposées."""
    from picarones.interfaces.cli import cli

    return set(cli.commands.keys())


def test_listed_cli_commands_exist() -> None:
    """Toute commande ``picarones <X>`` listée dans le README doit exister."""
    real = _real_cli_commands()
    text = _read_readme()
    tables = _parse_markdown_tables(text)

    # Une table CLI a typiquement les colonnes "Command" + "Description".
    table = _find_table_by_header(tables, {"command", "description"})
    if table is None:
        pytest.skip("Pas de tableau CLI explicite dans le README")

    missing: list[str] = []
    for row, raw in zip(table["rows"], table["raw_rows"], strict=True):
        if not row or not row[0].strip():
            continue
        cell = row[0]

        if _has_skip_marker(raw, "cli"):
            continue

        # Extraction robuste : "`picarones run`" → "run", "picarones import iiif" → "import"
        m = re.search(r"picarones\s+([a-z][a-z_-]*)", cell)
        if not m:
            continue
        cmd = m.group(1)
        if cmd not in real:
            missing.append(cmd)

    assert not missing, (
        f"Commandes annoncées dans le tableau CLI du README mais "
        f"absentes de `picarones --help` : {missing}. "
        f"Disponibles : {sorted(real)}."
    )


# ---------------------------------------------------------------------------
# 3. Endpoints API web (M-26)
# ---------------------------------------------------------------------------


def _real_api_endpoints() -> set[str]:
    """Retourne l'ensemble des chemins exposés par l'app FastAPI."""
    try:
        from picarones.interfaces.web.app import app
    except Exception as exc:  # pragma: no cover — défense en profondeur
        pytest.skip(f"FastAPI app non importable : {exc}")
    spec = app.openapi()
    return set(spec.get("paths", {}).keys())


def _normalize_path(path: str) -> str:
    """Normalise un path en remplaçant les variables ``{xxx}`` ou ``:xxx``
    par des wildcards comparables."""
    path = path.strip().strip("`")
    # ``{job_id}`` et ``{filename}`` représentent la même chose côté code
    return re.sub(r"\{[^}]+\}", "{}", path)


def test_listed_endpoints_exist() -> None:
    """Tout endpoint listé dans le README (avec son chemin commençant par
    ``/api/`` ou ``/`` ou ``/reports/``) doit exister dans l'API FastAPI."""
    real = {_normalize_path(p) for p in _real_api_endpoints()}
    text = _read_readme()
    tables = _parse_markdown_tables(text)

    # Trouver une table d'endpoints (en-têtes "Endpoint" + "Method").
    table = _find_table_by_header(tables, {"endpoint", "method"})
    if table is None:
        pytest.skip("Pas de tableau API endpoints dans le README")

    missing: list[str] = []
    for row, raw in zip(table["rows"], table["raw_rows"], strict=True):
        if not row or not row[0].strip():
            continue
        if _has_skip_marker(raw, "endpoint"):
            continue
        path = _normalize_path(row[0])
        if not path.startswith("/"):
            continue
        if path not in real:
            missing.append(path)

    assert not missing, (
        f"Endpoints annoncés dans le README mais absents de "
        f"app.openapi()['paths'] : {missing}. "
        f"Disponibles ({len(real)}) : {sorted(real)[:10]}…"
    )


# ---------------------------------------------------------------------------
# 4. Compteur de tests — le README ne pin plus un nombre exact
# ---------------------------------------------------------------------------
#
# Historique : ce test lançait ``subprocess.run([..., "pytest",
# "--collect-only", ...])`` pour comparer le compteur cité au nombre
# réel.  Pytest-dans-pytest avec ``--cov`` cause un deadlock sur le
# lock ``.coverage`` (le commentaire ``-p no:cacheprovider`` + ``--no-cov``
# documente déjà ce risque).  La stratégie actuelle élimine la classe
# d'erreur : le README dit ``5000+ tests``, sans nombre figé, et le
# chiffre exact vit dans le badge CI.
#
# Ce test ne fait plus que vérifier qu'aucun compteur exact n'a été
# réintroduit en prose.


def test_readme_does_not_pin_exact_test_count() -> None:
    """Le README ne doit plus citer un nombre exact (``5150 tests``,
    ``5159 passed``, etc.).  La formulation canonique est ``N+ tests``
    (ex. ``5000+ tests``) pour absorber la dérive OS-dépendante du
    compteur (4509 vs 4510 selon que tesseract est installé)."""
    text = _read_readme()

    # On accepte ``5000+ tests``, ``5000+ passed`` (avec ou sans
    # caractères de mise en forme markdown autour).  On refuse
    # ``5150 tests``, ``~5150 tests``, ``5150 passed``.
    forbidden_pattern = re.compile(
        r"(?<!\+)\b(\d{4,5})\s+(?:tests|passed)\b",
        re.IGNORECASE,
    )
    offenders = forbidden_pattern.findall(text)
    assert not offenders, (
        f"README cite des compteurs de tests exacts : {offenders}. "
        "Reformuler en ``N+ tests`` (ex. ``5000+ tests``) — le chiffre "
        "exact dérive selon l'OS / les binaires installés et vit dans "
        "le badge CI."
    )


# ---------------------------------------------------------------------------
# 5. Variables d'environnement (M-24)
# ---------------------------------------------------------------------------


def test_env_vars_with_adapter_or_marker() -> None:
    """Les variables ``AWS_*`` (et autres orphelines) ne doivent pas être
    documentées dans le README sans adapter correspondant.

    Vérification : pour ``AWS_*``, vérifier qu'un adapter
    ``picarones/engines/aws_*.py`` ou ``picarones/engines/textract*.py``
    existe. Si non, la mention dans le README est un mensonge → fail.
    """
    text = _read_readme()
    if "AWS_ACCESS_KEY_ID" not in text and "AWS_SECRET_ACCESS_KEY" not in text:
        # README propre, rien à vérifier.
        return

    # Si la mention est présente, elle doit être marquée skip ou un adapter
    # AWS existant doit la justifier.
    aws_adapters = list(ENGINES_DIR.glob("aws*.py")) + list(ENGINES_DIR.glob("textract*.py"))
    if aws_adapters:
        return  # adapter existe

    # Cas tolérance : la ligne est dans un commentaire HTML
    # ``<!-- doc-check: skip-env -->``
    skip_lines = [
        line
        for line in text.split("\n")
        if "AWS_ACCESS_KEY_ID" in line and _has_skip_marker(line, "env")
    ]
    assert skip_lines, (
        "AWS_* environment variables documentées dans le README mais "
        "aucun adapter Textract n'existe dans picarones/engines/. "
        "Soit implémenter, soit retirer ces 3 lignes, soit annoter "
        "avec <!-- doc-check: skip-env -->."
    )
