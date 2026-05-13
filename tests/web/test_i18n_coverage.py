"""Phase 9 audit code-quality (2026-05) — garde-fous i18n web.

L'UI web a son propre dict de traductions ``T = { fr: {...}, en: {...} }``
dans ``picarones/interfaces/web/static/web-app.js`` (~86 clés par
langue), indépendant du JSON serveur ``picarones/reports/i18n/{fr,en}.json``
(420 clés) qui alimente les rapports HTML.

L'audit avait identifié deux risques de drift :

1. **Templates HTML vs dict JS** : un attribut ``data-i18n="key"``
   dans un template qui n'a pas d'entrée correspondante dans
   ``T.fr`` ou ``T.en`` affiche la clé brute à l'utilisateur (ex.
   « bench_synthesis_title » au lieu de « Synthèse »).

2. **Asymétrie fr/en** : une clé présente dans ``T.fr`` mais pas
   ``T.en`` (ou inversement) casse le toggle de langue.

Ce module verrouille ces deux invariants par parsing AST (Python)
et regex (JS).  La refactorisation profonde (charger ``T`` via
fetch d'un endpoint serveur) est laissée à une PR dédiée — ces
tests assurent au moins l'absence de régression.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_APP_JS = REPO_ROOT / "picarones" / "interfaces" / "web" / "static" / "web-app.js"
TEMPLATES_DIR = REPO_ROOT / "picarones" / "interfaces" / "web" / "templates"

#: ``key: "value"`` ou ``key: 'value'`` (multi-line possible mais on
#: prend juste l'identifiant à gauche du ``:``).
_JS_KEY_RE = re.compile(r"^\s*([a-z_][a-z_0-9]*)\s*:", re.MULTILINE)


def _extract_t_section(text: str, lang: str) -> str:
    """Retourne le contenu textuel de ``T.<lang> = { ... }``.

    Accouchement : on cherche ``  fr: {`` puis on prend tout
    jusqu'au ``  },`` aligné en début de ligne (indentation
    homogène dans web-app.js).
    """
    pattern = re.compile(rf"^\s*{lang}:\s*\{{(.*?)^\s*\}},", re.DOTALL | re.MULTILINE)
    m = pattern.search(text)
    assert m, f"Section T.{lang} introuvable dans web-app.js"
    return m.group(1)


def _js_dict_keys(lang: str) -> set[str]:
    """Clés du dict ``T.<lang>`` dans ``web-app.js``."""
    text = WEB_APP_JS.read_text(encoding="utf-8")
    section = _extract_t_section(text, lang)
    return set(_JS_KEY_RE.findall(section))


def _template_i18n_keys() -> set[str]:
    """Toutes les valeurs d'attribut ``data-i18n="..."`` dans les
    templates HTML."""
    pattern = re.compile(r'data-i18n="([^"]+)"')
    keys: set[str] = set()
    for path in TEMPLATES_DIR.rglob("*.html"):
        for m in pattern.finditer(path.read_text(encoding="utf-8")):
            keys.add(m.group(1))
    return keys


# --------------------------------------------------------------------------
# Invariant 1 : parité fr/en dans le dict JS
# --------------------------------------------------------------------------


def test_js_dict_fr_en_keys_parity() -> None:
    """``T.fr`` et ``T.en`` ont exactement les mêmes clés."""
    fr = _js_dict_keys("fr")
    en = _js_dict_keys("en")

    fr_only = fr - en
    en_only = en - fr

    msg_parts: list[str] = []
    if fr_only:
        msg_parts.append(
            f"Clés présentes en FR mais pas EN ({len(fr_only)}) : "
            + ", ".join(sorted(fr_only))
        )
    if en_only:
        msg_parts.append(
            f"Clés présentes en EN mais pas FR ({len(en_only)}) : "
            + ", ".join(sorted(en_only))
        )
    assert not msg_parts, (
        "Le toggle de langue UI est cassé pour ces clés.\n\n"
        + "\n".join(msg_parts)
        + "\n\nAjouter la traduction manquante dans web-app.js."
    )


# --------------------------------------------------------------------------
# Invariant 2 : ratchet — chaque ``data-i18n`` doit exister dans T.fr/T.en
# --------------------------------------------------------------------------


#: Nombre de clés ``data-i18n`` orphelines toléré (baseline Phase 9).
#: Le dict JS et les templates ont historiquement divergé ; le test
#: ratchet permet la réduction progressive sans bloquer les PRs.
TEMPLATE_KEY_DRIFT_BASELINE = 0


def test_template_i18n_keys_covered_by_js_dict() -> None:
    """Chaque ``data-i18n="key"`` doit avoir une entrée dans ``T.fr``
    (et donc, par parité, dans ``T.en``)."""
    template_keys = _template_i18n_keys()
    fr_keys = _js_dict_keys("fr")

    missing = template_keys - fr_keys
    if len(missing) > TEMPLATE_KEY_DRIFT_BASELINE:
        sample = "\n".join(f"  - {k}" for k in sorted(missing)[:30])
        more = (
            f"\n  ... ({len(missing) - 30} de plus)"
            if len(missing) > 30
            else ""
        )
        raise AssertionError(
            f"{len(missing)} clé(s) ``data-i18n`` orpheline(s) "
            f"(> baseline {TEMPLATE_KEY_DRIFT_BASELINE}) :\n"
            + sample
            + more
            + "\n\nChaque ``data-i18n=\"key\"`` dans un template HTML "
            "doit avoir une entrée correspondante dans le dict ``T.fr`` "
            "/ ``T.en`` de ``web-app.js``.  Sans entrée, l'utilisateur "
            "voit la clé brute (`bench_synthesis_title`) au lieu de la "
            "traduction (« Synthèse »)."
        )


# --------------------------------------------------------------------------
# Invariant 3 : cohérence interne du dict JS (clés citées via t("...")
# --------------------------------------------------------------------------


def _js_t_calls() -> set[str]:
    """Toutes les chaînes citées via ``t("key")`` ou ``t('key')``
    dans ``web-app.js``."""
    text = WEB_APP_JS.read_text(encoding="utf-8")
    pattern = re.compile(r"""\bt\(\s*["']([a-z_][a-z_0-9]*)["']\s*[,)]""")
    return set(pattern.findall(text))


def test_t_calls_covered_by_js_dict() -> None:
    """Chaque ``t("key")`` dans le JS doit avoir une entrée dans T.fr."""
    cited = _js_t_calls()
    fr_keys = _js_dict_keys("fr")

    missing = cited - fr_keys
    assert not missing, (
        f"Appels ``t(\"key\")`` dont la clé manque dans T.fr "
        f"({len(missing)}) :\n"
        + "\n".join(f"  - {k}" for k in sorted(missing)[:20])
        + "\n\nLa fonction ``t()`` retourne la clé brute en fallback, "
        "donc l'UI affiche un identifiant technique au lieu d'un texte."
    )


# --------------------------------------------------------------------------
# Invariant 4 : labels.get() dans renderers ↔ JSON serveur (rapports HTML)
# --------------------------------------------------------------------------


REPORTS_DIR = REPO_ROOT / "picarones" / "reports"
SERVER_I18N_FR = REPO_ROOT / "picarones" / "reports" / "i18n" / "fr.json"
SERVER_I18N_EN = REPO_ROOT / "picarones" / "reports" / "i18n" / "en.json"


def _server_i18n_keys(path: Path) -> set[str]:
    import json
    return set(json.loads(path.read_text(encoding="utf-8")).keys())


def _renderer_labels_get_keys() -> set[str]:
    """Toutes les clés citées par ``labels.get("key", ...)`` dans les
    renderers Python (couche 7)."""
    pattern = re.compile(r"""labels\.get\(\s*["']([a-z_][a-z_0-9.]*)["']""")
    keys: set[str] = set()
    for path in REPORTS_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        keys |= set(pattern.findall(path.read_text(encoding="utf-8")))
    return keys


def test_server_i18n_fr_en_parity() -> None:
    """``reports/i18n/fr.json`` et ``en.json`` ont exactement les
    mêmes clés (le toggle de langue rapport ne doit jamais voir
    une asymétrie)."""
    fr = _server_i18n_keys(SERVER_I18N_FR)
    en = _server_i18n_keys(SERVER_I18N_EN)
    only_fr = fr - en
    only_en = en - fr
    assert not (only_fr or only_en), (
        "Asymétrie reports/i18n/{fr,en}.json :\n"
        + (f"  FR sans EN : {sorted(only_fr)}\n" if only_fr else "")
        + (f"  EN sans FR : {sorted(only_en)}\n" if only_en else "")
    )


#: Baseline du nombre de ``labels.get()`` dont la clé manque dans
#: ``reports/i18n/fr.json``.  Phase 9.2 audit code-quality (2026-05) :
#: 22 clés orphelines portées vers le JSON serveur ; baseline 0 pour
#: bloquer toute nouvelle régression.
RENDERER_LABEL_DRIFT_BASELINE = 0


def test_renderer_labels_covered_by_server_i18n() -> None:
    """Chaque ``labels.get("key", ...)`` dans les renderers doit avoir
    une entrée dans ``reports/i18n/fr.json`` (et par parité, dans
    ``en.json``).

    Sans cette entrée, le fallback du ``.get(key, default)`` retourne
    la chaîne par défaut **en français**, même quand l'utilisateur a
    sélectionné l'anglais — bug majeur de bilinguisme rapport.
    """
    cited = _renderer_labels_get_keys()
    fr = _server_i18n_keys(SERVER_I18N_FR)
    missing = cited - fr

    if len(missing) > RENDERER_LABEL_DRIFT_BASELINE:
        sample = "\n".join(f"  - {k}" for k in sorted(missing)[:30])
        more = (
            f"\n  ... ({len(missing) - 30} de plus)"
            if len(missing) > 30
            else ""
        )
        raise AssertionError(
            f"{len(missing)} clé(s) ``labels.get()`` orpheline(s) "
            f"(> baseline {RENDERER_LABEL_DRIFT_BASELINE}) :\n"
            + sample + more
            + "\n\nAjouter la clé dans ``picarones/reports/i18n/fr.json`` "
            "ET ``en.json`` (parité obligatoire).  Sans cela, le rapport "
            "EN affiche le ``default`` français du ``.get(key, default)``."
        )
