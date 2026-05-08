"""Marqueurs typographiques et abréviations des archives modernes
(XIXᵉ-XXᵉ siècles) — Sprint 59.

Sprint 59 — Étape 3 / extension philologique du plan d'évolution
2026.

Pourquoi ce module
------------------
Les Sprints 56-57 sont orientés **médiéval scribal** (Capelli, MUFI),
le Sprint 58 cible l'**imprimé ancien** XVIᵉ-XVIIIᵉ.  Ce sprint étend
la couverture aux **archives modernes** (XIXᵉ-XXᵉ), période où la
typographie historique a disparu mais où subsistent des conventions
d'abréviation propres aux corpus institutionnels (état civil,
recensements, presse, monographies, archives militaires).

Distinction avec les modules précédents
---------------------------------------
- ``mufi.py`` (Sprint 57) : caractères médiévaux scribaux.
- ``abbreviations.py`` (Sprint 56) : signes scribaux médiévaux.
- ``early_modern_typography.py`` (Sprint 58) : marqueurs
  typographiques imprimé ancien (ﬁ ſ ı &…).
- ``modern_archives.py`` (ce module) : abréviations et conventions
  de l'archive moderne XIXᵉ-XXᵉ.

Catégories
----------
1. ``civility_titles`` : Mme, M., Mlle, Mgr, Dr, Pr, Me, R.P., S.M.,
   S.A.R., S.E., S.S.
2. ``ordinals`` : 1ᵉʳ, 1ʳᵉ, 2ᵉ, 2ᵈ, Vᵉ (avec exposants Unicode)
3. ``currency`` : ₶ (livre tournois), ₣ ƒ (franc), £, l. s. d.
   (livre/sol/denier d'Ancien Régime)
4. ``administrative`` : arr., dép., cant., com., reg., prov.
5. ``civil_status`` : °, †, ✶, ⚭, ép., vve
6. ``typographic_punctuation`` : « », –, —, …, ’
7. ``latin_abbr_modern`` : e.g., i.e., etc., cf., ibid., op. cit.,
   ad lib.
8. ``bibliographic`` : vol., t., p., pp., n°, fasc., éd., ms.,
   r°, v°
9. ``address`` : bd, av., r., pl., imp., fbg

Sortie
------
``compute_modern_archives_metrics(ref, hyp)`` retourne deux scores
par catégorie (pattern Sprint 56) :

- ``strict_score`` : forme abrégée préservée telle quelle ;
- ``expansion_score`` : forme abrégée OU forme développée présente.

Le **ratio strict/expansion** par catégorie permet au chercheur de
juger lui-même la convention adoptée par chaque moteur, sans
classification automatique imposée par le module.

Stratégie de découpage
----------------------
Cohérente avec NER (38), Flesch (52), Reading order F1 (53),
Layout F1 (54), Bloc Unicode (55), Abréviations (56), MUFI (57),
Imprimé ancien (58) : couche de calcul pure d'abord ; câblage
runner et HTML dans des sprints dédiés.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from picarones.evaluation.metric_registry import register_metric
from picarones.domain.artifacts import ArtifactType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Tables d'abréviations par catégorie
# ──────────────────────────────────────────────────────────────────────────
#
# Format : tuple ``(marker, expansions, regex_strict_pattern_or_None)``
# où :
#   - ``marker``                 : forme abrégée canonique (str)
#   - ``expansions``             : tuple de formes développées
#                                   acceptées (insensible à la casse)
#   - ``regex_strict_pattern``   : pattern Python regex pour la
#                                   détection dans la GT.  ``None``
#                                   = on dérive automatiquement
#                                   ``\b<marker_escaped>\b`` (avec
#                                   garde-fou sur les abréviations
#                                   contenant un point).
#
# Détection : pour les abréviations contenant un ``.`` (« M. »),
# on n'utilise pas ``\b`` standard car « M.\b » match dans
# « M.A. » (le ``.`` étant non-mot, ``\b`` est satisfait).  On
# exige donc explicitement une frontière espace/début/fin/
# ponctuation après le point.

CIVILITY_TITLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Mme",      ("Madame",)),
    ("Mlle",     ("Mademoiselle",)),
    ("Mgr",      ("Monseigneur",)),
    ("Dr",       ("Docteur",)),
    ("Pr",       ("Professeur",)),
    ("Me",       ("Maître",)),
    ("M.",       ("Monsieur",)),
    ("R.P.",     ("Révérend Père",)),
    ("S.M.",     ("Sa Majesté",)),
    ("S.A.R.",   ("Son Altesse Royale",)),
    ("S.E.",     ("Son Excellence",)),
    ("S.S.",     ("Sa Sainteté",)),
)

# Ordinaux : la forme **strict** porte l'exposant Unicode
# (1ᵉʳ U+1D49 U+02B3, 1ʳᵉ, 2ᵈ, 2ᵉ, 3ᵉ…) ; la forme **expansion**
# accepte la version plate (« 1er », « 1re », « 2nd ») ou la forme
# textuelle (« premier », « première »).
#
# On définit chaque ordinal explicitement (1-12 + Vᵉ pour les
# numéraux romains de siècle).  Au-delà, l'exposant ᵉ seul couvre
# les usages courants (3ᵉ, 4ᵉ, 5ᵉ, 6ᵉ, 7ᵉ, 8ᵉ, 9ᵉ, 10ᵉ).

ORDINALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("1ᵉʳ",      ("1er", "premier")),
    ("1ʳᵉ",      ("1re", "première", "premiere")),
    ("2ᵈ",       ("2d", "second")),
    ("2ᵈᵉ",      ("2de", "seconde")),
    ("2ᵉ",       ("2e", "deuxième", "deuxieme")),
    ("3ᵉ",       ("3e", "troisième", "troisieme")),
    ("Iᵉʳ",      ("Ier", "premier")),
    ("Vᵉ",       ("Ve", "cinquième", "cinquieme")),
    ("XIᵉ",      ("XIe", "onzième", "onzieme")),
    ("XIIᵉ",     ("XIIe", "douzième", "douzieme")),
    ("XVIᵉ",     ("XVIe", "seizième", "seizieme")),
    ("XVIIᵉ",    ("XVIIe", "dix-septième", "dix-septieme")),
    ("XVIIIᵉ",   ("XVIIIe", "dix-huitième", "dix-huitieme")),
    ("XIXᵉ",     ("XIXe", "dix-neuvième", "dix-neuvieme")),
    ("XXᵉ",      ("XXe", "vingtième", "vingtieme")),
)

CURRENCY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("₶",        ("livre tournois", "livres tournois")),
    ("₣",        ("franc", "francs")),
    ("ƒ",        ("florin", "florins")),
    ("£",        ("livre", "livres", "pound", "pounds")),
    ("l.",       ("livre", "livres")),
    ("s.",       ("sol", "sols", "sou", "sous")),
    ("d.",       ("denier", "deniers")),
)

ADMINISTRATIVE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("arr.",     ("arrondissement",)),
    ("dép.",     ("département", "departement")),
    ("cant.",    ("canton",)),
    ("com.",     ("commune",)),
    ("reg.",     ("régiment", "regiment")),
    ("prov.",    ("province",)),
)

# État civil : signes typographiques (° = né, † = mort, ⚭ = marié)
# et abréviations textuelles (ép. = épouse/époux, vve = veuve).
CIVIL_STATUS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("°",        ("né", "née")),
    ("†",        ("mort", "morte", "décédé", "décédée")),
    ("✶",        ("naissance",)),
    ("⚭",        ("marié", "mariée", "épousa", "epousa")),
    ("ép.",      ("épouse", "époux", "epouse", "epoux")),
    ("vve",      ("veuve",)),
)

# Ponctuation typographique : ces marqueurs sont préservés en
# diplomatique et remplacés par leur équivalent ASCII en
# modernisant.  L'expansion n'est pas une « expansion » au sens
# linguistique mais un substitut typographique.
TYPOGRAPHIC_PUNCTUATION: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("«",        ('"',)),
    ("»",        ('"',)),
    ("—",        ("-", "--")),
    ("–",        ("-",)),
    ("…",        ("...",)),
    ("’",        ("'",)),
    ("‘",        ("'",)),
)

LATIN_ABBR_MODERN: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("e.g.",     ("for example", "par exemple", "exempli gratia")),
    ("i.e.",     ("c'est-à-dire", "id est", "that is")),
    ("etc.",     ("et cetera", "et caetera")),
    ("cf.",      ("confer", "voir")),
    ("ibid.",    ("ibidem",)),
    ("op. cit.", ("opere citato", "opus citatum")),
    ("ad lib.",  ("ad libitum",)),
    ("N.B.",     ("nota bene",)),
)

BIBLIOGRAPHIC: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vol.",     ("volume",)),
    ("t.",       ("tome",)),
    ("p.",       ("page",)),
    ("pp.",      ("pages",)),
    ("n°",       ("numéro", "numero", "no")),
    ("fasc.",    ("fascicule",)),
    ("éd.",      ("édition", "edition")),
    ("ms.",      ("manuscrit",)),
    ("f.",       ("folio",)),
    ("r°",       ("recto",)),
    ("v°",       ("verso",)),
)

ADDRESS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bd",       ("boulevard",)),
    ("av.",      ("avenue",)),
    ("r.",       ("rue",)),
    ("pl.",      ("place",)),
    ("imp.",     ("impasse",)),
    ("fbg",      ("faubourg",)),
)


# ──────────────────────────────────────────────────────────────────────────
# Indexation par catégorie
# ──────────────────────────────────────────────────────────────────────────

_CATEGORIES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "civility_titles":          CIVILITY_TITLES,
    "ordinals":                 ORDINALS,
    "currency":                 CURRENCY,
    "administrative":           ADMINISTRATIVE,
    "civil_status":             CIVIL_STATUS,
    "typographic_punctuation":  TYPOGRAPHIC_PUNCTUATION,
    "latin_abbr_modern":        LATIN_ABBR_MODERN,
    "bibliographic":            BIBLIOGRAPHIC,
    "address":                  ADDRESS,
}

# Liste plate de tous les marqueurs avec leur catégorie.  Triée par
# longueur décroissante pour que la détection préfère le marqueur
# le plus long quand plusieurs préfixes matchent (ex. « S.A.R. »
# avant « S.A. ").
_ALL_MARKERS: list[tuple[str, tuple[str, ...], str]] = sorted(
    [
        (marker, expansions, category)
        for category, entries in _CATEGORIES.items()
        for marker, expansions in entries
    ],
    key=lambda triple: -len(triple[0]),
)


# ──────────────────────────────────────────────────────────────────────────
# Compilation des patterns regex
# ──────────────────────────────────────────────────────────────────────────
#
# Pour chaque marqueur, on compile un pattern qui exige une
# frontière de mot adaptée :
#
# - Marqueur alphabétique seul (« Mme », « bd ») → ``\b<marker>\b``
#   (le ``\b`` Python gère correctement les bords).
# - Marqueur contenant un point (« M. », « S.A.R. », « arr. »,
#   « r° », « n° ») → frontière espace/début/fin/ponctuation
#   explicite (le ``.`` final étant non-mot, ``\b`` standard
#   matcherait dans « arr.acher »).
# - Marqueur contenant un caractère non ASCII (exposant, monnaie,
#   guillemet, croix d'état civil) → match littéral, pas de
#   frontière de mot car ``\b`` ne fonctionne pas sur les
#   caractères non-mot Unicode.
#
# La frontière de droite après un point exige soit la fin de
# chaîne, soit un blanc, soit une ponctuation usuelle (« , ; : ! ? )
# … » »).

_TRAILING_BOUNDARY = r"(?=$|[\s,;:!?\)\]\»\"\'\n\r\t…])"
_LEADING_BOUNDARY = r"(?:^|(?<=[\s,;:!?\(\[\«\"\'\n\r\t]))"


def _is_alphanumeric_only(text: str) -> bool:
    """Vrai si tous les caractères sont alphanumériques ASCII."""
    return all(c.isascii() and c.isalnum() for c in text)


def _compile_pattern(marker: str) -> re.Pattern[str]:
    """Compile le pattern regex pour la détection d'un marqueur
    dans la GT et l'hypothèse.

    La logique de frontière de mot dépend de la composition du
    marqueur (cf. commentaire principal).
    """
    escaped = re.escape(marker)
    if "." in marker:
        # Frontière explicite après le point final.
        return re.compile(_LEADING_BOUNDARY + escaped + _TRAILING_BOUNDARY)
    if _is_alphanumeric_only(marker):
        return re.compile(r"\b" + escaped + r"\b")
    # Marqueurs Unicode (exposants, monnaies, guillemets, ponctuation
    # typographique, croix) : match littéral, pas de \b.
    return re.compile(escaped)


# Cache des patterns compilés : (marker, category) → pattern.
_PATTERNS: dict[tuple[str, str], re.Pattern[str]] = {
    (marker, category): _compile_pattern(marker)
    for marker, _expansions, category in _ALL_MARKERS
}

# Patterns d'expansion (insensibles à la casse, frontière de mot
# si la forme développée est purement alphabétique).
_EXPANSION_PATTERNS: dict[str, list[re.Pattern[str]]] = {}
for marker, expansions, _category in _ALL_MARKERS:
    compiled: list[re.Pattern[str]] = []
    for exp in expansions:
        escaped = re.escape(exp)
        if exp and _is_alphanumeric_only(exp):
            compiled.append(re.compile(r"\b" + escaped + r"\b", re.IGNORECASE))
        else:
            compiled.append(re.compile(escaped, re.IGNORECASE))
    _EXPANSION_PATTERNS[marker] = compiled


# ──────────────────────────────────────────────────────────────────────────
# API publique : catégorisation + détection
# ──────────────────────────────────────────────────────────────────────────


def get_category(marker: str) -> Optional[str]:
    """Retourne la catégorie d'un marqueur ou ``None`` si inconnu.

    La comparaison est exacte (sensible à la casse, aux exposants
    Unicode et aux points).
    """
    if not marker:
        return None
    for category, entries in _CATEGORIES.items():
        for known, _expansions in entries:
            if known == marker:
                return category
    return None


def get_expansions(marker: str) -> tuple[str, ...]:
    """Retourne les formes développées connues pour un marqueur,
    ou un tuple vide si inconnu."""
    if not marker:
        return ()
    for _category, entries in _CATEGORIES.items():
        for known, expansions in entries:
            if known == marker:
                return expansions
    return ()


def detect_modern_markers(
    text: Optional[str],
) -> list[tuple[int, str, str]]:
    """Retourne les marqueurs trouvés dans ``text``.

    Forme de sortie : ``[(index, marker, category), ...]`` triée
    par index croissant.  Si plusieurs marqueurs se chevauchent, le
    plus long gagne (ex. « S.A.R. » plutôt que « S. " puis « A.R. »).

    Tolérance casse
    ---------------
    Les marqueurs alphabétiques courts (« Mme », « Dr », « bd »)
    sont matchés tels quels (sensibilité à la casse) — on n'élargit
    pas car « me » en minuscule n'est pas une abréviation de
    « Maître ».
    """
    if not text:
        return []
    # Collecte tous les matches de tous les marqueurs.
    candidates: list[tuple[int, int, str, str]] = []  # start, end, marker, cat
    for marker, _expansions, category in _ALL_MARKERS:
        pattern = _PATTERNS[(marker, category)]
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), marker, category))
    # Tri par (start, -length) pour appliquer une stratégie greedy
    # « plus long gagne » à chaque position.
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    chosen: list[tuple[int, str, str]] = []
    last_end = -1
    for start, end, marker, category in candidates:
        if start < last_end:
            continue
        chosen.append((start, marker, category))
        last_end = end
    return chosen


# ──────────────────────────────────────────────────────────────────────────
# Calcul des scores strict / expansion
# ──────────────────────────────────────────────────────────────────────────


def _hyp_contains_marker(
    hypothesis: str, marker: str, category: str,
) -> bool:
    """Vrai si le marqueur est présent (au moins une occurrence) dans
    l'hypothèse, avec la même règle de frontière qu'en GT."""
    pattern = _PATTERNS[(marker, category)]
    return pattern.search(hypothesis) is not None


def _hyp_contains_expansion(hypothesis: str, marker: str) -> bool:
    """Vrai si une forme développée connue du marqueur est présente
    dans l'hypothèse (insensible à la casse)."""
    for pattern in _EXPANSION_PATTERNS.get(marker, ()):
        if pattern.search(hypothesis) is not None:
            return True
    return False


def compute_modern_archives_metrics(
    reference: Optional[str],
    hypothesis: Optional[str],
) -> dict:
    """Calcule la préservation des marqueurs d'archives modernes.

    Pour chaque catégorie : retourne le ``strict_score`` (forme
    abrégée préservée) et l'``expansion_score`` (abrégée OU
    développée présente).  Le ratio des deux donne au chercheur la
    convention adoptée (diplomatique / modernisante / mixte) sans
    qu'aucune classification ne soit imposée.

    Returns
    -------
    dict
        ``{
            "n_markers_reference": int,
            "n_strict_preserved": int,
            "n_expansion_preserved": int,
            "global_strict_score": float,
            "global_expansion_score": float,
            "per_category": {
                category: {
                    "n_total": int,
                    "n_strict_preserved": int,
                    "n_expansion_preserved": int,
                    "strict_score": float,
                    "expansion_score": float,
                }
            },
            "missed_markers": [
                {"index": int, "marker": str, "category": str,
                 "expansion_preserved": bool}
            ],
        }``

    Cas dégénérés
    -------------
    - GT vide ou sans marqueur → tous les compteurs à 0, scores à
      ``0.0``, ``per_category == {}``.
    - GT non vide avec marqueurs + hyp vide → tous les scores à
      ``0.0``, tous les marqueurs dans ``missed_markers``.
    """
    ref = reference or ""
    hyp = hypothesis or ""

    detected = detect_modern_markers(ref)
    n_total = len(detected)
    if n_total == 0:
        return {
            "n_markers_reference": 0,
            "n_strict_preserved": 0,
            "n_expansion_preserved": 0,
            "global_strict_score": 0.0,
            "global_expansion_score": 0.0,
            "per_category": {},
            "missed_markers": [],
        }

    per_cat_total: dict[str, int] = {}
    per_cat_strict: dict[str, int] = {}
    per_cat_expansion: dict[str, int] = {}
    n_strict = 0
    n_expansion = 0
    missed: list[dict] = []

    for index, marker, category in detected:
        per_cat_total[category] = per_cat_total.get(category, 0) + 1
        strict_ok = _hyp_contains_marker(hyp, marker, category)
        # Convention identique à Sprint 56 : si l'abrégé est
        # préservé, c'est aussi un succès pour expansion (l'OCR n'a
        # pas perdu l'information).
        expansion_ok = strict_ok or _hyp_contains_expansion(hyp, marker)
        if strict_ok:
            per_cat_strict[category] = per_cat_strict.get(category, 0) + 1
            n_strict += 1
        if expansion_ok:
            per_cat_expansion[category] = per_cat_expansion.get(category, 0) + 1
            n_expansion += 1
        if not strict_ok:
            missed.append({
                "index": index,
                "marker": marker,
                "category": category,
                "expansion_preserved": expansion_ok,
            })

    per_category = {
        cat: {
            "n_total": per_cat_total[cat],
            "n_strict_preserved": per_cat_strict.get(cat, 0),
            "n_expansion_preserved": per_cat_expansion.get(cat, 0),
            "strict_score": (
                per_cat_strict.get(cat, 0) / per_cat_total[cat]
                if per_cat_total[cat] > 0 else 0.0
            ),
            "expansion_score": (
                per_cat_expansion.get(cat, 0) / per_cat_total[cat]
                if per_cat_total[cat] > 0 else 0.0
            ),
        }
        for cat in sorted(per_cat_total)
    }

    return {
        "n_markers_reference": n_total,
        "n_strict_preserved": n_strict,
        "n_expansion_preserved": n_expansion,
        "global_strict_score": n_strict / n_total,
        "global_expansion_score": n_expansion / n_total,
        "per_category": per_category,
        "missed_markers": missed,
    }


def modern_archives_strict_score(
    reference: Optional[str], hypothesis: Optional[str],
) -> float:
    """Raccourci : taux global de préservation **stricte** des
    marqueurs d'archives modernes ∈ [0, 1]."""
    return compute_modern_archives_metrics(
        reference, hypothesis,
    )["global_strict_score"]


def modern_archives_expansion_score(
    reference: Optional[str], hypothesis: Optional[str],
) -> float:
    """Raccourci : taux global de préservation **étendue** (abrégée
    OU développée) des marqueurs d'archives modernes ∈ [0, 1]."""
    return compute_modern_archives_metrics(
        reference, hypothesis,
    )["global_expansion_score"]


# ──────────────────────────────────────────────────────────────────────────
# Enregistrement dans le registre typé (Sprint 34)
# ──────────────────────────────────────────────────────────────────────────


@register_metric(
    name="modern_archives_strict_score",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Taux de préservation stricte des abréviations et marqueurs "
        "typographiques caractéristiques des archives modernes "
        "(XIXᵉ-XXᵉ) : titres de civilité, ordinaux, monnaies, "
        "abréviations administratives, état civil, ponctuation "
        "typographique, abréviations latines, abréviations "
        "bibliographiques, abréviations d'adresse. Forme abrégée "
        "préservée telle quelle (signal d'édition diplomatique)."
    ),
    higher_is_better=True,
    tags={"text", "modern_archives", "philology", "abbreviations"},
)
def _registered_strict(reference: str, hypothesis: str) -> float:
    return modern_archives_strict_score(reference, hypothesis)


@register_metric(
    name="modern_archives_expansion_score",
    input_types=(ArtifactType.TEXT, ArtifactType.TEXT),
    description=(
        "Taux de préservation étendue (forme abrégée OU forme "
        "développée présente) des marqueurs d'archives modernes "
        "XIXᵉ-XXᵉ. Le ratio strict/expansion par catégorie "
        "permet au chercheur de juger lui-même la convention "
        "éditoriale adoptée."
    ),
    higher_is_better=True,
    tags={"text", "modern_archives", "philology", "abbreviations"},
)
def _registered_expansion(reference: str, hypothesis: str) -> float:
    return modern_archives_expansion_score(reference, hypothesis)


__all__ = [
    "CIVILITY_TITLES",
    "ORDINALS",
    "CURRENCY",
    "ADMINISTRATIVE",
    "CIVIL_STATUS",
    "TYPOGRAPHIC_PUNCTUATION",
    "LATIN_ABBR_MODERN",
    "BIBLIOGRAPHIC",
    "ADDRESS",
    "compute_modern_archives_metrics",
    "detect_modern_markers",
    "get_category",
    "get_expansions",
    "modern_archives_strict_score",
    "modern_archives_expansion_score",
]
