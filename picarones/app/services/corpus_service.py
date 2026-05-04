"""``CorpusService`` — upload ZIP sandboxé + détection des paires image/GT.

Sprint A14-S20 du rewrite ciblé.

Le service applicatif qui prend en entrée un blob ZIP (uploadé par
le web ou la CLI) et produit un ``CorpusSpec`` immédiatement
consommable par le ``BenchmarkService`` (S17), avec :

- **Extraction sandboxée** dans un sous-dossier d'un
  ``WorkspaceManager`` (S19) — refus du path traversal, des symlinks,
  et des zip bombs.
- **Détection des paires** image / GT par convention de nommage,
  alignée sur l'historique (Sprint 32) :

  ::

      mon_doc.png
      mon_doc.gt.txt
      mon_doc.gt.alto.xml
      mon_doc.gt.page.xml
      mon_doc.gt.entities.json
      mon_doc.gt.reading_order.json

  Toutes les GT partageant le **même stem** que l'image sont rattachées
  au même ``DocumentRef``.

- **Filtrage silencieux** des artefacts macOS / Windows (``__MACOSX/``,
  ``._*``, ``.DS_Store``, ``Thumbs.db``) — bruit standard d'un ZIP
  produit par un poste de travail patrimonial.

- **Rapport** ``CorpusImportReport`` qui agrège warnings (image
  sans GT, GT orpheline) et compte les entrées sautées — l'utilisateur
  doit pouvoir vérifier visuellement que son corpus a été interprété
  correctement.

Anti-sur-ingénierie
-------------------
- Pas d'OCR à l'import.  Le service ne lit pas les contenus, il
  organise.
- Pas de validation de schéma ALTO/PAGE à l'import (c'est lourd).
  Les fichiers sont juste catalogués ; la validation se fait à la
  demande par les projecteurs/loaders.
- Pas de quotas par utilisateur ou rate-limiting (responsabilité
  du caller web/CLI ; les paramètres ``max_*`` du constructeur sont
  des plafonds défensifs absolus).
- Pas d'autodétection de format image (PNG vs JPEG vs TIFF) — on
  reconnaît par extension.  Si un attaquant met un EXE en ``.png``,
  Pillow protégera plus tard (S21+ pour la web).
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from picarones.app.services.path_security import (
    WorkspaceManager,
    safe_report_name,
)
from picarones.domain.artifacts import ArtifactType
from picarones.domain.corpus import CorpusSpec
from picarones.domain.documents import DocumentRef, GroundTruthRef
from picarones.domain.errors import PicaronesError

logger = logging.getLogger(__name__)


class CorpusImportError(PicaronesError):
    """Levée quand l'import ZIP échoue de manière irrécupérable.

    Cas typiques :
    - Archive corrompue / non-ZIP.
    - Path traversal détecté.
    - Symlink détecté.
    - Plafond de taille / nombre d'entrées dépassé (zip bomb).
    """


# ──────────────────────────────────────────────────────────────────────
# Conventions de nommage GT (alignées sur picarones/core/corpus.py
# Sprint 32, mais exprimées en ``ArtifactType`` pour le rewrite).
# ──────────────────────────────────────────────────────────────────────

#: Suffixes de GT reconnus, dans l'ordre du plus spécifique au moins
#: spécifique (``.gt.alto.xml`` doit être testé AVANT ``.gt.txt`` qui
#: est une sous-chaîne moins discriminante).
_GT_SUFFIX_TO_TYPE: tuple[tuple[str, ArtifactType], ...] = (
    (".gt.alto.xml", ArtifactType.ALTO_XML),
    (".gt.page.xml", ArtifactType.PAGE_XML),
    (".gt.entities.json", ArtifactType.ENTITIES),
    (".gt.reading_order.json", ArtifactType.READING_ORDER),
    (".gt.txt", ArtifactType.RAW_TEXT),
)

#: Extensions image reconnues (case-insensitive).  L'absence de ``.gt.``
#: dans le chemin est requise pour distinguer ``foo.png`` (image) d'un
#: éventuel ``foo.gt.alto.xml`` (qui ne match pas ces extensions, mais
#: par défense).
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp",
})

#: Patterns à ignorer silencieusement (artefacts OS).
_OS_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)__MACOSX(/|$)"),
    re.compile(r"(^|/)\._[^/]*$"),
    re.compile(r"(^|/)\.DS_Store$"),
    re.compile(r"(^|/)Thumbs\.db$", re.IGNORECASE),
)


# ──────────────────────────────────────────────────────────────────────
# Rapport d'import
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorpusImportReport:
    """Résultat lisible humainement d'un ``import_zip``.

    Attributs
    ---------
    spec:
        Le ``CorpusSpec`` construit, prêt à être passé au
        ``BenchmarkService``.
    extracted_dir:
        Chemin filesystem absolu du sous-dossier où le ZIP a été
        extrait.  Vit sous le ``WorkspaceManager.root``.
    n_documents:
        Nombre de documents avec au moins une image (= longueur de
        ``spec.documents``).
    n_images_without_gt:
        Nombre d'images trouvées sans GT.  Ces documents sont quand
        même inclus dans le corpus (l'utilisateur peut juste vouloir
        OCRiser, pas évaluer).
    n_gt_without_image:
        Nombre de GT orphelines (stem qui n'a pas d'image
        correspondante).  Loggées en warning et non rattachées —
        ne participent pas au corpus.
    n_skipped_noise:
        Nombre d'entrées sautées silencieusement (artefacts OS).
    warnings:
        Messages humainement lisibles à présenter au caller (web
        affiche dans une bannière, CLI affiche en stderr).
    skipped_paths:
        Liste des chemins (relatifs au root du ZIP) qui ont été
        sautés ou non rattachés — utile au debug d'un import qui
        a perdu des fichiers.
    """

    spec: CorpusSpec
    extracted_dir: Path
    n_documents: int
    n_images_without_gt: int
    n_gt_without_image: int
    n_skipped_noise: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    skipped_paths: tuple[str, ...] = field(default_factory=tuple)


# ──────────────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────────────


class CorpusService:
    """Service d'import et d'analyse de structure d'un corpus.

    Parameters
    ----------
    workspace:
        ``WorkspaceManager`` dans lequel extraire le ZIP.  Le service
        crée un sous-dossier par import — plusieurs imports peuvent
        coexister dans un même workspace.
    max_zip_size_bytes:
        Plafond sur la **taille du blob ZIP** lui-même (avant
        extraction).  Défaut 100 Mo.  Le caller (web layer) doit
        idéalement vérifier ça aussi en amont via
        ``Content-Length``.
    max_entry_count:
        Plafond sur le nombre d'entrées dans le ZIP (anti-bombe par
        nombre).  Défaut 5000.
    max_uncompressed_bytes:
        Plafond sur la taille totale **décompressée** (anti-bombe
        par expansion).  Défaut 500 Mo.
    """

    def __init__(
        self,
        workspace: WorkspaceManager,
        *,
        max_zip_size_bytes: int = 100 * 1024 * 1024,
        max_entry_count: int = 5000,
        max_uncompressed_bytes: int = 500 * 1024 * 1024,
    ) -> None:
        self._workspace = workspace
        self._max_zip_size = max_zip_size_bytes
        self._max_entries = max_entry_count
        self._max_uncompressed = max_uncompressed_bytes

    # ──────────────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────────────

    def import_zip(
        self,
        zip_bytes: bytes,
        *,
        corpus_name: str,
        metadata: dict[str, str] | None = None,
    ) -> CorpusImportReport:
        """Extrait un ZIP et construit le ``CorpusSpec`` correspondant.

        Étapes :

        1. Validation des plafonds (taille blob, nb entrées,
           taille décompressée prévisible si dispo).
        2. Validation de chaque entrée (refus traversal, symlinks).
        3. Extraction sécurisée dans un sous-dossier dédié.
        4. Catalogage : détection images + GT + appariement par stem.
        5. Construction du ``CorpusSpec``.

        Le ``corpus_name`` est nettoyé via :func:`safe_report_name`
        (le caller peut passer un nom utilisateur sans pré-validation).
        """
        if len(zip_bytes) > self._max_zip_size:
            raise CorpusImportError(
                f"ZIP trop volumineux : {len(zip_bytes)} octets > "
                f"plafond {self._max_zip_size}.",
            )

        safe_name = safe_report_name(corpus_name, max_length=64)
        # Sous-dossier d'extraction unique pour cet import — permet
        # plusieurs imports sans collision.
        extract_dir = self._workspace.subpath(f"corpus_{safe_name}")
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        except zipfile.BadZipFile as exc:
            raise CorpusImportError(f"Archive ZIP invalide : {exc}") from exc

        with zf:
            self._validate_archive(zf)
            extracted_files, n_noise = self._extract_safely(zf, extract_dir)

        spec, warnings, n_orphan_gt, n_no_gt, skipped_paths = (
            self._build_corpus_spec(
                extracted_files=extracted_files,
                corpus_name=safe_name,
                extract_dir=extract_dir,
                metadata=metadata or {},
            )
        )

        return CorpusImportReport(
            spec=spec,
            extracted_dir=extract_dir,
            n_documents=len(spec.documents),
            n_images_without_gt=n_no_gt,
            n_gt_without_image=n_orphan_gt,
            n_skipped_noise=n_noise,
            warnings=tuple(warnings),
            skipped_paths=tuple(skipped_paths),
        )

    # ──────────────────────────────────────────────────────────────────
    # Étape 1 : validation globale de l'archive
    # ──────────────────────────────────────────────────────────────────

    def _validate_archive(self, zf: zipfile.ZipFile) -> None:
        """Vérifie les plafonds globaux (entrées, taille décompressée)."""
        infos = zf.infolist()
        if len(infos) > self._max_entries:
            raise CorpusImportError(
                f"ZIP contient trop d'entrées : {len(infos)} > "
                f"plafond {self._max_entries} (zip bomb suspectée).",
            )
        total_uncompressed = sum(info.file_size for info in infos)
        if total_uncompressed > self._max_uncompressed:
            raise CorpusImportError(
                f"ZIP décompressé trop volumineux : {total_uncompressed} "
                f"octets > plafond {self._max_uncompressed} (zip bomb "
                "suspectée).",
            )

    # ──────────────────────────────────────────────────────────────────
    # Étape 2 + 3 : extraction sécurisée
    # ──────────────────────────────────────────────────────────────────

    def _extract_safely(
        self,
        zf: zipfile.ZipFile,
        extract_dir: Path,
    ) -> tuple[list[tuple[str, Path]], int]:
        """Extrait chaque fichier en validant son chemin cible.

        Returns
        -------
        tuple[list[tuple[str, Path]], int]
            ``(extracted_files, n_skipped_noise)`` — liste des paires
            ``(relative_in_zip, absolute_on_disk)`` des fichiers
            réellement extraits, et compte des entrées sautées car
            artefact OS.
        """
        out: list[tuple[str, Path]] = []
        n_noise = 0
        for info in zf.infolist():
            arc_name = info.filename
            # Saut des répertoires nus.
            if arc_name.endswith("/"):
                continue
            # Saut des artefacts OS (silencieux par design).
            if _is_os_noise(arc_name):
                n_noise += 1
                continue
            # Refus des chemins absolus, traversals, octets nuls.
            self._reject_unsafe_arcname(arc_name)
            # Refus des symlinks (mode UNIX bit S_IFLNK = 0xA000).
            unix_mode = (info.external_attr >> 16) & 0xF000
            if unix_mode == 0xA000:
                raise CorpusImportError(
                    f"Symlink dans le ZIP refusé : {arc_name!r}.",
                )

            target = (extract_dir / arc_name).resolve()
            # Garde-fou final : le path résolu doit rester sous extract_dir.
            try:
                target.relative_to(extract_dir.resolve())
            except ValueError as exc:
                raise CorpusImportError(
                    f"Entrée ZIP {arc_name!r} sort du dossier "
                    f"d'extraction après résolution.",
                ) from exc

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(64 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            out.append((arc_name, target))
        return out, n_noise

    @staticmethod
    def _reject_unsafe_arcname(arc_name: str) -> None:
        if not arc_name:
            raise CorpusImportError("Entrée ZIP au nom vide.")
        if "\x00" in arc_name:
            raise CorpusImportError(
                f"Entrée ZIP avec octet nul dans le nom : {arc_name!r}.",
            )
        # Refus chemin absolu (Unix ``/`` ou Windows ``C:\``).
        if arc_name.startswith("/") or arc_name.startswith("\\"):
            raise CorpusImportError(
                f"Chemin absolu interdit dans le ZIP : {arc_name!r}.",
            )
        if len(arc_name) >= 3 and arc_name[1] == ":" and arc_name[2] in ("/", "\\"):
            raise CorpusImportError(
                f"Chemin absolu Windows interdit dans le ZIP : "
                f"{arc_name!r}.",
            )
        # Refus des traversals (``..`` comme composant).
        parts = arc_name.replace("\\", "/").split("/")
        if any(p == ".." for p in parts):
            raise CorpusImportError(
                f"Traversal détecté dans le ZIP : {arc_name!r}.",
            )

    # ──────────────────────────────────────────────────────────────────
    # Étape 4 + 5 : catalogage et construction de la spec
    # ──────────────────────────────────────────────────────────────────

    def _build_corpus_spec(
        self,
        *,
        extracted_files: list[tuple[str, Path]],
        corpus_name: str,
        extract_dir: Path,
        metadata: dict[str, str],
    ) -> tuple[CorpusSpec, list[str], int, int, list[str]]:
        """Catalogue images et GT puis construit le ``CorpusSpec``.

        Returns
        -------
        tuple[CorpusSpec, warnings, n_orphan_gt, n_no_gt, skipped_paths]
        """
        images_by_stem: dict[str, Path] = {}
        gts_by_stem: dict[str, dict[ArtifactType, Path]] = {}
        skipped_paths: list[str] = []
        warnings_list: list[str] = []

        for arc_name, abs_path in extracted_files:
            # Conserver l'arc_name comme « chemin source » pour le doc
            # id (relatif, lisible).  L'image_uri / gt.uri sera l'absolu.
            kind = _classify(arc_name)
            if kind is None:
                skipped_paths.append(arc_name)
                continue
            if isinstance(kind, ArtifactType):
                # GT
                stem = _strip_gt_suffix(arc_name, kind)
                if stem is None:
                    skipped_paths.append(arc_name)
                    continue
                gts_by_stem.setdefault(stem, {})[kind] = abs_path
            else:
                # Image
                stem = _strip_image_extension(arc_name)
                if stem in images_by_stem:
                    warnings_list.append(
                        f"Plusieurs images partagent le stem "
                        f"{stem!r} — première gardée, "
                        f"{arc_name!r} ignorée.",
                    )
                    skipped_paths.append(arc_name)
                    continue
                images_by_stem[stem] = abs_path

        # Appariement.
        documents: list[DocumentRef] = []
        n_no_gt = 0
        for stem in sorted(images_by_stem):
            image_path = images_by_stem[stem]
            gts = gts_by_stem.pop(stem, {})
            if not gts:
                n_no_gt += 1
                warnings_list.append(
                    f"Image {stem!r} sans GT — incluse mais non "
                    "évaluable.",
                )
            ground_truths = tuple(
                GroundTruthRef(type=art_type, uri=str(path))
                for art_type, path in sorted(
                    gts.items(), key=lambda kv: kv[0].value,
                )
            )
            doc_id = _doc_id_from_stem(stem)
            documents.append(
                DocumentRef(
                    id=doc_id,
                    image_uri=str(image_path),
                    ground_truths=ground_truths,
                ),
            )

        # GT orphelines (stems sans image correspondante).
        n_orphan_gt = 0
        for stem, gts in gts_by_stem.items():
            for art_type in gts:
                n_orphan_gt += 1
                warnings_list.append(
                    f"GT orpheline (pas d'image pour stem "
                    f"{stem!r}) : niveau {art_type.value!r}.",
                )

        spec = CorpusSpec(
            name=corpus_name,
            documents=tuple(documents),
            metadata=metadata,
        )
        return spec, warnings_list, n_orphan_gt, n_no_gt, skipped_paths


# ──────────────────────────────────────────────────────────────────────
# Helpers de classification
# ──────────────────────────────────────────────────────────────────────


def _is_os_noise(arc_name: str) -> bool:
    return any(p.search(arc_name) for p in _OS_NOISE_PATTERNS)


def _classify(arc_name: str) -> ArtifactType | str | None:
    """Classifie une entrée en ``ArtifactType`` (GT) ou ``"image"``.

    Returns
    -------
    ArtifactType si GT reconnue, "image" si image reconnue,
    None si non classifiable.
    """
    lower = arc_name.lower()
    for suffix, art_type in _GT_SUFFIX_TO_TYPE:
        if lower.endswith(suffix):
            return art_type
    # On distingue les images : extension reconnue ET pas de ``.gt.``.
    # (``foo.gt.png`` est conceptuellement pas une convention valide,
    # mais on défend.)
    if ".gt." in lower:
        return None
    for ext in _IMAGE_EXTENSIONS:
        if lower.endswith(ext):
            return "image"
    return None


def _strip_gt_suffix(arc_name: str, art_type: ArtifactType) -> str | None:
    """Retire le suffixe GT et retourne le stem.  ``None`` si non match."""
    lower = arc_name.lower()
    for suffix, t in _GT_SUFFIX_TO_TYPE:
        if t is art_type and lower.endswith(suffix):
            return arc_name[: len(arc_name) - len(suffix)]
    return None


def _strip_image_extension(arc_name: str) -> str:
    """Retire l'extension image (case-insensitive)."""
    lower = arc_name.lower()
    for ext in _IMAGE_EXTENSIONS:
        if lower.endswith(ext):
            return arc_name[: len(arc_name) - len(ext)]
    return arc_name


_DOC_ID_INVALID_RE = re.compile(r"[^A-Za-z0-9_.\-/]")


def _doc_id_from_stem(stem: str) -> str:
    """Convertit un stem (chemin relatif) en ``DocumentRef.id`` valide.

    Le validateur de ``DocumentRef`` exige
    ``[A-Za-z0-9_.\\-/]+`` — on remplace tout caractère hors de cet
    alphabet par ``_`` (typique : espaces, accents, parenthèses dans
    des noms BnF).
    """
    cleaned = _DOC_ID_INVALID_RE.sub("_", stem)
    if not cleaned:
        return "doc"
    return cleaned


__all__ = [
    "CorpusImportError",
    "CorpusImportReport",
    "CorpusService",
]
