# picarones.spec — Configuration PyInstaller
#
# Génère un exécutable standalone Picarones pour Linux, macOS et Windows.
# L'exécutable embarque Python et toutes les dépendances — aucune installation requise.
#
# Usage :
#   pip install pyinstaller
#   pyinstaller picarones.spec --noconfirm
#
# Sortie :
#   dist/picarones/picarones          (Linux/macOS)
#   dist/picarones/picarones.exe      (Windows)
#
# Pour un seul fichier (démarrage plus lent) :
#   pyinstaller picarones.spec --noconfirm --onefile

import sys
from pathlib import Path

# Chemin racine du projet
ROOT = Path(spec_file).parent  # noqa: F821 (spec_file est défini par PyInstaller)

# ──────────────────────────────────────────────────────────────────
# Analyse des dépendances
# ──────────────────────────────────────────────────────────────────
a = Analysis(
    # Point d'entrée : le script CLI principal
    scripts=[str(ROOT / "picarones" / "__main__.py")],

    # Chemins de recherche des modules
    pathex=[str(ROOT)],

    # Dépendances binaires supplémentaires (DLLs, .so)
    binaries=[],

    # Fichiers de données à embarquer
    datas=[
        # Données de configuration
        (str(ROOT / "picarones"), "picarones"),
        # Prompts LLM (si présents)
        # (str(ROOT / "prompts"), "prompts"),
    ],

    # Imports cachés (non détectés automatiquement par PyInstaller)
    hiddenimports=[
        # CLI
        "picarones.cli",
        "picarones.core.corpus",
        "picarones.core.metrics",
        "picarones.core.results",
        "picarones.core.runner",
        "picarones.core.normalization",
        "picarones.core.statistics",
        "picarones.core.confusion",
        "picarones.core.char_scores",
        "picarones.core.taxonomy",
        "picarones.core.structure",
        "picarones.core.image_quality",
        "picarones.core.difficulty",
        "picarones.core.history",
        "picarones.core.robustness",
        "picarones.engines.base",
        "picarones.engines.tesseract",
        "picarones.engines.pero_ocr",
        "picarones.engines.mistral_ocr",
        "picarones.engines.google_vision",
        "picarones.engines.azure_doc_intel",
        "picarones.llm.base",
        "picarones.llm.openai_adapter",
        "picarones.llm.anthropic_adapter",
        "picarones.llm.mistral_adapter",
        "picarones.llm.ollama_adapter",
        "picarones.importers.iiif",
        "picarones.importers.gallica",
        "picarones.importers.escriptorium",
        "picarones.importers.huggingface",
        "picarones.importers.htr_united",
        "picarones.pipelines.base",
        "picarones.pipelines.over_normalization",
        "picarones.report.generator",
        "picarones.report.diff_utils",
        "picarones.fixtures",
        # Dépendances tiers
        "click",
        "jiwer",
        "PIL",
        "PIL.Image",
        "PIL.ImageFilter",
        "PIL.ImageOps",
        "yaml",
        "tqdm",
        "numpy",
        "pytesseract",
        # SQLite (stdlib, mais parfois manquant)
        "sqlite3",
        # Encodage
        "unicodedata",
    ],

    # Fichiers à exclure pour réduire la taille
    excludes=[
        "tkinter",
        "matplotlib",
        "scipy",
        "sklearn",
        "pandas",
        "IPython",
        "jupyter",
        "notebook",
    ],

    # Options de collection
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

# ──────────────────────────────────────────────────────────────────
# Archive PYZ (modules Python compilés)
# ──────────────────────────────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data)  # noqa: F821

# ──────────────────────────────────────────────────────────────────
# Exécutable principal
# ──────────────────────────────────────────────────────────────────
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="picarones",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,            # Compression UPX si disponible
    console=True,        # Mode console (pas de fenêtre graphique)
    disable_windowed_traceback=False,
    argv_emulation=False,
    # Icône (optionnelle)
    # icon=str(ROOT / "assets" / "picarones.ico"),
)

# ──────────────────────────────────────────────────────────────────
# Collection finale (dossier dist/picarones/)
# ──────────────────────────────────────────────────────────────────
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="picarones",
)
