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

# Sprint A9 (m-15) — utilitaire PyInstaller pour auto-détecter les
# imports d'un package entier. Remplace la liste hiddenimports manuelle
# qui dérivait silencieusement à chaque refactor.
from PyInstaller.utils.hooks import collect_submodules  # noqa: F401

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

    # Sprint A9 (m-15) — auto-détection des hiddenimports.
    #
    # Avant Sprint A9, la liste était maintenue manuellement et
    # dérivait : elle référençait des modules qui ont migré dans
    # ``measurements/`` ou ``extras/`` au moment du refactor des
    # Cercles 1/2/3 (Sprint 33).  Bug latent : la PyInstaller build
    # produisait un exécutable qui ratait silencieusement à
    # l'``import`` de ces modules.
    #
    # ``collect_submodules`` parcourt tout le sous-arbre du package
    # à la construction et inclut tout ce qui s'importe.  Plus rien
    # à maintenir à la main quand on ajoute un sous-module.
    #
    # Liste explicite des dépendances tierces conservée car certaines
    # (PIL.ImageFilter, jiwer) ne sont pas trouvées par ``collect_submodules``
    # de leur propre fait (importées paresseusement).
    hiddenimports=(
        collect_submodules("picarones")  # noqa: F821 — défini par PyInstaller
        + [
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
            "defusedxml",
            "defusedxml.ElementTree",
            "sqlite3",
            "unicodedata",
            # Sprint A1 — type-checking et tests embarqués au build dev
            # uniquement.  En build release pur, retirer.
        ]
    ),

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
