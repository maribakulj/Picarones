#!/usr/bin/env python3
"""Génère le SBOM (Software Bill of Materials) au format CycloneDX.

Audience : DSI institutionnelle, conformité EU CRA (Cyber Resilience
Act, exigible à partir de 2027 pour livraisons institutionnelles).

Le SBOM liste tous les paquets Python installés dans l'environnement
courant avec leur version, licence et hash, au format CycloneDX 1.5
JSON.

Usage
-----

::

    pip install cyclonedx-bom
    python scripts/gen_sbom.py [--output sbom.json]

Le SBOM produit doit être attaché à chaque release tag (artefact
GitHub Release) — c'est ce que l'institution archive aux côtés du
wheel pour traçabilité supply-chain.

Pour un SBOM signé Sigstore (SLSA level 3), voir le pipeline GitHub
Actions ``release.yml`` qui invoque ``cosign sign-blob`` sur le
fichier produit.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="sbom.json",
        help="Fichier de sortie CycloneDX JSON (défaut : sbom.json).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "xml"],
        default="json",
        help="Format CycloneDX (défaut : json).",
    )
    args = parser.parse_args()

    if shutil.which("cyclonedx-py") is None:
        sys.stderr.write(
            "[gen_sbom] cyclonedx-py absent.  Installer avec : "
            "pip install cyclonedx-bom\n",
        )
        return 1

    out = Path(args.output).resolve()
    cmd = [
        "cyclonedx-py",
        "environment",
        "--output-format",
        args.format,
        "--outfile",
        str(out),
    ]
    print(f"[gen_sbom] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode
    print(f"[gen_sbom] SBOM écrit dans {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
