"""Réduction optionnelle d'une image base64 avant un appel multimodal.

Pourquoi
--------
Les scans patrimoniaux sont souvent en très haute résolution
(plusieurs mégapixels).  Les modèles de vision facturent / limitent
l'entrée image en **tokens**, proportionnellement à la résolution :
un appel ``image+texte`` non redimensionné coûte 1 à 2 ordres de
grandeur plus de tokens que le prompt texte, et sature la limite
tokens-par-minute du fournisseur (cause racine observée des HTTP 429
qui ne touchaient que les appels ``image=oui``).

Politique
---------
- **Désactivé par défaut** : sans configuration explicite
  (``max_image_dimension`` absent / ``0``), l'image est envoyée
  **inchangée** — aucun changement méthodologique, résultats
  existants non affectés.
- Activé (``max_image_dimension > 0``) : le plus grand côté est
  ramené à cette valeur (ratio préservé).  Sortie ré-encodée en PNG
  pour rester cohérent avec le préfixe ``data:image/png;base64,``
  qu'utilisent les adapters VLM.
- **Robustesse avant tout** : toute erreur (Pillow absent, décodage,
  format exotique) ⇒ image renvoyée telle quelle.  Un
  redimensionnement ne doit jamais faire échouer un appel LLM.
"""

from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger(__name__)


def downscale_b64_image(image_b64: str, max_edge: int) -> str:
    """Réduit l'image pour que ``max(largeur, hauteur) <= max_edge``.

    Retourne le base64 ré-encodé (sans préfixe ``data:``), ou
    **l'entrée inchangée** si ``max_edge <= 0``, image déjà assez
    petite, Pillow indisponible, ou toute erreur.
    """
    if max_edge <= 0 or not image_b64:
        return image_b64
    try:
        from PIL import Image
    except ImportError:
        logger.warning(
            "[image] Pillow indisponible — downscale ignoré, "
            "image envoyée pleine résolution",
        )
        return image_b64
    try:
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        longest = max(w, h)
        if longest <= max_edge:
            return image_b64
        scale = max_edge / float(longest)
        resized = img.resize(
            (max(1, round(w * scale)), max(1, round(h * scale)))
        )
        # PNG accepte L/LA/P/RGB/RGBA ; on neutralise seulement les
        # modes exotiques (CMYK, I, F…) en RGB.
        if resized.mode not in ("L", "LA", "P", "RGB", "RGBA"):
            resized = resized.convert("RGB")
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        out = base64.b64encode(buf.getvalue()).decode("ascii")
        logger.info(
            "[image] downscale %dx%d → %dx%d (max_edge=%d) : "
            "b64 %d → %d octets",
            w, h, resized.size[0], resized.size[1],
            max_edge, len(image_b64), len(out),
        )
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[image] downscale ignoré (%s) — image envoyée telle quelle",
            exc,
        )
        return image_b64


__all__ = ["downscale_b64_image"]
