"""Mode public : restrictions fonctionnelles (extrait de ``security.py``).

dégonflage du god-module ``security``.  Cluster
*sans état*, sans dépendance interne (os only).  Réimporté par
``security`` (API publique préservée).
"""

from __future__ import annotations

import os
from typing import Iterable


#: Identifiants de moteurs cloud dont les clefs API sont mutualisées côté
#: serveur. En mode public on refuse toute requête qui les invoque.
CLOUD_OCR_ENGINES: frozenset[str] = frozenset({
    "mistral_ocr",
    "google_vision",
    "azure_doc_intel",
})

#: Identifiants de fournisseurs LLM facturés à la clef serveur.
CLOUD_LLM_PROVIDERS: frozenset[str] = frozenset({
    "openai",
    "anthropic",
    "mistral",
    "ollama",  # local mais quand même mutualisé
})


def is_public_mode() -> bool:
    """Vrai si l'instance tourne en mode public (HuggingFace Space, etc.)."""
    return os.environ.get("PICARONES_PUBLIC_MODE", "").strip() in ("1", "true", "yes")


def assert_engines_allowed(engines: Iterable[str]) -> None:
    """Lève ``PermissionError`` si la liste contient un moteur cloud bloqué.

    Réponse à utiliser côté FastAPI : ``HTTPException(403, str(exc))``.
    """
    if not is_public_mode():
        return
    banned = [e for e in engines if e in CLOUD_OCR_ENGINES]
    if banned:
        raise PermissionError(
            "Mode public actif (PICARONES_PUBLIC_MODE=1) — les moteurs OCR "
            f"cloud sont désactivés : {', '.join(banned)}. Faites tourner "
            "Picarones localement ou désactivez le mode public."
        )


def assert_llm_provider_allowed(llm_provider: str) -> None:
    """Lève ``PermissionError`` si un LLM mutualisé est sollicité en mode public."""
    if not is_public_mode():
        return
    if llm_provider and llm_provider.strip() in CLOUD_LLM_PROVIDERS:
        raise PermissionError(
            "Mode public actif — les pipelines OCR+LLM sont désactivés "
            f"(provider '{llm_provider}'). En production institutionnelle, "
            "exiger une clef API utilisateur via l'en-tête X-User-API-Key."
        )


def entity_extractor_allowlist() -> frozenset[str]:
    """Dotted paths d'extracteurs NER explicitement autorisés côté web.

    Lue depuis ``PICARONES_ENTITY_EXTRACTOR_ALLOWLIST`` (séparateur
    virgule).  Vide par défaut : le champ ``entity_extractor`` du
    payload web déclenche un ``importlib.import_module`` *puis un
    appel* du symbole résolu (cf. ``run_orchestrator._resolve_entity_
    extractor``).  C'est un gadget d'exécution — il doit être opt-in
    explicite, jamais ouvert par défaut sur une instance partagée.
    """
    raw = os.environ.get("PICARONES_ENTITY_EXTRACTOR_ALLOWLIST", "")
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def assert_entity_extractor_allowed(dotted_path: str) -> None:
    """Lève ``PermissionError`` si le dotted path NER n'est pas autorisé.

    Politique fail-closed **stricte côté web, tous modes confondus**
    (audit prod P0.2) :

    - Vide ⇒ aucun NER attaché, rien à valider.
    - Allowlist vide ⇒ refusé **quel que soit le mode**.  Le web est
      une surface réseau : importer dynamiquement + appeler un symbole
      utilisateur est trop puissant, même hors mode public, même
      derrière SSO.  L'ancienne tolérance « hors mode public » était
      un trou (un déploiement non-public mais exposé restait ouvert).
      La CLI, elle, appelle ``_resolve_entity_extractor`` directement
      sans passer par ce garde-fou : elle reste libre.
    - Allowlist définie ⇒ le dotted path doit en faire partie.
    """
    dotted_path = (dotted_path or "").strip()
    if not dotted_path:
        return
    allowlist = entity_extractor_allowlist()
    if not allowlist:
        raise PermissionError(
            "entity_extractor est désactivé côté web (import dynamique "
            "+ appel d'un symbole = surface réseau trop puissante). "
            "Définir PICARONES_ENTITY_EXTRACTOR_ALLOWLIST (séparateur "
            "virgule) pour autoriser des dotted paths précis, ou "
            "utiliser la CLI."
        )
    if dotted_path not in allowlist:
        raise PermissionError(
            f"entity_extractor {dotted_path!r} hors allowlist. "
            "Ajouter le dotted path à PICARONES_ENTITY_EXTRACTOR_"
            "ALLOWLIST (séparateur virgule) pour l'autoriser."
        )


__all__ = [
    "CLOUD_LLM_PROVIDERS",
    "CLOUD_OCR_ENGINES",
    "assert_engines_allowed",
    "assert_entity_extractor_allowed",
    "assert_llm_provider_allowed",
    "entity_extractor_allowlist",
    "is_public_mode",
]
