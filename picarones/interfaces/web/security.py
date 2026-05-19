"""Garde-fous sécurité pour l'interface web.

Ce module centralise quatre durcissements pour rendre Picarones déployable
sur un Space HuggingFace public ou un serveur d'institution sans donner
les clefs du royaume au premier visiteur :

1. **Mode public**  (``PICARONES_PUBLIC_MODE=1``) — désactive les
   pipelines OCR+LLM et les moteurs OCR cloud, dont les clefs API sont
   mutualisées côté serveur (OPENAI_API_KEY, ANTHROPIC_API_KEY,
   MISTRAL_API_KEY, etc.). Sans ce garde-fou, n'importe quel visiteur
   consomme le quota du mainteneur via 10 lignes de ``curl``.

2. **Browse roots restreints** — ``PICARONES_BROWSE_ROOTS`` (chemins
   séparés par ``:``) remplace la liste hardcodée. Par défaut,
   uniquement ``./uploads/`` est exposé en mode public ; en mode ``dev``
   on conserve l'ancien comportement (cwd, ``/workspaces``, ``tempdir``).

3. **Validation des images uploadées** — appel à ``Image.verify()`` dans
   un ``try/except`` capturant ``DecompressionBombError``,
   ``UnidentifiedImageError`` et l'exception générique de Pillow.
   Limite de taille via ``PICARONES_MAX_UPLOAD_MB`` (défaut 100).

4. **Rate limiting + plafond de jobs concurrents** — limiteur en mémoire
   par IP (``PICARONES_RATE_LIMIT_PER_HOUR``) et sémaphore global
   (``PICARONES_MAX_CONCURRENT_JOBS``).

Le tout est piloté par variables d'environnement pour ne pas obliger un
mainteneur à patcher du code lors du passage à la prod.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode public
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Validation des chemins utilisateur (Sprint A14-S1, A.I.0 P0)
#
# Ré-importé depuis le foyer définitif ``picarones.app.services.path_security``
# (Sprint A14-S19).  Pas de duplication — le code vit en un seul
# endroit dans la couche app, accessible aussi par la CLI et les jobs
# background.
# ---------------------------------------------------------------------------

from picarones.app.services.path_security import (
    PathValidationError as PathValidationError,
    safe_report_name as safe_report_name,
    validated_path as validated_path,
    validated_prompt_filename as validated_prompt_filename,
)
from picarones.app.services.path_security import (
    _is_within as _is_within,  # noqa: F401
)


# ---------------------------------------------------------------------------
# Browse roots
# ---------------------------------------------------------------------------

def compute_browse_roots(uploads_dir: Path) -> list[Path]:
    """Retourne la liste de répertoires autorisés pour ``/api/corpus/browse``.

    - Variable d'env ``PICARONES_BROWSE_ROOTS`` (séparateur ``os.pathsep``,
      ``:`` sur Linux/macOS, ``;`` sur Windows) : prioritaire si définie.
    - Sinon, mode public ⇒ uniquement ``uploads_dir``.
    - Sinon, mode dev (défaut) ⇒ cwd + uploads_dir + ``/workspaces``
      (Codespaces) + ``tempdir`` (compatibilité ascendante).
    """
    raw = os.environ.get("PICARONES_BROWSE_ROOTS")
    if raw:
        roots = [Path(p).resolve() for p in raw.split(os.pathsep) if p.strip()]
        return roots

    if is_public_mode():
        return [uploads_dir.resolve()]

    import tempfile
    return [
        Path(".").resolve(),
        uploads_dir.resolve(),
        Path("/workspaces").resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]


def compute_workspace_roots(uploads_dir: Path) -> list[Path]:
    """Retourne les racines autorisées pour les opérations de benchmark.

    Sprint A14-S1 — A.I.0 P0 : utilisé par les endpoints
    ``/api/benchmark/start`` et ``/api/benchmark/run`` pour valider
    ``corpus_path`` et ``output_dir`` via :func:`validated_path`.

    Sémantique :

    - Si ``PICARONES_WORKSPACE_ROOTS`` est défini, prend précédence
      absolue (admin sait ce qu'il fait).
    - Sinon, en mode public : uniquement ``uploads_dir`` (lecture)
      et ``./rapports`` (écriture des rapports générés).
    - Sinon, mode dev : ``compute_browse_roots`` + ``./rapports`` +
      ``./corpus`` (corpus locaux des développeurs).

    En production institutionnelle, exporter ``PICARONES_WORKSPACE_ROOTS``
    pour épingler explicitement les répertoires autorisés.
    """
    raw = os.environ.get("PICARONES_WORKSPACE_ROOTS")
    if raw:
        return [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip()]

    base = compute_browse_roots(uploads_dir)
    extras = [
        Path("./rapports").resolve(),
        Path("./corpus").resolve(),
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for p in base + extras:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Clusters extraits (audit prod P1.2 — dégonflage du god-module).
# Réimportés ici : ``security`` reste le point d'import public
# (``from picarones.interfaces.web.security import validate_image_safe``
# etc. reste valide ; ``security._csrf_secret_runtime`` aussi car le
# bloc CSRF n'a pas bougé).  Les ``as`` signalent un ré-export
# explicite (ruff F401 OK).
# ---------------------------------------------------------------------------

from picarones.interfaces.web.security_csp import (
    DEFAULT_CSP as DEFAULT_CSP,
    _CSP_BASE as _CSP_BASE,
    _HF_FRAME_ANCESTORS as _HF_FRAME_ANCESTORS,
    _frame_ancestors_directive as _frame_ancestors_directive,
    csp_middleware as csp_middleware,
    get_csp_policy as get_csp_policy,
    is_huggingface_space as is_huggingface_space,
)
from picarones.interfaces.web.security_rate_limit import (
    RateLimiter as RateLimiter,
    get_max_concurrent_jobs as get_max_concurrent_jobs,
    get_rate_limit_per_hour as get_rate_limit_per_hour,
)
from picarones.interfaces.web.security_uploads import (
    UPLOAD_CHUNK_SIZE as UPLOAD_CHUNK_SIZE,
    _verify_image_with_pillow as _verify_image_with_pillow,
    get_max_total_upload_mb as get_max_total_upload_mb,
    get_max_upload_mb as get_max_upload_mb,
    validate_image_file_safe as validate_image_file_safe,
    validate_image_safe as validate_image_safe,
)


# ---------------------------------------------------------------------------
# CSRF — Sprint A4 (item B-11)
#
# Pattern « double-submit cookie » : à chaque GET, le serveur pose un
# cookie ``picarones_csrf`` (httponly=False car le JS doit le lire) qui
# contient un token signé. Sur POST/PUT/DELETE/PATCH, le client doit
# renvoyer ce token dans le header ``X-CSRF-Token``. Le serveur compare
# les deux (constant-time) et refuse 403 sinon.
#
# Activation : ``PICARONES_CSRF_REQUIRED=1`` (défaut désactivé pour
# rétrocompat HuggingFace Space sans session). En mode institutionnel
# derrière SSO, à activer d'office.
#
# Secret : ``PICARONES_CSRF_SECRET`` env var. Si absent, généré au
# démarrage (warning explicite — perte du secret entre redémarrages,
# acceptable pour des sessions courtes).
# ---------------------------------------------------------------------------

import hashlib
import hmac
import secrets

#: Nom du cookie CSRF (httponly=False — lu par le JS du frontend).
CSRF_COOKIE = "picarones_csrf"

#: Header HTTP que le client doit renvoyer sur POST/PUT/DELETE/PATCH.
CSRF_HEADER = "X-CSRF-Token"

#: Méthodes HTTP qui exigent un token valide.
CSRF_PROTECTED_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Préfixes de chemin exemptés. Les endpoints purement informatifs ou
#: appelés depuis des outils CLI tiers (curl, wget) restent accessibles
#: sans token. Tout endpoint qui modifie l'état applicatif doit rester
#: protégé — ne pas étendre cette liste sans revue sécurité.
CSRF_EXEMPT_PATH_PREFIXES: tuple[str, ...] = (
    "/health",
    "/api/csrf/token",  # le endpoint qui *donne* le token
)

_csrf_secret_runtime: bytes | None = None


def is_csrf_required() -> bool:
    """Vrai si la protection CSRF doit être active (mode institutionnel)."""
    return os.environ.get("PICARONES_CSRF_REQUIRED", "").strip() in ("1", "true", "yes")


def secure_cookies() -> bool:
    """Décide de l'attribut ``Secure`` des cookies posés par l'app.

    Auparavant codé en dur à ``False`` (foot-gun : le cookie CSRF
    transitait en clair même derrière TLS).  Désormais :

    - ``PICARONES_SECURE_COOKIES`` explicite ⇒ prioritaire
      (``1/true/yes`` vs ``0/false/no``) ;
    - sinon, ``True`` **uniquement** sur HuggingFace Space (servi en
      HTTPS de façon certaine — signal factuel) ;
    - sinon ``False``.

    Axe découplé volontairement de ``PICARONES_PUBLIC_MODE`` :
    « public/mutualisé » (restriction fonctionnelle) n'implique pas
    « servi en HTTPS ».  Le compose local met ``PUBLIC_MODE=1`` mais
    binde ``127.0.0.1`` en HTTP simple : un cookie ``Secure`` y serait
    silencieusement ignoré par le navigateur (préférence de langue
    cassée).  Le durcissement prod pose ``PICARONES_SECURE_COOKIES=1``
    explicitement (override ``docker-compose.prod.yml``).
    """
    raw = os.environ.get("PICARONES_SECURE_COOKIES", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    return is_huggingface_space()


def check_deployment_coherence() -> None:
    """Échoue *au démarrage* sur une combinaison de config dangereuse.

    Principe : un défaut sûr ne doit pas dépendre de la lecture
    intégrale de la doc par l'opérateur (cf. audit prod).  On bloque
    le seul cas réellement contradictoire et exploitable :

      CSRF exigé + cookies non-``Secure`` + déploiement exposé
      (HF Space ou mode public)

    → le token CSRF transiterait en clair, vidant la protection de
    son sens.  Les autres incohérences ne sont que loggées (warning
    non bloquant) pour ne pas casser un dev local légitime.
    """
    exposed = is_huggingface_space() or is_public_mode()
    if is_csrf_required() and not secure_cookies() and exposed:
        raise RuntimeError(
            "Config incohérente : PICARONES_CSRF_REQUIRED=1 sur un "
            "déploiement exposé (HF Space / mode public) mais cookies "
            "non-Secure. Le token CSRF transiterait en clair. Poser "
            "PICARONES_SECURE_COOKIES=1 (derrière TLS) ou désactiver "
            "le mode public."
        )

    cloud_keys = [
        k for k in (
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY",
            "AZURE_DOC_INTEL_KEY", "AWS_SECRET_ACCESS_KEY",
        )
        if os.environ.get(k, "").strip()
    ]
    if cloud_keys and not is_public_mode() and not is_csrf_required():
        logger.warning(
            "[security] Clés cloud présentes (%s) sans mode public ni "
            "CSRF : une surface web non authentifiée peut déclencher "
            "des appels facturés. Activer PICARONES_PUBLIC_MODE=1 ou "
            "PICARONES_CSRF_REQUIRED=1 (+ SSO) si l'instance est "
            "accessible au-delà de localhost.",
            ", ".join(cloud_keys),
        )


def _get_csrf_secret() -> bytes:
    """Retourne le secret HMAC.  Priorité ``PICARONES_CSRF_SECRET``,
    sinon génère un secret runtime persistant durant la vie du process.
    """
    global _csrf_secret_runtime
    env = os.environ.get("PICARONES_CSRF_SECRET")
    if env:
        return env.encode("utf-8")
    if _csrf_secret_runtime is None:
        _csrf_secret_runtime = secrets.token_bytes(32)
        logger.warning(
            "[security] PICARONES_CSRF_SECRET non défini — secret généré au "
            "démarrage. Les tokens CSRF seront invalidés au prochain "
            "redémarrage. En production, exporter un secret stable."
        )
    return _csrf_secret_runtime


class CSRFConfigError(RuntimeError):
    """Levée au démarrage si la config CSRF est incohérente.

    Sprint S6.9 — en mode institutionnel
    (``PICARONES_CSRF_REQUIRED=1``), exiger un ``PICARONES_CSRF_SECRET``
    stable ; sans lui, les tokens sont invalidés à chaque redémarrage,
    ce qui dégrade l'UX et masque une mauvaise configuration ops.
    """


def validate_csrf_config() -> None:
    """Refuse le démarrage si la config CSRF est dangereuse.

    Sprint S6.9 — appelé au lifespan de l'app FastAPI.  Trois cas :

    1. ``PICARONES_CSRF_REQUIRED`` désactivé → bypass total
       (mode public HF Space).  Aucun secret nécessaire.
    2. ``PICARONES_CSRF_REQUIRED=1`` ET ``PICARONES_CSRF_SECRET``
       défini → OK.
    3. ``PICARONES_CSRF_REQUIRED=1`` ET ``PICARONES_CSRF_SECRET``
       absent → :class:`CSRFConfigError` levée (refus démarrage).

    Cas 3 est dangereux : sans secret stable, le serveur génère un
    secret aléatoire au démarrage.  Tous les tokens CSRF sont
    invalidés à chaque restart → UX cassée, et une équipe ops qui
    voit ``PICARONES_CSRF_REQUIRED=1`` croit (à tort) que la
    config est complète.

    Raises
    ------
    CSRFConfigError
        Si le mode CSRF est requis sans secret stable.
    """
    if not is_csrf_required():
        return
    secret = os.environ.get("PICARONES_CSRF_SECRET", "").strip()
    if not secret:
        raise CSRFConfigError(
            "PICARONES_CSRF_REQUIRED=1 mais PICARONES_CSRF_SECRET "
            "n'est pas défini.  En mode institutionnel, le secret "
            "doit être stable entre redémarrages — sinon tous les "
            "tokens CSRF émis sont invalidés à chaque restart.\n\n"
            "Solution : exporter un secret généré une fois pour "
            "toutes :\n"
            "  export PICARONES_CSRF_SECRET=$(openssl rand -hex 32)\n\n"
            "Et le persister dans le mécanisme de secrets de "
            "l'institution (Vault, AWS Secrets Manager, "
            "kubernetes Secret, etc.)."
        )
    # Garde-fou : un secret évident (vide après strip, "secret",
    # "changeme", ...) doit aussi alerter.
    weak_values = {"changeme", "secret", "password", "test", "dev"}
    if secret.lower() in weak_values:
        raise CSRFConfigError(
            f"PICARONES_CSRF_SECRET a une valeur trivialement faible "
            f"({secret!r}).  Utiliser ``openssl rand -hex 32``."
        )


def generate_csrf_token() -> str:
    """Produit un token signé HMAC-SHA256.

    Format : ``<nonce_hex>.<signature_hex>`` où la signature est
    ``HMAC-SHA256(secret, nonce)``. Le nonce est rotué à chaque
    génération — pas de réutilisation.
    """
    nonce = secrets.token_bytes(16)
    sig = hmac.new(_get_csrf_secret(), nonce, hashlib.sha256).digest()
    return f"{nonce.hex()}.{sig.hex()}"


def verify_csrf_token(token: str | None) -> bool:
    """Valide la signature d'un token. Compare en temps constant.

    Retourne ``False`` sur token absent, mal formé, ou signature
    incorrecte. Pas de fuite d'information sur la cause.
    """
    if not token or "." not in token:
        return False
    try:
        nonce_hex, sig_hex = token.split(".", 1)
        nonce = bytes.fromhex(nonce_hex)
        sig_provided = bytes.fromhex(sig_hex)
    except (ValueError, AttributeError):
        return False
    sig_expected = hmac.new(_get_csrf_secret(), nonce, hashlib.sha256).digest()
    return hmac.compare_digest(sig_provided, sig_expected)


async def csrf_middleware(request, call_next):
    """Middleware FastAPI — protège les méthodes mutantes en mode CSRF.

    Comportement :

    1. Si ``PICARONES_CSRF_REQUIRED`` n'est pas activé → bypass complet
       (rétrocompat HuggingFace Space public).
    2. Sinon, si la méthode est dans ``CSRF_PROTECTED_METHODS`` et que
       le chemin n'est pas exempté → exiger un token valide. Renvoie
       403 si manquant ou invalide.
    3. Pose un cookie ``picarones_csrf`` à chaque réponse pour les
       chemins non exempts (rotation à chaque GET).

    Le pattern « double-submit cookie » + signature HMAC garantit que
    seul un client qui a *à la fois* le cookie et a pu lire sa valeur
    via JS (donc qui n'est pas un site tiers) peut soumettre le header
    correspondant.
    """
    from fastapi.responses import JSONResponse

    if not is_csrf_required():
        return await call_next(request)

    path = request.url.path
    is_exempt = any(path.startswith(p) for p in CSRF_EXEMPT_PATH_PREFIXES)
    method = request.method.upper()

    # Vérification : méthode mutante non exemptée → token obligatoire
    if method in CSRF_PROTECTED_METHODS and not is_exempt:
        cookie_token = request.cookies.get(CSRF_COOKIE)
        header_token = request.headers.get(CSRF_HEADER)
        if not cookie_token or not header_token:
            logger.warning(
                "[security/csrf] %s %s refusé : token cookie=%r header=%r",
                method,
                path,
                bool(cookie_token),
                bool(header_token),
            )
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        "CSRF token requis sur cette méthode. Récupérer un "
                        f"token via GET /api/csrf/token et le passer dans "
                        f"l'en-tête {CSRF_HEADER}."
                    ),
                },
            )
        if not hmac.compare_digest(cookie_token, header_token):
            logger.warning(
                "[security/csrf] %s %s refusé : cookie/header divergent",
                method, path,
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF token cookie/header divergent."},
            )
        if not verify_csrf_token(cookie_token):
            logger.warning(
                "[security/csrf] %s %s refusé : signature invalide",
                method, path,
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF token invalide ou expiré."},
            )

    response = await call_next(request)

    # Rotation : on pose un cookie frais sur tout GET non-exempt qui n'a
    # pas déjà un cookie, ou si la réponse est un endpoint qui force la
    # rotation. Pour les autres méthodes, on conserve le cookie courant.
    if method == "GET" and not is_exempt:
        if CSRF_COOKIE not in request.cookies:
            response.set_cookie(
                key=CSRF_COOKIE,
                value=generate_csrf_token(),
                httponly=False,  # le JS doit pouvoir le lire
                samesite="strict",
                secure=secure_cookies(),
            )
    return response
