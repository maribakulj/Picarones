"""``WorkspaceManager`` + helpers de validation de chemin — Sprint A14-S19.

Foyer définitif des helpers ``validated_path``, ``safe_report_name``,
``validated_prompt_filename`` créés au S1.  Les callers web
(``picarones.interfaces.web.security``) ré-importent depuis ce module.

Pourquoi ici
------------
La sécurité chemin n'est pas un détail web — c'est une garantie
applicative qui doit valoir aussi pour la CLI, les tests d'intégration,
les jobs background, et tout caller qui manipule des paths utilisateur.

Le service ``WorkspaceManager`` centralise la création d'un dossier
isolé par session et garantit que toute écriture/lecture y reste
confinée — c'est ce qui permettra au ``BenchmarkService`` (S17) de
tourner sur un upload utilisateur sans risque de path traversal.

Anti-sur-ingénierie
-------------------
- Pas d'auto-cleanup au garbage collector — le caller appelle
  ``cleanup()`` explicitement (équivalent à
  ``tempfile.TemporaryDirectory.cleanup``).  Une session web peut
  vouloir conserver les artefacts pour téléchargement ultérieur, c'est
  son choix.
- Pas de quota disque — c'est une responsabilité OS-level
  (cgroup, ulimit, quota fs).  Le service ne se substitue pas.
- Pas de chiffrement at-rest — les fichiers sont en clair sous le
  workspace.  Si un institutionnel veut chiffrement, c'est au
  niveau filesystem (LUKS, eCryptfs).
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from picarones.domain.errors import PicaronesError


class PathValidationError(PicaronesError, ValueError):
    """Levée quand un chemin utilisateur sort de la zone autorisée.

    Hérite à la fois de :class:`PicaronesError` (convention métier
    du nouveau code) et :class:`ValueError` (rétrocompat S1 — un
    caller historique qui ``except ValueError`` continue de marcher).
    """


def validated_path(
    user_path: str,
    allowed_roots: list[Path],
    must_exist: bool = False,
    must_be_dir: bool = False,
) -> Path:
    """Résout un chemin utilisateur et vérifie qu'il reste dans une racine
    autorisée.

    Garde-fou central contre la traversée de répertoires (path traversal)
    et l'écriture/lecture arbitraire dans le système de fichiers du
    serveur.

    Algorithme :

    1. Refuse les chemins vides ou contenant des octets nuls.
    2. Résout le chemin de manière absolue (``Path.resolve()``) — ça
       écrase ``..``, les liens symboliques et les chemins relatifs.
    3. Vérifie que le résultat est ``.is_relative_to(root)`` pour au
       moins une des ``allowed_roots`` (elles aussi pré-résolues).
    4. Optionnellement : vérifie l'existence et le type (dir).

    Parameters
    ----------
    user_path:
        Chemin tel que reçu de l'utilisateur (str).  Peut être absolu
        ou relatif.
    allowed_roots:
        Liste de répertoires racines (``Path``) au sein desquels le
        chemin résolu doit se trouver.  Liste vide = tout refuser.
    must_exist:
        Si ``True``, exige que le chemin résolu existe sur le disque.
    must_be_dir:
        Si ``True``, exige que le chemin résolu existe ET soit un
        répertoire.  Implique ``must_exist=True``.

    Returns
    -------
    Path
        Chemin résolu absolu, garanti dans une des racines autorisées.

    Raises
    ------
    PathValidationError
        Si le chemin est vide, contient un octet nul, sort des racines
        autorisées, ou ne satisfait pas ``must_exist`` / ``must_be_dir``.
    """
    if not user_path or not user_path.strip():
        raise PathValidationError("Chemin vide.")
    if "\x00" in user_path:
        raise PathValidationError("Chemin contient un octet nul.")
    if not allowed_roots:
        raise PathValidationError(
            "Aucune racine autorisée — refus de toute requête de chemin."
        )

    try:
        resolved = Path(user_path).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise PathValidationError(f"Chemin invalide : {exc}") from exc

    resolved_roots = [Path(r).expanduser().resolve() for r in allowed_roots]
    if not any(_is_within(resolved, root) for root in resolved_roots):
        raise PathValidationError(
            f"Chemin hors zone autorisée : {user_path!r}.  "
            f"Racines acceptées : {[str(r) for r in resolved_roots]}."
        )

    if must_be_dir or must_exist:
        if not resolved.exists():
            raise PathValidationError(f"Chemin inexistant : {user_path!r}.")
    if must_be_dir and not resolved.is_dir():
        raise PathValidationError(
            f"Chemin n'est pas un répertoire : {user_path!r}."
        )

    return resolved


def _is_within(child: Path, parent: Path) -> bool:
    """Vrai si ``child`` est ``parent`` ou un descendant.

    ``Path.is_relative_to`` n'existe qu'à partir de Python 3.9 — on
    utilise ``relative_to`` via try/except pour rester explicite sur
    l'intention.
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validated_prompt_filename(name: str) -> str:
    """Valide qu'un ``prompt_file`` est un simple nom de fichier sûr.

    Restreint la valeur reçue à un simple nom de fichier de la
    bibliothèque de prompts intégrée (``picarones/prompts/``).  Pas de
    ``/``, pas de ``\\``, pas de ``..``, pas d'absolu.

    Le caller (web layer, CLI, etc.) est responsable d'appeler cette
    fonction AVANT de transmettre la valeur au pipeline.

    Returns
    -------
    str
        Nom de fichier validé (basename uniquement).

    Raises
    ------
    PathValidationError
        Si la valeur contient un séparateur de chemin, un caractère de
        contrôle, ou ressemble à un chemin absolu/relatif suspect.
    """
    if not name:
        raise PathValidationError("Nom de prompt vide.")
    if "\x00" in name:
        raise PathValidationError("Nom de prompt contient un octet nul.")
    if any(c in name for c in ("/", "\\")):
        raise PathValidationError(
            f"Nom de prompt invalide (séparateur de chemin) : {name!r}.  "
            "Le caller n'accepte que les prompts de la bibliothèque "
            "intégrée — fournir le simple nom de fichier."
        )
    if name.startswith(".") or ".." in name:
        raise PathValidationError(
            f"Nom de prompt suspect : {name!r}.  "
            "Refus des préfixes ``.`` et des séquences ``..``."
        )
    if any(ord(c) < 0x20 for c in name):
        raise PathValidationError(
            "Nom de prompt contient un caractère de contrôle."
        )
    return name


def safe_report_name(name: str, max_length: int = 128) -> str:
    """Sanitize un nom de rapport utilisateur en composant de chemin sûr.

    Refuse les séparateurs de chemin (``/``, ``\\``), les caractères
    de contrôle, les octets nuls.  Tronque à ``max_length``.  Si la
    chaîne devient vide après nettoyage, lève ``PathValidationError``.

    Cette fonction NE produit PAS un chemin — elle produit un nom
    qu'un caller peut concaténer à un répertoire qu'il a déjà validé
    avec ``validated_path`` (ou via ``WorkspaceManager.subpath``).
    """
    if not name:
        raise PathValidationError("Nom de rapport vide.")
    if "\x00" in name:
        raise PathValidationError("Nom de rapport contient un octet nul.")
    bad = set("/\\")
    cleaned = "".join(
        c for c in name
        if c not in bad and ord(c) >= 0x20
    )
    cleaned = cleaned.strip().strip(".")
    if not cleaned:
        raise PathValidationError(
            f"Nom de rapport invalide après nettoyage : {name!r}."
        )
    if cleaned in (".", "..", ""):
        raise PathValidationError(f"Nom de rapport réservé : {name!r}.")
    return cleaned[:max_length]


# ──────────────────────────────────────────────────────────────────────
# WorkspaceManager — sandbox par session
# ──────────────────────────────────────────────────────────────────────


class WorkspaceManager:
    """Crée et gère un dossier isolé pour une session.

    Garanties
    ---------
    - Le workspace est unique par session (UUID4 par défaut, ou
      ``session_id`` explicite).
    - Toute lecture/écriture passe par :meth:`subpath` qui empêche la
      traversée hors du root via :func:`validated_path`.
    - :meth:`cleanup` supprime récursivement le dossier (irréversible
      — le caller est responsable du moment d'appel).

    Parameters
    ----------
    base_dir:
        Répertoire parent dans lequel créer le workspace.  Doit
        exister ; un sous-dossier ``<session_id>`` y sera créé.
    session_id:
        Identifiant de session.  ``None`` (défaut) génère un UUID4
        hexadécimal.  Sinon, doit passer :func:`safe_report_name`
        (refus des séparateurs et caractères de contrôle) — sinon
        ``PathValidationError``.

    Raises
    ------
    PathValidationError
        Si ``base_dir`` n'existe pas, n'est pas un répertoire, ou
        si ``session_id`` est invalide.
    OSError
        Si la création du sous-dossier échoue (permissions, etc.).

    Notes
    -----
    Pas d'auto-cleanup au garbage collector.  Le caller appelle
    :meth:`cleanup` explicitement.  Pour un usage RAII, utiliser
    le pattern context manager (le service expose ``__enter__`` et
    ``__exit__`` comme sucre).
    """

    def __init__(
        self,
        base_dir: Path | str,
        session_id: str | None = None,
    ) -> None:
        base = Path(base_dir).expanduser()
        if not base.exists():
            raise PathValidationError(
                f"WorkspaceManager : base_dir inexistant : {base!r}.",
            )
        if not base.is_dir():
            raise PathValidationError(
                f"WorkspaceManager : base_dir n'est pas un répertoire : "
                f"{base!r}.",
            )
        # base_dir résolu (absolu, sans symlinks).
        self._base = base.resolve()

        if session_id is None:
            session_id = uuid.uuid4().hex
        else:
            # Validation stricte : un session_id est un identifiant — on
            # le veut exact, pas silencieusement sanitizé.  Refus net si
            # contient un séparateur de chemin, ``..``, ou un caractère
            # de contrôle.  ``safe_report_name`` est ensuite utilisé
            # pour les contraintes additionnelles (longueur).
            if any(c in session_id for c in ("/", "\\")):
                raise PathValidationError(
                    f"WorkspaceManager : session_id contient un "
                    f"séparateur de chemin : {session_id!r}.",
                )
            if ".." in session_id:
                raise PathValidationError(
                    f"WorkspaceManager : session_id contient ``..`` : "
                    f"{session_id!r}.",
                )
            session_id = safe_report_name(session_id, max_length=64)
        self._session_id = session_id

        # Création du sous-dossier.  Si déjà présent, on accepte
        # (idempotent) — le caller peut vouloir réutiliser une session
        # interrompue.
        self._root = (self._base / self._session_id).resolve()
        # Vérification anti-collision de symlink : le root résolu doit
        # rester dans base.
        if not _is_within(self._root, self._base):
            raise PathValidationError(
                f"WorkspaceManager : root résolu {self._root!r} hors de "
                f"base {self._base!r} — symlink suspect ?",
            )
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Chemin absolu du workspace, garanti existant."""
        return self._root

    @property
    def session_id(self) -> str:
        return self._session_id

    def subpath(
        self,
        relative_or_absolute: str | Path,
        *,
        must_exist: bool = False,
        must_be_dir: bool = False,
    ) -> Path:
        """Résout un chemin et garantit qu'il reste dans le workspace.

        Accepte un chemin relatif (résolu sous ``root``) ou absolu (qui
        doit être lui-même sous ``root``).  Lève
        :class:`PathValidationError` sinon — c'est l'API à utiliser
        pour toute lecture/écriture déclenchée par une entrée
        utilisateur.

        Parameters
        ----------
        relative_or_absolute:
            Chemin tel que fourni par le caller.  Si relatif, on le
            joint à ``root``.  Si absolu, on vérifie qu'il est dans
            ``root``.
        must_exist:
            Si ``True``, exige que le chemin existe.
        must_be_dir:
            Si ``True``, exige que le chemin existe ET soit un dir.

        Returns
        -------
        Path
            Chemin résolu absolu, garanti sous ``root``.
        """
        # Refus explicite des entrées vides ou avec octet nul AVANT
        # ``Path()`` qui les normalise silencieusement (``Path("")`` ==
        # ``Path(".")``, ce qui pointerait sur le root).
        if isinstance(relative_or_absolute, str):
            if not relative_or_absolute or not relative_or_absolute.strip():
                raise PathValidationError("Chemin vide.")
            if "\x00" in relative_or_absolute:
                raise PathValidationError("Chemin contient un octet nul.")
        rel = Path(relative_or_absolute)
        if rel.is_absolute():
            target_str = str(rel)
        else:
            target_str = str(self._root / rel)
        return validated_path(
            target_str,
            allowed_roots=[self._root],
            must_exist=must_exist,
            must_be_dir=must_be_dir,
        )

    def safe_output_path(self, name: str, *, max_length: int = 128) -> Path:
        """Combine :func:`safe_report_name` avec :meth:`subpath`.

        Pour produire un chemin de sortie depuis un nom utilisateur
        sans séparateurs ni traversée.  Le caller peut ensuite écrire
        à ce chemin sans risque.
        """
        sanitized = safe_report_name(name, max_length=max_length)
        return self.subpath(sanitized)

    def cleanup(self) -> None:
        """Supprime récursivement le workspace.

        Idempotent : si le dossier n'existe plus, no-op silencieux.
        Après ``cleanup()``, toute opération sur ce manager est
        non définie (créer un nouveau manager pour une nouvelle
        session).

        Cross-OS robustesse
        ~~~~~~~~~~~~~~~~~~~
        Sur Windows, ``shutil.rmtree`` peut lever ``PermissionError``
        si un fichier porte l'attribut ``read-only`` (cas typique :
        ``__pycache__/*.pyc`` extraits depuis un ZIP).  Le handler
        ``_on_rmtree_error`` retire l'attribut puis retry.

        Sur certains filesystems (NFS, Windows avec
        anti-virus / indexeur), un fichier peut rester verrouillé
        quelques ms après sa fermeture.  Le handler propose un seul
        retry — au-delà, on laisse remonter l'erreur (signal d'un
        problème environnemental réel, pas un cas dégénéré du
        rewrite).
        """
        if not self._root.exists():
            return
        # Python 3.12+ utilise ``onexc`` (signature plus propre que
        # l'ancien ``onerror``).  On utilise ``onerror`` pour rester
        # compatible 3.11+ ; ``shutil`` continuera de l'accepter
        # jusqu'à la 3.14.
        shutil.rmtree(self._root, onerror=_on_rmtree_error)

    # ──────────────────────────────────────────────────────────────────
    # Context manager (sucre RAII)
    # ──────────────────────────────────────────────────────────────────

    def __enter__(self) -> "WorkspaceManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


def _on_rmtree_error(func, path, exc_info):
    """Handler pour ``shutil.rmtree`` Windows-safe.

    Cas typique : un fichier en read-only refuse d'être supprimé
    sur Windows (``PermissionError``).  On retire l'attribut puis
    on retry une fois.  Si ça échoue encore, on propage — c'est un
    vrai problème environnemental.
    """
    import os
    import stat
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        # Le chmod lui-même a échoué — on laisse la prochaine
        # tentative remonter l'erreur originale.
        pass
    func(path)


__all__ = [
    "PathValidationError",
    "WorkspaceManager",
    "safe_report_name",
    "validated_path",
    "validated_prompt_filename",
]
