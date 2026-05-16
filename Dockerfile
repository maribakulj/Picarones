# Dockerfile — Picarones
# Image Docker multi-étape avec Tesseract OCR pré-installé
#
# Usage :
#   docker build -t picarones:latest .
#   docker run -p 8000:8000 picarones:latest
#   docker run -p 8000:8000 -v $(pwd)/corpus:/app/corpus picarones:latest
#
# Variables d'environnement supportées :
#   OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY
#   GOOGLE_APPLICATION_CREDENTIALS
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
#   AZURE_DOC_INTEL_ENDPOINT, AZURE_DOC_INTEL_KEY

# ──────────────────────────────────────────────────────────────────
# Étape 1 : builder — installe les dépendances Python dans un venv
# ──────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────
# Sprint A8 (M-2) + Sprint A16 (build déterministe) — image de base
# épinglée à la fois par tag (lisibilité humaine) et par digest sha256
# (reproductibilité bit-à-bit).
#
# Pourquoi le digest : ``python:3.11.13-slim`` peut être re-publié au
# fil des patches Debian avec un même tag mais un contenu différent.
# Pour la reproductibilité institutionnelle BnF, ``@sha256:...`` fige
# l'image binaire — deux ``docker build`` séparés produisent une
# couche de base identique octet par octet.
#
# Rotation trimestrielle (avant chaque release majeure) :
#
#     TOKEN=$(curl -s "https://auth.docker.io/token?\
#       service=registry.docker.io&scope=repository:library/python:pull" \
#       | jq -r .token)
#     curl -sI -H "Authorization: Bearer $TOKEN" \
#       -H "Accept: application/vnd.oci.image.index.v1+json" \
#       https://registry-1.docker.io/v2/library/python/manifests/3.11.13-slim \
#       | grep -i docker-content-digest
#     # → mettre à jour le digest ci-dessous + bumper PYTHON_BASE_IMAGE
#
# La forme ``image:tag@sha256:...`` est documentée par Docker comme
# valide ; les machines de développement sans registry proxy peuvent
# pull aussi bien que par tag — le digest étant immuable, le pull
# est strictement équivalent à un pull par tag actuel.
# ──────────────────────────────────────────────────────────────────
ARG PYTHON_BASE_IMAGE=python:3.11.13-slim@sha256:9bffe4353b925a1656688797ebc68f9c525e79b1d377a764d232182a519eeec4

FROM ${PYTHON_BASE_IMAGE} AS builder

WORKDIR /app

# Sprint A14 (correctif suite scan Trivy CI) — applique en priorité les
# patches Debian disponibles AVANT d'installer build-essential/git, pour
# éviter d'embarquer les CVE de la base image (libssl3t64, libc6, etc.).
#
# ``--fix-missing`` + ``Acquire::Retries=3`` : tolère la rotation du pool
# debian-security. Quand Debian publie un point-release, l'ancien .deb
# est émondé du pool quasi immédiatement alors que l'index fraîchement
# récupéré le référence encore → 404 transitoire sur un paquet hors
# scope (ex. linux-libc-dev, en-têtes noyau inutiles au runtime). On
# saute ce paquet au lieu de casser le build ; tous les patches CVE
# téléchargeables (libssl3t64, libc6, openssl, zlib1g…) sont appliqués.
RUN apt-get update -o Acquire::Retries=3 && \
    apt-get upgrade -y --fix-missing -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=3 \
        build-essential \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copier les fichiers de configuration du package + lock file Docker.
# ``requirements-docker.lock`` (Sprint A16) gèle l'arbre de dépendances
# transitif résolu par ``uv pip compile pyproject.toml --extra web --extra llm``.
COPY pyproject.toml .
COPY README.md .
COPY requirements-docker.lock .
COPY picarones/ picarones/

# Crée le venv isolé /opt/venv et l'active pour les ``RUN`` suivants.
# Le runtime fera ``COPY --from=builder /opt/venv /opt/venv`` ; sans cette
# création explicite le COPY échoue (régression remontée par CI A14).
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Sprint A16 : installation déterministe via lock file.
#
# 1. Patch pip/setuptools/wheel (Trivy scanne /opt/venv/lib/python3.11/
#    site-packages — sans patch les CVE setuptools/wheel ressortent).
# 2. ``--no-deps`` sur le lock empêche pip de re-résoudre — l'arbre
#    pinné par ``uv pip compile`` est complet, transitives incluses.
# 3. ``--no-deps`` sur picarones lui-même : le lock contient déjà
#    toutes ses dépendances ; cette ligne installe juste le code.
RUN pip install --upgrade --no-cache-dir \
        "pip>=24.2" "setuptools>=78.1.1" "wheel>=0.46.2" && \
    pip install --no-cache-dir --no-deps -r requirements-docker.lock && \
    pip install --no-cache-dir --no-deps -e . && \
    pip cache purge

# Patch également la copie système de pip/setuptools/wheel (hors venv)
# que Trivy détecte via ``/usr/local/lib/python3.11/site-packages`` —
# subsiste dans l'image runtime même quand le venv est utilisé.
RUN /usr/local/bin/pip install --upgrade --no-cache-dir \
    "pip>=24.2" "setuptools>=78.1.1" "wheel>=0.46.2"

# ──────────────────────────────────────────────────────────────────
# Étape 2 : runtime — image finale légère avec Tesseract
# ──────────────────────────────────────────────────────────────────
# ARG redéclaré ici car les variables ARG hors ``FROM`` sont scopées
# par étape ; sans cette redéclaration le ``FROM`` du runtime perd
# l'épinglage du builder. La valeur DOIT correspondre à celle de
# l'étape builder (digest inclus) — sinon les couches OS divergent.
ARG PYTHON_BASE_IMAGE=python:3.11.13-slim@sha256:9bffe4353b925a1656688797ebc68f9c525e79b1d377a764d232182a519eeec4
FROM ${PYTHON_BASE_IMAGE} AS runtime

LABEL description="Picarones — Plateforme de comparaison de moteurs OCR pour documents patrimoniaux"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/maribakulj/Picarones"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# ── Dépendances système ─────────────────────────────────────────
# Sprint A14 (correctif Trivy) : ``apt-get upgrade -y`` avant install
# pour récupérer les patches de sécurité Debian (libssl3t64, libc6,
# openssl, etc.) — la base image Python ne les inclut pas par défaut.
# ``--fix-missing`` + ``Acquire::Retries=3`` : même résilience à la
# rotation du pool debian-security que l'étape builder (cf. supra).
#
# Sprint S6.1 — reproductibilité institutionnelle (BnF) :
#
# ``tesseract-ocr`` n'est PAS pinné à une version exacte (ex :
# ``=5.3.0-2``) car Debian point-release rebump fréquemment :
# ``5.3.0-2`` → ``5.3.0-2+deb12u1`` → ``5.3.4-1``.  Un pin exact
# casse le build dès que la version disparaît du miroir.
#
# Le contrat de reproductibilité repose plutôt sur :
#
# 1. La base image Python pinée par digest SHA256 (cf. ``ARG
#    PYTHON_BASE_IMAGE`` ci-dessus) — Debian bookworm garantit la
#    stabilité ABI au sein du même point-release.
# 2. ``requirements-docker.lock`` qui fige les versions Python.
# 3. Le ``RunManifest.dependencies_lock`` capture la version
#    Tesseract effective au runtime (``tesseract --version``)
#    pour traçabilité scientifique.
#
# Si une version mineure de Tesseract introduit une régression
# CER, le mainteneur peut pinner explicitement ICI à ce moment-là
# (avec une note CHANGELOG).
RUN apt-get update -o Acquire::Retries=3 && \
    apt-get upgrade -y --fix-missing -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=3 \
        # Tesseract OCR 5 + modèles de langues (Debian bookworm).
        tesseract-ocr \
        tesseract-ocr-fra \
        tesseract-ocr-lat \
        tesseract-ocr-eng \
        tesseract-ocr-deu \
        tesseract-ocr-ita \
        tesseract-ocr-spa \
        # Bibliothèques image pour Pillow
        libpng16-16 \
        libjpeg62-turbo \
        libtiff6 \
        libwebp7 \
        # Utilitaires
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ── Vérification fail-fast de Tesseract (incident 2026-05-16) ───
# ``apt-get upgrade --fix-missing`` (résilience à la rotation du pool
# debian-security, commit d5d68ae) peut laisser un jeu de libs
# runtime INCOHÉRENT si un .deb co-dépendant est sauté : le binaire
# ``tesseract`` se fige alors à la reconnaissance (deadlock OpenMP /
# mismatch ABI liblept/libstdc++) et le run prod timeoute sur CHAQUE
# document — sans que le build n'ait rien signalé.
#
# On exige donc, au build : (a) que Tesseract charge ses libs et
# expose la langue ``fra``, (b) qu'une reconnaissance réelle se
# termine sous 30 s.  Un Tesseract cassé OU figé casse désormais le
# BUILD (le Space conserve l'image précédente fonctionnelle) au lieu
# de déployer une image qui gèle en silence.  ``--fix-missing`` est
# conservé : l'intention CVE de d5d68ae n'est pas régressée, seule
# la défaillance silencieuse est convertie en échec bruyant.
RUN set -eu; \
    timeout 30 tesseract --version; \
    timeout 30 tesseract --list-langs 2>&1 | grep -qx fra; \
    printf 'P1\n16 16\n' > /tmp/tess_smoke.pbm; \
    i=0; while [ "$i" -lt 16 ]; do \
        printf '1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0\n' >> /tmp/tess_smoke.pbm; \
        i=$((i + 1)); \
    done; \
    rc=0; \
    timeout 30 tesseract /tmp/tess_smoke.pbm - -l fra --psm 6 \
        > /dev/null 2>&1 || rc=$?; \
    rm -f /tmp/tess_smoke.pbm; \
    if [ "$rc" = 124 ]; then \
        echo "FATAL: tesseract a gelé (timeout reconnaissance) — \
build refusé, libs runtime probablement incohérentes." >&2; \
        exit 1; \
    fi

# Patch pip/setuptools/wheel système du runtime (en dehors du venv).
# Trivy scanne /usr/local/lib/python3.11/site-packages indépendamment.
RUN /usr/local/bin/pip install --upgrade --no-cache-dir \
    "pip>=24.2" "setuptools>=78.1.1" "wheel>=0.46.2"

# ── Venv Python depuis le builder ──────────────────────────────
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Code source de l'application ───────────────────────────────
COPY --from=builder /app /app

# ── Répertoires de données ──────────────────────────────────────
RUN mkdir -p /app/corpus /app/rapports /app/data

# ── Utilisateur non-root pour la sécurité ──────────────────────
RUN useradd -m -u 1000 picarones && \
    chown -R picarones:picarones /app /opt/venv
USER picarones

# ── Variables d'environnement par défaut ───────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
# Tesseract LSTM (oem=3) parallélise via OpenMP. Sur le Space HF
# (~2 vCPU partagés) l'OpenMP non borné suroccupe le CPU (N threads
# pour 2 cœurs) → OCR plus LENT et instable. Le forcer mono-thread
# par appel est ici PLUS rapide et déterministe (le parallélisme
# inter-documents est déjà géré par le ThreadPool du CorpusRunner).
ENV OMP_THREAD_LIMIT=1

# ── Ports ───────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Démarrage ───────────────────────────────────────────────────
CMD ["picarones", "serve", "--host", "0.0.0.0", "--port", "7860"]
