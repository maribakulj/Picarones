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
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
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
#
# Sprint S6.1 — Reproductibilité institutionnelle (BnF) :
# ``tesseract-ocr`` est pinné à ``5.3.0-2`` (version Debian 12
# bookworm).  Sans ce pin, ``apt-get install tesseract-ocr`` peut
# remonter une version mineure différente entre deux builds (ex :
# ``5.3.0-2`` → ``5.4.1-1``) et faire dériver les CER mesurés.
# Pour un benchmark reproductible cité dans une publication
# scientifique, c'est inacceptable.
#
# Si Debian 12 sort un point release (bookworm-2 → bookworm-3) et
# rebump tesseract-ocr, le build échouera avec un message clair —
# le mainteneur devra alors décider explicitement de la mise à jour
# (et regénérer les baselines CER si la nouvelle version a un
# comportement différent).
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        # Tesseract OCR 5.3.0 (Debian 12) + modèles de langues
        tesseract-ocr=5.3.0-2 \
        tesseract-ocr-fra=1:4.1.0-2 \
        tesseract-ocr-lat=1:4.1.0-2 \
        tesseract-ocr-eng=1:4.1.0-2 \
        tesseract-ocr-deu=1:4.1.0-2 \
        tesseract-ocr-ita=1:4.1.0-2 \
        tesseract-ocr-spa=1:4.1.0-2 \
        # Bibliothèques image pour Pillow
        libpng16-16 \
        libjpeg62-turbo \
        libtiff6 \
        libwebp7 \
        # Utilitaires
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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

# ── Ports ───────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Démarrage ───────────────────────────────────────────────────
CMD ["picarones", "serve", "--host", "0.0.0.0", "--port", "7860"]
