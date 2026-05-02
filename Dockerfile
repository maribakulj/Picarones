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
# Sprint A8 (M-2) — image de base épinglée à un patch stable.
#
# Pourquoi : ``python:3.11-slim`` (sans patch) suit le stream et peut
# changer entre deux ``docker build`` consécutifs. Pour la
# reproductibilité institutionnelle BnF, on épingle au patch précis.
#
# Rotation trimestrielle : avant chaque release majeure, exécuter :
#
#     docker pull python:3.11.13-slim
#     docker inspect python:3.11.13-slim --format='{{index .RepoDigests 0}}'
#     # → mettre à jour DIGEST ci-dessous
#
# Le digest sha256 est volontairement laissé en commentaire plutôt
# qu'en directive ``@sha256:...`` pour éviter un build cassé sur
# les machines de développement qui n'ont pas accès à un registry
# proxy. Un futur sprint dédié au build determinist (post-release v1.2)
# basculera sur la forme ``@sha256:...`` une fois le pipeline release
# stabilisé.
# ──────────────────────────────────────────────────────────────────
ARG PYTHON_BASE_IMAGE=python:3.11.13-slim
# Last verified digest (rotate quarterly):
#   python:3.11.13-slim @ sha256:<obtain via ``docker inspect``>

FROM ${PYTHON_BASE_IMAGE} AS builder

WORKDIR /app

# Dépendances système pour la compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de configuration du package
COPY pyproject.toml .
COPY README.md .
COPY picarones/ picarones/

# Créer un venv isolé et installer Picarones avec les extras web
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install -e ".[web,llm]" && \
    pip cache purge

# ──────────────────────────────────────────────────────────────────
# Étape 2 : runtime — image finale légère avec Tesseract
# ──────────────────────────────────────────────────────────────────
# ARG redéclaré ici car les variables ARG hors ``FROM`` sont scopées
# par étape ; sans cette redéclaration le ``FROM`` du runtime perd
# l'épinglage du builder.
ARG PYTHON_BASE_IMAGE=python:3.11.13-slim
FROM ${PYTHON_BASE_IMAGE} AS runtime

LABEL description="Picarones — Plateforme de comparaison de moteurs OCR pour documents patrimoniaux"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/maribakulj/Picarones"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# ── Dépendances système ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tesseract OCR 5 et modèles de langues
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
    curl \
    && rm -rf /var/lib/apt/lists/*

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
