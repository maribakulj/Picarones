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
FROM python:3.11-slim AS builder

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
    pip install -e ".[web]" && \
    pip cache purge

# ──────────────────────────────────────────────────────────────────
# Étape 2 : runtime — image finale légère avec Tesseract
# ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL description="Picarones — Plateforme de comparaison de moteurs OCR pour documents patrimoniaux"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/bnf/picarones"
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
    chown -R picarones:picarones /app
USER picarones

# ── Variables d'environnement par défaut ───────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# ── Ports ───────────────────────────────────────────────────────
EXPOSE 8000
EXPOSE 7860

# ── Health check ────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Démarrage ───────────────────────────────────────────────────
CMD ["picarones", "serve", "--host", "0.0.0.0", "--port", "7860"]
