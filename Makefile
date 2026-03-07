# Makefile — Picarones
# Usage : make <cible>
# Cibles principales : install, test, demo, serve, build, build-exe, docker-build, clean

.PHONY: all install install-dev install-all test test-cov lint demo serve \
        build build-exe docker-build docker-run docker-compose-up clean help

PYTHON     := python3
PIP        := pip
VENV       := .venv
VENV_BIN   := $(VENV)/bin
PICARONES  := $(VENV_BIN)/picarones
PYTEST     := $(VENV_BIN)/pytest
PACKAGE    := picarones

# Couleurs
BOLD  := \033[1m
GREEN := \033[32m
CYAN  := \033[36m
RESET := \033[0m

# ──────────────────────────────────────────────────────────────────
# Aide
# ──────────────────────────────────────────────────────────────────

help: ## Affiche cette aide
	@echo ""
	@echo "$(BOLD)Picarones — Commandes disponibles$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*## "}; {printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2}'
	@echo ""

all: install test  ## Installer et tester

# ──────────────────────────────────────────────────────────────────
# Installation
# ──────────────────────────────────────────────────────────────────

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)  ## Installe Picarones en mode éditable (dépendances de base)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -e .
	@echo "$(GREEN)✓ Installation de base terminée$(RESET)"
	@echo "  Activez l'environnement : source $(VENV)/bin/activate"

install-dev: $(VENV)  ## Installe avec les dépendances de développement (tests, lint)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -e ".[dev]"
	@echo "$(GREEN)✓ Installation dev terminée$(RESET)"

install-web: $(VENV)  ## Installe avec l'interface web (FastAPI + uvicorn)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -e ".[web,dev]"
	@echo "$(GREEN)✓ Installation web terminée$(RESET)"

install-all: $(VENV)  ## Installe avec tous les extras (web, HuggingFace, dev)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -e ".[web,hf,dev]"
	@echo "$(GREEN)✓ Installation complète terminée$(RESET)"

# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────

test:  ## Lance la suite de tests complète
	$(PYTEST) tests/ -q --tb=short
	@echo "$(GREEN)✓ Tests terminés$(RESET)"

test-cov:  ## Tests avec rapport de couverture HTML
	$(PYTEST) tests/ --cov=$(PACKAGE) --cov-report=html --cov-report=term-missing -q
	@echo "$(GREEN)✓ Rapport de couverture : htmlcov/index.html$(RESET)"

test-fast:  ## Tests rapides uniquement (exclut les tests lents)
	$(PYTEST) tests/ -q --tb=short -x

test-sprint9:  ## Tests Sprint 9 uniquement
	$(PYTEST) tests/test_sprint9_packaging.py -v

# ──────────────────────────────────────────────────────────────────
# Qualité du code
# ──────────────────────────────────────────────────────────────────

lint:  ## Vérifie le style du code (ruff si disponible, sinon flake8)
	@if command -v ruff > /dev/null 2>&1; then \
	  ruff check $(PACKAGE)/ tests/; \
	elif $(VENV_BIN)/python -m ruff --version > /dev/null 2>&1; then \
	  $(VENV_BIN)/python -m ruff check $(PACKAGE)/ tests/; \
	elif command -v flake8 > /dev/null 2>&1; then \
	  flake8 $(PACKAGE)/ tests/ --max-line-length=100 --ignore=E501,W503; \
	else \
	  echo "Aucun linter disponible (installez ruff : pip install ruff)"; \
	fi

typecheck:  ## Vérification de types avec mypy (si installé)
	@$(VENV_BIN)/python -m mypy $(PACKAGE)/ --ignore-missing-imports --no-strict-optional 2>/dev/null \
	  || echo "mypy non installé : pip install mypy"

# ──────────────────────────────────────────────────────────────────
# Démonstration
# ──────────────────────────────────────────────────────────────────

demo:  ## Génère un rapport de démonstration complet (rapport_demo.html)
	$(PICARONES) demo --docs 12 --output rapport_demo.html \
	  --with-history --with-robustness
	@echo "$(GREEN)✓ Rapport demo : rapport_demo.html$(RESET)"
	@echo "  Ouvrez : file://$(PWD)/rapport_demo.html"

demo-json:  ## Génère rapport demo + export JSON
	$(PICARONES) demo --docs 12 --output rapport_demo.html --json-output resultats_demo.json
	@echo "$(GREEN)✓ Rapport : rapport_demo.html | JSON : resultats_demo.json$(RESET)"

demo-history:  ## Démonstration du suivi longitudinal
	$(PICARONES) history --demo --regression

demo-robustness:  ## Démonstration de l'analyse de robustesse
	mkdir -p /tmp/picarones_demo_corpus
	$(PICARONES) robustness \
	  --corpus /tmp/picarones_demo_corpus \
	  --engine tesseract \
	  --demo \
	  --degradations noise,blur,rotation

# ──────────────────────────────────────────────────────────────────
# Serveur web
# ──────────────────────────────────────────────────────────────────

serve:  ## Lance l'interface web locale (http://localhost:8000)
	$(PICARONES) serve --host 127.0.0.1 --port 8000

serve-public:  ## Lance le serveur en mode public (0.0.0.0:8000)
	$(PICARONES) serve --host 0.0.0.0 --port 8000

serve-dev:  ## Lance le serveur en mode développement (rechargement automatique)
	$(PICARONES) serve --reload --verbose

# ──────────────────────────────────────────────────────────────────
# Build & packaging
# ──────────────────────────────────────────────────────────────────

build:  ## Construit la distribution Python (wheel + sdist)
	$(VENV_BIN)/pip install --upgrade build
	$(VENV_BIN)/python -m build
	@echo "$(GREEN)✓ Distribution : dist/$(RESET)"

build-exe:  ## Génère un exécutable standalone avec PyInstaller
	@echo "$(CYAN)Construction de l'exécutable standalone…$(RESET)"
	$(VENV_BIN)/pip install pyinstaller
	$(VENV_BIN)/pyinstaller picarones.spec --noconfirm
	@echo "$(GREEN)✓ Exécutable : dist/picarones/$(RESET)"

build-exe-onefile:  ## Génère un exécutable unique (plus lent au démarrage)
	$(VENV_BIN)/pip install pyinstaller
	$(VENV_BIN)/pyinstaller picarones.spec --noconfirm --onefile
	@echo "$(GREEN)✓ Exécutable : dist/picarones$(RESET)"

# ──────────────────────────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────────────────────────

docker-build:  ## Construit l'image Docker Picarones
	docker build -t picarones:latest -t picarones:1.0.0 .
	@echo "$(GREEN)✓ Image Docker : picarones:latest$(RESET)"

docker-run:  ## Lance Picarones dans Docker (http://localhost:8000)
	docker run --rm -p 8000:8000 \
	  -e OPENAI_API_KEY="$${OPENAI_API_KEY:-}" \
	  -e ANTHROPIC_API_KEY="$${ANTHROPIC_API_KEY:-}" \
	  -e MISTRAL_API_KEY="$${MISTRAL_API_KEY:-}" \
	  -v "$(PWD)/corpus:/app/corpus:ro" \
	  picarones:latest

docker-compose-up:  ## Lance Picarones + Ollama avec Docker Compose
	docker compose up -d
	@echo "$(GREEN)✓ Services démarrés$(RESET)"
	@echo "  Picarones : http://localhost:8000"
	@echo "  Ollama    : http://localhost:11434"

docker-compose-down:  ## Arrête les services Docker Compose
	docker compose down

docker-compose-logs:  ## Affiche les logs Docker Compose
	docker compose logs -f picarones

# ──────────────────────────────────────────────────────────────────
# Nettoyage
# ──────────────────────────────────────────────────────────────────

clean:  ## Supprime les fichiers générés (cache, build, dist)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .eggs/ htmlcov/ .coverage .pytest_cache/
	@echo "$(GREEN)✓ Nettoyage terminé$(RESET)"

clean-all: clean  ## Supprime aussi l'environnement virtuel
	rm -rf $(VENV)/
	@echo "$(GREEN)✓ Environnement virtuel supprimé$(RESET)"

# ──────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────

info:  ## Affiche les informations de version Picarones
	$(PICARONES) info

engines:  ## Liste les moteurs OCR disponibles
	$(PICARONES) engines

history-demo:  ## Affiche l'historique de démonstration
	$(PICARONES) history --demo --regression

changelog:  ## Affiche le CHANGELOG
	@cat CHANGELOG.md | head -80

version:  ## Affiche la version courante
	@grep -m1 'version' pyproject.toml | awk '{print $$3}' | tr -d '"'
