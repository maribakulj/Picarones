# Guide d'installation — Picarones

> Guide détaillé pour Linux, macOS et Windows.
> Pour une installation en 5 minutes : voir [README.md](README.md#installation-rapide).

---

## Sommaire

1. [Prérequis](#1-prérequis)
2. [Installation Linux (Ubuntu/Debian)](#2-installation-linux-ubuntudebian)
3. [Installation macOS](#3-installation-macos)
4. [Installation Windows](#4-installation-windows)
5. [Configuration des moteurs OCR](#5-configuration-des-moteurs-ocr)
6. [Configuration des APIs](#6-configuration-des-apis)
7. [Lancement de l'interface web](#7-lancement-de-linterface-web)
8. [Installation Docker](#8-installation-docker)
9. [Vérification de l'installation](#9-vérification-de-linstallation)
10. [Résolution des problèmes courants](#10-résolution-des-problèmes-courants)

---

## 1. Prérequis

| Composant | Version minimale | Obligatoire |
|-----------|-----------------|-------------|
| Python | 3.11 | Oui |
| pip | 23.0+ | Oui |
| Git | 2.x | Oui (pour cloner) |
| Tesseract | 5.0+ | Pour le moteur Tesseract |
| Pero OCR | 0.1+ | Pour le moteur Pero OCR |
| Docker | 24.x | Pour déploiement containerisé |

---

## 2. Installation Linux (Ubuntu/Debian)

### 2.1 Python et pip

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip git
python3.11 --version   # Vérifier : Python 3.11.x
```

### 2.2 Tesseract OCR

```bash
# Tesseract 5 (PPA pour Ubuntu < 22.04)
sudo add-apt-repository ppa:alex-p/tesseract-ocr5 -y
sudo apt update
sudo apt install tesseract-ocr

# Modèles de langues (choisir selon votre corpus)
sudo apt install tesseract-ocr-fra   # Français
sudo apt install tesseract-ocr-lat   # Latin
sudo apt install tesseract-ocr-eng   # Anglais
sudo apt install tesseract-ocr-deu   # Allemand
sudo apt install tesseract-ocr-ita   # Italien
sudo apt install tesseract-ocr-spa   # Espagnol

# Vérifier
tesseract --version   # Tesseract 5.x.x
tesseract --list-langs
```

### 2.3 Picarones

```bash
git clone https://github.com/maribakulj/Picarones.git
cd picarones

# Créer un environnement virtuel (recommandé)
python3.11 -m venv .venv
source .venv/bin/activate

# Installation de base
pip install -e .

# Installation avec interface web (FastAPI + uvicorn)
pip install -e ".[web]"

# Installation complète (tous les extras)
pip install -e ".[web,hf,dev]"
```

### 2.4 Pero OCR (optionnel)

```bash
# Pero OCR nécessite quelques dépendances système
sudo apt install libgl1 libglib2.0-0

pip install pero-ocr

# Télécharger un modèle pré-entraîné
# Voir https://github.com/DCGM/pero-ocr pour les modèles disponibles
```

---

## 3. Installation macOS

### 3.1 Homebrew (si non installé)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3.2 Python et Tesseract

```bash
brew install python@3.11 tesseract

# Modèles de langues Tesseract
brew install tesseract-lang   # Installe tous les modèles

# Ou modèles individuels via les données de tessdata
# Voir https://github.com/tesseract-ocr/tessdata
```

### 3.3 Picarones

```bash
git clone https://github.com/maribakulj/Picarones.git
cd picarones

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ".[web]"
```

### 3.4 Résolution d'un problème courant macOS

Si `pytesseract` ne trouve pas Tesseract :

```bash
# Trouver le chemin de Tesseract
which tesseract   # Ex : /opt/homebrew/bin/tesseract

# L'indiquer explicitement dans votre script Python :
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
```

Ou définir la variable d'environnement :

```bash
export TESSDATA_PREFIX=/opt/homebrew/share/tessdata/
```

---

## 4. Installation Windows

### 4.1 Python

1. Télécharger Python 3.11+ depuis [python.org](https://www.python.org/downloads/windows/)
2. Cocher "Add Python to PATH" lors de l'installation
3. Vérifier : `python --version` dans PowerShell

### 4.2 Tesseract

1. Télécharger l'installateur depuis [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Choisir la version 5.x (64-bit recommandé)
3. **Pendant l'installation** : cocher les modèles de langues souhaités (Français, Latin…)
4. Ajouter Tesseract au PATH :
   - Chercher "Variables d'environnement" dans le menu Démarrer
   - Ajouter `C:\Program Files\Tesseract-OCR` à la variable `Path`
5. Vérifier : `tesseract --version` dans PowerShell

### 4.3 Git

Télécharger depuis [git-scm.com](https://git-scm.com/download/win) et installer.

### 4.4 Picarones

```powershell
git clone https://github.com/maribakulj/Picarones.git
cd picarones

python -m venv .venv
.venv\Scripts\activate

pip install -e ".[web]"
```

### 4.5 Problème d'encodage Windows

Si vous rencontrez des erreurs d'encodage, définir :

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

Ou dans votre profil PowerShell : `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`

---

## 5. Configuration des moteurs OCR

### 5.1 Tesseract — Configuration avancée

```bash
# Vérifier les modèles installés
tesseract --list-langs

# Tester sur une image
tesseract image.jpg sortie -l fra --psm 6

# Configuration dans Picarones
picarones run --corpus ./corpus/ --engines tesseract --lang fra --psm 6
```

Modes PSM (Page Segmentation Mode) recommandés :

| PSM | Usage |
|-----|-------|
| 6 (défaut) | Bloc de texte uniforme |
| 3 | Détection automatique de la mise en page |
| 11 | Texte épars, sans mise en page |
| 1 | Détection automatique avec OSD |

### 5.2 Pero OCR

```bash
# Télécharger un modèle pré-entraîné (exemple)
mkdir -p ~/.pero/models
# Voir https://github.com/DCGM/pero-ocr/releases

# Configurer via YAML
cat > pero_config.yaml << 'EOF'
name: pero_printed
type: pero_ocr
config_path: /path/to/pero_model/config.yaml
EOF
```

### 5.3 Kraken (optionnel)

```bash
pip install kraken

# Télécharger un modèle
kraken get 10.5281/zenodo.XXXXXXX

# Lister les modèles installés
kraken list
```

### 5.4 Ollama (LLMs locaux)

```bash
# Installer Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Démarrer le service
ollama serve

# Télécharger un modèle
ollama pull llama3
ollama pull gemma2

# Vérifier
ollama list
```

---

## 6. Configuration des APIs

Les clés API sont lues depuis les variables d'environnement. **Ne jamais les écrire dans le code.**

### 6.1 Fichier `.env` (recommandé)

Créer un fichier `.env` à la racine du projet (ajouté au `.gitignore`) :

```bash
# .env — Ne pas commiter ce fichier !

# OpenAI (GPT-4o, GPT-4o mini)
OPENAI_API_KEY=sk-...

# Anthropic (Claude Sonnet, Haiku)
ANTHROPIC_API_KEY=sk-ant-...

# Mistral (Mistral Large, Pixtral, Mistral OCR)
MISTRAL_API_KEY=...

# Google Vision
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# AWS Textract
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=eu-west-1

# Azure Document Intelligence
AZURE_DOC_INTEL_ENDPOINT=https://...cognitiveservices.azure.com/
AZURE_DOC_INTEL_KEY=...
```

Charger avec `python-dotenv` ou directement dans le shell :

```bash
# Linux/macOS
export $(cat .env | grep -v '^#' | xargs)

# Ou avec python-dotenv
pip install python-dotenv
```

### 6.2 Vérification des APIs

```bash
# Tester les APIs configurées
picarones engines   # affiche les moteurs disponibles et leur statut
```

---

## 7. Lancement de l'interface web

```bash
# Installer les dépendances web
pip install -e ".[web]"

# Lancer le serveur (localhost uniquement)
picarones serve

# Ou avec adresse publique (Docker, serveur distant)
picarones serve --host 0.0.0.0 --port 8000

# Mode développement (rechargement automatique)
picarones serve --reload --verbose

# Accéder dans le navigateur
# http://localhost:8000
```

---

## 8. Installation Docker

### 8.1 Utiliser l'image Docker officielle

```bash
# Construire l'image
docker build -t picarones:latest .

# Lancer le service
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v $(pwd)/corpus:/app/corpus \
  picarones:latest

# Accéder dans le navigateur
# http://localhost:8000
```

### 8.2 Docker Compose (Picarones + Ollama)

```bash
# Lancer tous les services
docker compose up -d

# Avec Ollama pour les LLMs locaux
docker compose --profile ollama up -d

# Arrêter
docker compose down
```

Voir [docker-compose.yml](docker-compose.yml) pour la configuration complète.

### 8.3 Variables d'environnement pour Docker

Créer un fichier `.env.docker` :

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
```

```bash
docker compose --env-file .env.docker up -d
```

---

## 9. Vérification de l'installation

```bash
# 1. Version et dépendances
picarones info

# 2. Moteurs disponibles
picarones engines

# 3. Rapport de démonstration (sans moteur OCR réel)
picarones demo --docs 3 --output test_demo.html
# Ouvrir test_demo.html dans un navigateur

# 4. Suivi longitudinal (demo)
picarones history --demo

# 5. Analyse de robustesse (demo)
picarones robustness --corpus . --engine tesseract --demo

# 6. Suite de tests complète
make test
# ou
pytest
```

---

## 10. Résolution des problèmes courants

### `tesseract: command not found`

```bash
# Ubuntu : réinstaller
sudo apt install tesseract-ocr

# macOS : vérifier Homebrew
brew install tesseract

# Windows : vérifier le PATH
where tesseract   # doit retourner un chemin
```

### `Error: No module named 'picarones'`

```bash
# Réinstaller en mode éditable
pip install -e .

# Vérifier l'environnement virtuel actif
which python   # doit pointer vers .venv/bin/python
```

### `pytesseract.pytesseract.TesseractNotFoundError`

```bash
# Linux/macOS : vérifier le PATH
which tesseract

# Windows : vérifier l'installation et le PATH
# Puis dans Python :
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Erreur d'encodage UTF-8 (Windows)

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
```

### Interface web inaccessible

```bash
# Vérifier que le port n'est pas occupé
lsof -i :8000   # Linux/macOS
netstat -ano | findstr :8000   # Windows

# Utiliser un autre port
picarones serve --port 8080
```

### `ImportError: No module named 'fastapi'`

```bash
pip install -e ".[web]"
```

### Tesseract lent sur de grands corpus

```bash
# Augmenter le parallélisme (si votre machine le permet)
picarones run --corpus ./corpus/ --engines tesseract   # traitement séquentiel par défaut
```

---

## Désinstallation

```bash
# Dans l'environnement virtuel
pip uninstall picarones

# Supprimer l'historique SQLite (optionnel)
rm -rf ~/.picarones/

# Supprimer l'environnement virtuel
deactivate
rm -rf .venv/
```
