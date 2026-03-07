# Guide de contribution — Picarones

Merci de votre intérêt pour Picarones ! Ce guide explique comment contribuer au projet.

---

## Sommaire

1. [Démarrage rapide](#1-démarrage-rapide)
2. [Ajouter un moteur OCR](#2-ajouter-un-moteur-ocr)
3. [Ajouter un adaptateur LLM](#3-ajouter-un-adaptateur-llm)
4. [Ajouter une source d'import](#4-ajouter-une-source-dimport)
5. [Écrire des tests](#5-écrire-des-tests)
6. [Soumettre une Pull Request](#6-soumettre-une-pull-request)
7. [Conventions de code](#7-conventions-de-code)

---

## 1. Démarrage rapide

```bash
# Forker le dépôt sur GitHub, puis :
git clone https://github.com/VOTRE_USERNAME/picarones.git
cd picarones

# Environnement de développement
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,web]"

# Vérifier que tout passe
make test
# ou : pytest

# Créer une branche de travail
git checkout -b feat/mon-nouveau-moteur
```

---

## 2. Ajouter un moteur OCR

Ajouter un nouveau moteur OCR nécessite de créer **un seul fichier Python** et de modifier
deux fichiers de configuration. Pas de refactoring du reste du code.

### 2.1 Créer l'adaptateur

Créer `picarones/engines/mon_moteur.py` en héritant de `BaseOCREngine` :

```python
"""Adaptateur pour Mon Moteur OCR.

Installation :
    pip install mon-moteur

Configuration :
    config:
        model: mon_modele_v2
        lang: fra
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from picarones.engines.base import BaseOCREngine

logger = logging.getLogger(__name__)


class MonMoteurEngine(BaseOCREngine):
    """Adaptateur pour Mon Moteur OCR.

    Args:
        config: Dictionnaire de configuration.
            - ``model`` (str): Identifiant du modèle. Défaut: ``"default"``.
            - ``lang`` (str): Code langue. Défaut: ``"fra"``.
    """

    name = "mon_moteur"

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config or {})
        self.model = self.config.get("model", "default")
        self.lang = self.config.get("lang", "fra")

    def get_version(self) -> str:
        """Retourne la version du moteur."""
        try:
            import mon_moteur
            return getattr(mon_moteur, "__version__", "inconnu")
        except ImportError:
            return "non installé"

    def process_image(self, image_path: str) -> str:
        """Transcrit une image et retourne le texte.

        Args:
            image_path: Chemin absolu vers l'image (JPEG, PNG, TIFF…).

        Returns:
            Texte transcrit par le moteur.

        Raises:
            RuntimeError: Si le moteur n'est pas installé ou si la transcription échoue.
        """
        try:
            import mon_moteur
        except ImportError as exc:
            raise RuntimeError(
                "mon-moteur n'est pas installé. Installez-le avec : pip install mon-moteur"
            ) from exc

        try:
            result = mon_moteur.transcribe(
                image_path,
                model=self.model,
                lang=self.lang,
            )
            return result.text.strip()
        except Exception as exc:
            raise RuntimeError(f"Erreur de transcription : {exc}") from exc
```

### 2.2 Enregistrer le moteur dans le CLI

Dans `picarones/cli.py`, modifier la fonction `_engine_from_name()` :

```python
def _engine_from_name(engine_name: str, lang: str, psm: int) -> "BaseOCREngine":
    from picarones.engines.tesseract import TesseractEngine
    if engine_name in {"tesseract", "tess"}:
        return TesseractEngine(config={"lang": lang, "psm": psm})

    # ↓ Ajouter ici
    try:
        from picarones.engines.mon_moteur import MonMoteurEngine
        if engine_name in {"mon_moteur", "monmoteur"}:
            return MonMoteurEngine(config={"lang": lang})
    except ImportError:
        pass
    # ↑

    raise click.BadParameter(...)
```

### 2.3 Ajouter dans la liste `picarones engines`

Dans `picarones/cli.py`, dans la fonction `engines_cmd()` :

```python
engines = [
    ("tesseract", "Tesseract 5 (pytesseract)", "pytesseract"),
    ("pero_ocr", "Pero OCR", "pero_ocr"),
    ("mon_moteur", "Mon Moteur OCR", "mon_moteur"),  # ← Ajouter
]
```

### 2.4 Ajouter l'extra dans `pyproject.toml` (optionnel)

```toml
[project.optional-dependencies]
mon-moteur = ["mon-moteur>=1.0.0"]
```

### 2.5 Écrire les tests

Créer `tests/test_mon_moteur.py` :

```python
"""Tests pour l'adaptateur Mon Moteur OCR."""

import pytest
from unittest.mock import patch


class TestMonMoteurEngine:

    def test_name(self):
        from picarones.engines.mon_moteur import MonMoteurEngine
        engine = MonMoteurEngine()
        assert engine.name == "mon_moteur"

    def test_process_image_mock(self):
        from picarones.engines.mon_moteur import MonMoteurEngine
        engine = MonMoteurEngine(config={"lang": "fra"})
        mock_result = type("R", (), {"text": "Texte transcrit"})()
        with patch("mon_moteur.transcribe", return_value=mock_result):
            text = engine.process_image("/tmp/test.jpg")
            assert text == "Texte transcrit"

    def test_process_image_import_error(self):
        from picarones.engines.mon_moteur import MonMoteurEngine
        engine = MonMoteurEngine()
        with patch.dict("sys.modules", {"mon_moteur": None}):
            with pytest.raises(RuntimeError, match="non installé"):
                engine.process_image("/tmp/test.jpg")
```

---

## 3. Ajouter un adaptateur LLM

Les adaptateurs LLM sont dans `picarones/llm/`. Créer `picarones/llm/mon_llm_adapter.py` :

```python
"""Adaptateur pour Mon LLM.

Supporte les modes : text_only, text_and_image, zero_shot.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

from picarones.llm.base import BaseLLMAdapter

logger = logging.getLogger(__name__)


class MonLLMAdapter(BaseLLMAdapter):
    """Adaptateur pour Mon LLM.

    Args:
        config: Configuration.
            - ``model`` (str): Modèle à utiliser.
            - ``api_key`` (str): Clé API (peut aussi être dans ``MON_LLM_API_KEY``).
            - ``temperature`` (float): Température (0.0 à 1.0). Défaut: 0.0.
            - ``max_tokens`` (int): Nombre maximum de tokens. Défaut: 4096.
    """

    name = "mon_llm"

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config or {})
        import os
        self.api_key = self.config.get("api_key") or os.getenv("MON_LLM_API_KEY", "")
        self.model = self.config.get("model", "mon-modele-v1")
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 4096))

    def correct_text(self, ocr_text: str, prompt: str) -> str:
        """Corrige le texte OCR en mode texte seul (Mode 1).

        Args:
            ocr_text: Sortie brute du moteur OCR à corriger.
            prompt: Prompt de correction.

        Returns:
            Texte corrigé par le LLM.
        """
        # Implémenter l'appel API ici
        full_prompt = prompt.replace("{ocr_output}", ocr_text)
        return self._call_api(messages=[{"role": "user", "content": full_prompt}])

    def correct_with_image(self, ocr_text: str, image_path: str, prompt: str) -> str:
        """Corrige le texte OCR avec l'image (Mode 2).

        Args:
            ocr_text: Sortie brute du moteur OCR.
            image_path: Chemin vers l'image originale.
            prompt: Prompt de correction.

        Returns:
            Texte corrigé.
        """
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        # Implémenter selon l'API de votre LLM
        return self._call_api_with_image(ocr_text, image_b64, prompt)

    def transcribe_image(self, image_path: str, prompt: str) -> str:
        """Transcription zero-shot depuis l'image seule (Mode 3).

        Args:
            image_path: Chemin vers l'image.
            prompt: Prompt de transcription.

        Returns:
            Transcription produite par le LLM.
        """
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        return self._call_api_with_image("", image_b64, prompt)

    def _call_api(self, messages: list[dict]) -> str:
        """Appel API générique."""
        raise NotImplementedError("Implémenter _call_api()")

    def _call_api_with_image(self, text: str, image_b64: str, prompt: str) -> str:
        """Appel API avec image."""
        raise NotImplementedError("Implémenter _call_api_with_image()")
```

---

## 4. Ajouter une source d'import

Les importeurs sont dans `picarones/importers/`. Voir `iiif.py` et `gallica.py` comme exemples.

Votre importeur doit retourner un objet `Corpus` de `picarones.core.corpus` :

```python
from picarones.core.corpus import Corpus, Document

def import_from_ma_source(url: str, output_dir: str) -> Corpus:
    documents = []
    # ... télécharger et préparer les documents ...
    for img_path, gt_text in zip(images, ground_truths):
        documents.append(Document(
            doc_id=Path(img_path).stem,
            image_path=str(img_path),
            ground_truth=gt_text,
            metadata={"source": "ma_source"},
        ))
    return Corpus(
        name="Corpus depuis Ma Source",
        source=url,
        documents=documents,
    )
```

Ajouter la nouvelle commande dans `picarones/cli.py` (sous-commande de `picarones import`).

---

## 5. Écrire des tests

### Conventions

- Un fichier de test par module/sprint : `tests/test_mon_module.py`
- Classes de test groupées par fonctionnalité : `class TestMonModule:`
- Mocker les appels réseau et les moteurs OCR avec `unittest.mock.patch`
- Viser **100% de couverture** sur les modules publics

### Structure recommandée

```python
"""Tests pour MonModule.

Classes
-------
TestFonctionnalite1    (N tests) — description
TestFonctionnalite2    (M tests) — description
"""

from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock


class TestFonctionnalite1:

    def test_cas_nominal(self):
        from picarones.mon_module import ma_fonction
        result = ma_fonction("entrée")
        assert result == "sortie attendue"

    def test_cas_erreur(self):
        from picarones.mon_module import ma_fonction
        with pytest.raises(ValueError, match="message d'erreur"):
            ma_fonction(None)

    def test_avec_mock(self):
        from picarones.mon_module import MonClient
        client = MonClient("https://example.org", token="tok")
        with patch.object(client, "_fetch", return_value=b"réponse"):
            result = client.appel_api()
            assert result is not None
```

### Lancer les tests

```bash
# Tous les tests
make test
# ou
pytest

# Un fichier spécifique
pytest tests/test_mon_module.py -v

# Avec couverture
pytest --cov=picarones --cov-report=html
open htmlcov/index.html

# Tests rapides (sans les tests lents)
pytest -m "not slow"
```

---

## 6. Soumettre une Pull Request

### Avant de soumettre

```bash
# 1. Vérifier que tous les tests passent
make test

# 2. Vérifier le style de code (si ruff/flake8 disponible)
make lint

# 3. Mettre à jour le CHANGELOG.md

# 4. Pousser votre branche
git push origin feat/mon-nouveau-moteur
```

### Checklist PR

- [ ] Tests unitaires pour toutes les nouvelles fonctions publiques
- [ ] Docstrings Google style sur les classes et méthodes publiques
- [ ] CHANGELOG.md mis à jour dans la section `[Unreleased]`
- [ ] Pas de régression sur la suite de tests existante (`pytest` passe en vert)
- [ ] Code compatible Python 3.11 et 3.12
- [ ] Pas de clés API en dur dans le code

### Description de PR

```markdown
## Résumé
- Ajout de l'adaptateur pour Mon Moteur OCR
- Support des langues latin et français

## Tests
- 15 tests unitaires dans `tests/test_mon_moteur.py`
- Mocké avec `unittest.mock.patch` (pas de dépendance externe requise pour les tests)

## Changements
- `picarones/engines/mon_moteur.py` : nouvel adaptateur
- `picarones/cli.py` : enregistrement du moteur
- `pyproject.toml` : extra `[mon-moteur]`
```

---

## 7. Conventions de code

### Style

- **Python 3.11+** avec annotations de type
- `from __future__ import annotations` en tête de fichier
- Format : PEP 8, lignes ≤ 100 caractères (pas de formatage automatique imposé)

### Docstrings — format Google

```python
def compute_cer(reference: str, hypothesis: str) -> float:
    """Calcule le Character Error Rate (CER) entre référence et hypothèse.

    Le CER est défini comme la distance de Levenshtein au niveau caractère
    divisée par la longueur de la référence.

    Args:
        reference: Texte de vérité terrain (GT).
        hypothesis: Texte produit par le moteur OCR.

    Returns:
        CER entre 0.0 (parfait) et 1.0+ (nombreuses erreurs).

    Raises:
        ValueError: Si ``reference`` est vide.

    Examples:
        >>> compute_cer("bonjour", "bnjour")
        0.14285714285714285
    """
```

### Nommage

- Classes : `PascalCase` (ex : `TesseractEngine`, `GallicaClient`)
- Fonctions/méthodes : `snake_case` (ex : `compute_metrics`, `list_projects`)
- Constantes : `UPPER_SNAKE_CASE` (ex : `DEGRADATION_LEVELS`)
- Fichiers de module : `snake_case.py` (ex : `gallica.py`, `char_scores.py`)

### Gestion des imports optionnels

```python
# Pattern recommandé pour les dépendances optionnelles
def process_image(self, image_path: str) -> str:
    try:
        import mon_moteur
    except ImportError as exc:
        raise RuntimeError(
            "mon-moteur n'est pas installé. Installez-le avec : pip install mon-moteur"
        ) from exc
    # utiliser mon_moteur...
```

### Variables d'environnement pour les clés API

```python
import os

api_key = config.get("api_key") or os.getenv("MON_API_KEY", "")
if not api_key:
    raise RuntimeError(
        "Clé API manquante. Définissez MON_API_KEY ou passez api_key dans la config."
    )
```

---

## Licence

En contribuant à Picarones, vous acceptez que votre contribution soit distribuée
sous licence Apache 2.0.
