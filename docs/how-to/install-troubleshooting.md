# Résolution des problèmes d'installation

Recueil des erreurs courantes rencontrées lors de l'installation de
Picarones. Pour l'installation initiale, voir
[`install.md`](install.md).

---

## `tesseract: command not found`

```bash
# Ubuntu : réinstaller
sudo apt install tesseract-ocr

# macOS : vérifier Homebrew
brew install tesseract

# Windows : vérifier le PATH
where tesseract   # doit retourner un chemin
```

## `Error: No module named 'picarones'`

```bash
# Réinstaller en mode éditable
pip install -e .

# Vérifier l'environnement virtuel actif
which python   # doit pointer vers .venv/bin/python
```

## `pytesseract.pytesseract.TesseractNotFoundError`

```bash
# Linux/macOS : vérifier le PATH
which tesseract

# Windows : vérifier l'installation et le PATH
# Puis dans Python :
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Erreur d'encodage UTF-8 (Windows)

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
```

## Interface web inaccessible

```bash
# Vérifier que le port n'est pas occupé
lsof -i :8000   # Linux/macOS
netstat -ano | findstr :8000   # Windows

# Utiliser un autre port
picarones serve --port 8080
```

## `ImportError: No module named 'fastapi'`

```bash
pip install -e ".[web]"
```

## Tesseract lent sur de grands corpus

```bash
# Augmenter le parallélisme (si votre machine le permet)
picarones run --corpus ./corpus/ --engines tesseract   # traitement séquentiel par défaut
```

---

## Pour aller plus loin

Si votre problème n'est pas listé ici :

- Consultez le [runbook opérationnel](../operations/runbook.md) pour
  les incidents en production.
- Ouvrez une issue GitHub avec les sorties de `picarones info` et
  `picarones engines`.
