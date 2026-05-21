# Premier benchmark Picarones

Ce tutoriel guide un nouvel utilisateur — chercheur, archiviste,
conservateur — à travers son **premier benchmark OCR** complet, de
l'installation jusqu'à la lecture du rapport produit. Comptez 15
minutes pour la première fois, 2 minutes une fois familier.

> **Pré-requis** : Python 3.11+ et `pip`. Sur Linux, le binaire
> `tesseract` est nécessaire pour le moteur OCR par défaut
> (`apt-get install tesseract-ocr tesseract-ocr-fra` sur Debian/Ubuntu).

---

## 1. Installation

```bash
pip install -e ".[dev,web]"
```

L'extra `dev` apporte la suite de tests, `web` apporte l'interface
FastAPI (utile dès la deuxième session). Pour une installation
minimale en production, voir [`how-to/install.md`](../how-to/install.md).

Vérifiez :

```bash
picarones info
picarones engines
```

Si `picarones engines` liste au moins `tesseract`, vous êtes prêt.

---

## 2. Générer un rapport de démonstration

Le mode `demo` produit un rapport HTML synthétique sans aucun moteur
installé. C'est le moyen le plus rapide de voir ce que Picarones
produit.

```bash
picarones demo --output rapport_demo.html
```

Ouvrez `rapport_demo.html` dans un navigateur. Vous obtenez un
rapport complet avec :

- agrégat CER/WER global ;
- diff caractère à caractère sur les documents ;
- diagramme CD (Critical Difference) si plus de 2 moteurs ;
- moteur narratif qui résume les faits saillants en prose.

Voir [`reading-a-report.md`](reading-a-report.md) pour la lecture
détaillée.

---

## 3. Benchmark sur un vrai corpus

Préparez un dossier `mon_corpus/` qui contient :

```
mon_corpus/
├── doc1.jpg
├── doc1.gt.txt          # transcription de référence
├── doc2.jpg
└── doc2.gt.txt
```

Le format des transcriptions de référence est documenté dans
[`reference/normalization-profiles.md`](../reference/normalization-profiles.md).

Lancez le benchmark :

```bash
picarones run \
  --corpus mon_corpus/ \
  --engines tesseract \
  --output rapport.html \
  --json rapport.json
```

`rapport.html` contient le rendu visuel ; `rapport.json` contient
l'agrégat machine-lisible (utile pour CI ou comparaisons
longitudinales — voir
[`reference/reproducibility-snapshots.md`](../reference/reproducibility-snapshots.md)).

---

## 4. Comparer plusieurs moteurs

```bash
picarones run \
  --corpus mon_corpus/ \
  --engines tesseract,pero_ocr,mistral_ocr \
  --output comparaison.html
```

Le rapport affiche désormais :

- une ligne par moteur avec CER moyen + IC95 ;
- le diagramme CD (qui domine statistiquement qui) ;
- les diffs côte à côte ;
- les coûts (si moteurs cloud).

Le moteur narratif énonce les écarts significatifs, ne désigne
jamais un « gagnant ».

---

## 5. Interface web (optionnelle)

```bash
picarones serve --port 7860
```

Ouvre `http://localhost:7860`. L'interface permet d'upload un ZIP
de corpus et de lancer un benchmark interactif. Pour le déploiement
institutionnel, voir
[`operations/deployment-institutional.md`](../operations/deployment-institutional.md).

---

## Étapes suivantes

- Comprendre les métriques :
  [`reference/views.md`](../reference/views.md),
  [`reference/normalization-profiles.md`](../reference/normalization-profiles.md)
- Lire un rapport en détail :
  [`reading-a-report.md`](reading-a-report.md)
- Écrire un module pour la pipeline :
  [`writing-a-pipeline-module.md`](writing-a-pipeline-module.md)
- Étudier des cas d'usage :
  [`case-studies/`](../case-studies/)
