# Corpus de référence — anti-régression CER (Sprint A5)

Item M-14 de l'audit institutional-readiness-2026-05.

Ce dossier sert de **gardien anti-régression de performance OCR**.
Le workflow [`.github/workflows/perf_regression.yml`](../../../.github/workflows/perf_regression.yml)
le réutilise toutes les semaines (cron) pour vérifier que Tesseract +
Pero OCR ne dérivent pas sur des entrées canoniques.

## Philosophie

- **Synthétique** : les documents sont générés via Pillow à partir
  de texte rendu en typographies courantes. Pas de manuscrit
  authentique embarqué (raisons : licence, taille du repo, indépendance
  vis-à-vis d'un fonds particulier).
- **Représentatif** : 3 strates couvertes (imprimé moderne propre,
  imprimé ancien stylisé, cursive simulée).
- **Reproductible** : graine fixe (`seed=4242` dans `_generate.py`),
  donc deux générations successives produisent des PNG bit-à-bit
  identiques.
- **Tolérance large** : le seuil par défaut est `CER < 15 %` sur
  Tesseract. Pas de finetuning à atteindre — on cherche juste à
  détecter une **régression franche** (CER × 2 du jour au lendemain
  signale qu'un PR a cassé un adapter ou la normalisation).

## Génération

```bash
python -m pytest tests/fixtures/reference_corpus/_generate.py
# (ou directement)
python tests/fixtures/reference_corpus/_generate.py
```

Le script (re)crée :
- `doc_<NN>.png` — image du document
- `doc_<NN>.gt.txt` — vérité terrain associée

## Limites assumées

- **Tesseract** : modèle `eng+fra` standard, OCR sur imprimé moderne
  fonctionne ; sur cursive simulée, le CER attendu est ~30 % et
  c'est le but (vérifie que le pipeline ne crashe pas).
- **Pas de paléographie réelle** : pour des benchmarks scientifiques
  de qualité paléographique, utiliser un corpus HTR-United ou IIIF
  via ``picarones import``.
