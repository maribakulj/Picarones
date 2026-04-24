# 02 — Édition critique d'un manuscrit médiéval unique

> 🎓 **Cas d'école** — scénario illustratif. Le corpus, les chiffres et
> l'institution sont fictifs mais conçus pour être réalistes.

## Contexte

| | |
|---|---|
| Institution | Laboratoire de philologie médiévale d'une université française |
| Projet | Édition critique numérique d'un manuscrit unique de 180 folios |
| Datation | Première moitié du XIIIᵉ siècle |
| Langue | Latin médiéval, glossé en ancien français en marge |
| Écriture | Textualis bookhand, abréviations denses (Knight 1973 : ~30 % du texte) |
| Disposition | 2 colonnes, notes interlinéaires, manchettes |
| GT disponible | 30 folios annotés ligne par ligne par 2 paléographes |
| Conventions | Édition diplomatique stricte (préservation `ſ`, abréviations notées `⟨⟩`) |
| Public visé | Médiévistes et historiens — recherche par formes diplomatiques exactes |

## Question

> Quel pipeline pour produire une transcription **diplomatique exacte**
> publiable sous forme d'édition critique TEI, où chaque expansion
> d'abréviation doit être marquée et où l'ordre de lecture des deux
> colonnes doit être préservé ?

## Métriques regardées en priorité

L'équipe a considéré ces métriques dans cet ordre, en utilisant le
glossaire embarqué pour valider sa compréhension :

1. **CER diplomatique** (et non CER exact) — `pero_ocr` était le seul
   moteur HTR à préserver correctement les `ſ` longs et les `u`/`v`. Le
   CER diplomatique de `pero_ocr` était à 5,1 % vs un CER "exact" à
   12,3 % — mais c'est le diplomatique qui compte pour l'usage métier.
2. **Score de ligatures** — référence MUFI requise. `pero_ocr` 0,82,
   `kraken` 0,71, `tesseract` 0,12.
3. **Erreurs d'abréviation (taxonomie classe 5)** — critique sur ce
   corpus. `pero_ocr` faisait 18 % d'erreurs sur les abréviations
   médiévales classiques (`ꝑ`, `ꝓ`, `ꝗ`, `ꝱ`).
4. **Pipeline `pero_ocr → claude-sonnet-4-6`** en post-correction image+texte
   — testé spécifiquement parce que le LLM peut **silencieusement développer
   les abréviations**. Le détecteur narratif `llm_hallucination_flag` a
   d'ailleurs alerté : « Signal d'hallucination sur claude-sonnet-4-6
   (sortie anormalement longue, ratio 1.31). »
5. **Score de structure / ordre de lecture** — pour vérifier que les deux
   colonnes étaient correctement séparées. `pero_ocr` LCS = 0,94,
   `tesseract` 0,52 (mélangeait les colonnes).

## Métriques **non** regardées

- Le coût (le projet est financé sur fonds ANR, non au volume).
- La vitesse (180 folios traités une seule fois sur 6 mois).
- Le carbone (le calcul se fait sur cluster universitaire, déjà comptabilisé
  en bilan annuel de l'unité).

## Verdict

**Pipeline retenu** : `pero_ocr` seul, **sans LLM en aval**.

**Arguments** :
- CER diplomatique au plafond humain : l'IC 95 % de `pero_ocr` (4,8 %–5,4 %)
  recouvrait l'accord inter-annotateur des deux paléographes (4,7 %).
- Score de ligatures et structure préservés.
- **Refus du LLM** parce que la moindre expansion silencieuse d'abréviation
  obligerait à re-vérifier chaque ligne. Le détecteur d'hallucination du
  rapport a confirmé que `claude-sonnet-4-6` produisait du texte plus long
  que la GT (ratio 1,31), incompatible avec l'édition diplomatique.
- Les erreurs résiduelles de `pero_ocr` (à 5,1 % CER diplomatique) sont
  corrigées manuellement par les paléographes lors de la validation TEI.

## Limites

- 30 folios annotés sur 180 = 17 % du corpus. La validation finale se
  fait à la main, donc la généralisation des chiffres est secondaire.
- L'absence de stratification par main de scribe : tout le manuscrit est
  d'une seule main. Les résultats peuvent ne pas se transposer à d'autres
  manuscrits.
- Pas de mesure structurelle au-delà du LCS — pas de F1 SegmOnto faute
  d'annotation région-par-région. Sera ajouté en Phase 2 (plan rapport).

## Reproductibilité

```yaml
# picarones-config.yml
corpus: ./benchmarks/manuscrit-XIII-30folios/
engines:
  - pero_ocr: { model: medieval-latin-2024 }
  - kraken: { model: textualis-medieval-2023 }
  - tesseract: { lang: lat, psm: 6 }
  - pipeline:
      ocr: pero_ocr
      llm: claude-sonnet-4-6
      mode: post_correction_image_texte
      prompt: correction_image_medieval_french.txt
normalization: medieval_latin
report:
  lang: fr
  output: rapport-mss-XIII.html
```
