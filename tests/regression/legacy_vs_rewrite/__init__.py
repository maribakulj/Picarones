"""Harness de régression legacy ↔ rewrite.

Ce package est l'**invariant** qui rend le retrait du legacy
vérifiable.  À chaque phase du plan de retrait
(`docs/migration/legacy-retirement-plan.md`), un fichier
``test_phase<N>_<module>.py`` est ajouté ici qui :

1. Exécute le legacy sur un corpus de référence et capture la sortie
   (la première fois — snapshot golden).
2. Exécute le rewrite sur le même corpus.
3. Compare la sortie rewrite à la golden, à la tolérance ε définie
   dans ``docs/migration/regression-tolerances.md``.

Le harness est **autonome** : pas de dépendance réseau, pas de
binaire système non installable via pip.  Les corpus de référence
vivent dans ``corpora/`` et sont versionnés (synthétiques pour
les small/medium, échantillons figés du domaine public pour large
si jamais ajouté).
"""
