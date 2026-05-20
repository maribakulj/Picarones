"""``MetricSpec``, ``EvaluationView``, ``EvaluationSpec``

Cœur de la valeur ajoutée du rewrite : **comparer librement des
pipelines hétérogènes en projetant leurs sorties vers une vue
d'évaluation explicite**.  L'utilisateur ne compare jamais
directement un OCR brut et une sortie ALTO reconstruite ; il
compare leur projection dans une vue commune (texte, ALTO,
recherchabilité, ...) et le rapport explicite ce que la vue
ignore.

Trois couches de contrat :

- ``MetricSpec`` — déclare une métrique (nom + signature de types).
- ``EvaluationView`` — déclare une vue (sélecteur de candidats +
  projection optionnelle + liste de métriques + dimensions
  ignorées).
- ``EvaluationSpec`` — container de N vues qu'un benchmark applique.

Différence avec l'existant ``core/metric_registry.py:MetricSpec``
-----------------------------------------------------------------
L'ancien ``MetricSpec`` (Sprint 34) porte un ``func: Callable``,
un singleton global ``_METRIC_REGISTRY``, et un décorateur
``@register_metric`` qui s'exécute par effet de bord d'import.
C'est exactement l'anti-pattern que le rewrite cherche à bannir
(cf. ``BACKLOG_POST_LIVRAISON.md`` §2.4 + tests d'architecture du
S3).

Le nouveau ``MetricSpec`` est purement **déclaratif** : pas de
callable.  L'association ``MetricSpec ↔ Callable`` se fait
explicitement dans ``picarones.evaluation.registry.MetricRegistry``
qu'un service applicatif construit au démarrage (S20).

Anti-sur-ingénierie
-------------------
Pas de validation cross-références à l'instanciation d'un
``EvaluationView`` (par exemple, on ne vérifie pas que les
``metric_names`` existent dans un registre).  Cette validation
est faite au moment de l'exécution par ``EvaluationViewExecutor``
(S13), avec un message d'erreur explicite si une métrique
référencée n'est pas enregistrée.  Raison : un ``EvaluationView``
est un objet déclaratif qu'on peut sérialiser dans un YAML sans
avoir besoin du registre runtime.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from picarones.domain.artifacts import ArtifactType
from picarones.domain.projection_spec import ProjectionSpec


class MetricSpec(BaseModel):
    """Description déclarative d'une métrique enregistrable.

    Attributs
    ---------
    name:
        Identifiant unique dans un ``MetricRegistry``.
    input_types:
        Tuple ``(reference_type, hypothesis_type)`` indiquant la
        signature attendue par la métrique.  Le registre sélectionne
        les métriques applicables à une jonction par cette signature.
    description:
        Phrase courte affichée dans le rapport et le glossaire.
    higher_is_better:
        ``True`` pour les métriques de qualité (F1, recall, accuracy),
        ``False`` pour les métriques d'erreur (CER, WER).  Utilisé
        par les vues pour orienter la coloration et le tri.
    tags:
        Étiquettes libres pour grouper les métriques (``"text"``,
        ``"structure"``, ``"icdar"``, ``"philological"``, ...).

    Contrairement à l'ancien ``core.metric_registry.MetricSpec``,
    aucun ``func: Callable`` n'est porté ici — un ``MetricSpec``
    est purement déclaratif et peut être chargé depuis un YAML.
    L'association nom → fonction est faite par ``MetricRegistry``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    input_types: tuple[ArtifactType, ArtifactType]
    description: str = ""
    higher_is_better: bool = False
    tags: frozenset[str] = Field(default_factory=frozenset)


class EvaluationView(BaseModel):
    """Une vue d'évaluation = une "lentille" pour comparer des pipelines.

    Une vue répond à une question précise : "lequel des pipelines
    disponibles produit la meilleure sortie sous cet angle ?"

    Trois exemples canoniques (à implémenter S14-S16) :

    - ``TextView`` (text_final) — accepte RAW_TEXT, CORRECTED_TEXT,
      ALTO_XML, PAGE_XML, projette tout vers RAW_TEXT, mesure CER/WER.
      Ignore : géométrie, blocs, ordre spatial, validité ALTO.
    - ``AltoView`` (alto_documentary) — exige ALTO_XML, mesure
      validité, alignement lignes/mots, ordre de lecture.  Ignore :
      qualité linguistique pure.
    - ``SearchView`` (searchability) — projette tout vers RAW_TEXT,
      mesure recall fuzzy, séquences numériques préservées, noms
      propres retrouvés.

    Attributs
    ---------
    name:
        Identifiant lisible (``"text_final"``, ``"alto_documentary"``).
    description:
        Phrase d'introduction affichée dans le rapport.
    candidate_types:
        Set des ``ArtifactType`` qu'on accepte en entrée.  Un pipeline
        ne produisant aucun artefact dans ce set est **omis
        explicitement** de la vue (pas de score factice).
    projection:
        Spec optionnelle de projection à appliquer aux candidats avant
        évaluation.  ``None`` = pas de projection (l'artefact est
        comparé tel quel au GT).
    normalization_profile:
        Nom d'un profil de normalisation texte
        (cf. ``picarones.formats.text.normalization``).  ``None`` =
        pas de normalisation (NFC implicite).
    metric_names:
        Liste ordonnée des métriques à calculer.  Validées par
        l'executor au runtime (le registre doit contenir chaque nom).
    ignored_dimensions:
        Liste de dimensions explicitement ignorées par cette vue.
        Affiché dans le rapport pour signaler ce que la comparaison
        ne dit PAS.  Ex : ``("geometry", "block_structure",
        "reading_order")`` pour TextView.
    warnings:
        Avertissement(s) méthodologique(s) à afficher en tête du
        bloc de la vue dans le rapport.  Ex : "Cette vue ignore la
        qualité spatiale et documentaire."
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    candidate_types: frozenset[ArtifactType] = Field(...)
    projection: ProjectionSpec | None = None
    """Projection unique appliquée à TOUS les candidats avant
    évaluation.  ``None`` = pas de projection (artefact comparé
    tel quel).  Si ``projections_by_source_type`` est aussi
    renseigné, ce champ sert de fallback pour les types non listés."""
    projections_by_source_type: dict[ArtifactType, ProjectionSpec] = Field(
        default_factory=dict,
    )
    """S14 — projection conditionnelle par type d'artefact source.

    Permet à une vue qui accepte plusieurs types (ex : ``TextView``
    qui accepte RAW_TEXT, ALTO_XML, PAGE_XML) d'utiliser un
    projecteur différent par type sans avoir à dupliquer la vue.

    Convention de résolution dans ``DefaultEvaluationViewExecutor`` :

    1. Si ``projections_by_source_type[candidate.type]`` existe :
       utiliser cette projection.
    2. Sinon, si ``projection`` est défini ET son ``source_type``
       matche ``candidate.type`` : utiliser cette projection.
    3. Sinon : pas de projection (artefact comparé tel quel).

    Toutes les projections référencées doivent exister dans le
    ``ProjectorRegistry`` au moment de l'exécution (validé runtime).
    """
    normalization_profile: str | None = Field(default=None, max_length=128)
    char_exclude: str | None = Field(default=None, max_length=512)
    """Phase B2.5 — caractères à filtrer des payloads texte avant
    évaluation.  Appliqué APRÈS ``normalization_profile``.  Cohérent
    sémantiquement avec le paramètre éponyme de
    ``compute_metrics`` (legacy)."""
    metric_names: tuple[str, ...] = Field(default_factory=tuple)
    ignored_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    def accepts(self, artifact_type: ArtifactType) -> bool:
        """Vrai si cette vue peut consommer un artefact du type donné."""
        return artifact_type in self.candidate_types

    def projection_for(
        self, source_type: ArtifactType,
    ) -> ProjectionSpec | None:
        """Retourne la projection à appliquer pour un artefact source
        de type ``source_type``, ou ``None`` si aucune projection n'est
        applicable (artefact comparé tel quel).

        Convention de résolution :

        1. ``projections_by_source_type[source_type]`` si présent.
        2. ``projection`` si son ``source_type`` matche.
        3. ``None``.
        """
        if source_type in self.projections_by_source_type:
            return self.projections_by_source_type[source_type]
        if (
            self.projection is not None
            and self.projection.source_type == source_type
        ):
            return self.projection
        return None


class EvaluationSpec(BaseModel):
    """Container de N ``EvaluationView`` qu'un benchmark applique.

    Un ``EvaluationSpec`` est versionné dans un YAML ; un service
    applicatif (S19) le résout en runtime contre un ``MetricRegistry``
    instancié, et le ``EvaluationViewExecutor`` (S13) l'applique aux
    artefacts produits par le pipeline executor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    views: tuple[EvaluationView, ...] = Field(default_factory=tuple)

    def view_by_name(self, name: str) -> EvaluationView | None:
        """Retourne la vue de nom ``name`` ou ``None``."""
        for v in self.views:
            if v.name == name:
                return v
        return None


__all__ = ["MetricSpec", "EvaluationView", "EvaluationSpec"]
