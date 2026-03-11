"""Suivi longitudinal des benchmarks — base SQLite optionnelle.

Fonctionnement
--------------
- Chaque run de benchmark est enregistré dans une table SQLite avec horodatage,
  corpus, moteurs, métriques agrégées.
- L'historique permet de tracer des courbes d'évolution du CER dans le temps.
- La détection de régression compare le dernier run à une baseline configurable.

Structure de la base
--------------------
Table ``runs`` :
    run_id      TEXT PRIMARY KEY  — UUID ou hash du run
    timestamp   TEXT              — ISO 8601
    corpus_name TEXT
    engine_name TEXT
    cer_mean    REAL
    wer_mean    REAL
    doc_count   INTEGER
    metadata    TEXT              — JSON

Usage
-----
>>> from picarones.core.history import BenchmarkHistory
>>> history = BenchmarkHistory("~/.picarones/history.db")
>>> history.record(benchmark_result)
>>> df = history.query(engine="tesseract", corpus="chroniques")
>>> regression = history.detect_regression(engine="tesseract", threshold=0.02)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class HistoryEntry:
    """Un enregistrement dans l'historique des benchmarks."""
    run_id: str
    timestamp: str
    corpus_name: str
    engine_name: str
    cer_mean: Optional[float]
    wer_mean: Optional[float]
    doc_count: int
    metadata: dict = field(default_factory=dict)

    @property
    def cer_percent(self) -> Optional[float]:
        return self.cer_mean * 100 if self.cer_mean is not None else None

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "corpus_name": self.corpus_name,
            "engine_name": self.engine_name,
            "cer_mean": self.cer_mean,
            "wer_mean": self.wer_mean,
            "doc_count": self.doc_count,
            "metadata": self.metadata,
        }


@dataclass
class RegressionResult:
    """Résultat d'une détection de régression."""
    engine_name: str
    corpus_name: str
    baseline_run_id: str
    baseline_timestamp: str
    baseline_cer: Optional[float]
    current_run_id: str
    current_timestamp: str
    current_cer: Optional[float]
    delta_cer: Optional[float]
    """Delta CER (current - baseline). Positif = régression."""
    is_regression: bool
    threshold: float

    def as_dict(self) -> dict:
        return {
            "engine_name": self.engine_name,
            "corpus_name": self.corpus_name,
            "baseline_run_id": self.baseline_run_id,
            "baseline_timestamp": self.baseline_timestamp,
            "baseline_cer": self.baseline_cer,
            "current_run_id": self.current_run_id,
            "current_timestamp": self.current_timestamp,
            "current_cer": self.current_cer,
            "delta_cer": self.delta_cer,
            "is_regression": self.is_regression,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# BenchmarkHistory
# ---------------------------------------------------------------------------

class BenchmarkHistory:
    """Gestionnaire de l'historique des benchmarks dans SQLite.

    Parameters
    ----------
    db_path:
        Chemin vers le fichier SQLite. Utiliser ``":memory:"`` pour les tests.

    Examples
    --------
    >>> history = BenchmarkHistory("~/.picarones/history.db")
    >>> history.record(benchmark)
    >>> entries = history.query(engine="tesseract")
    >>> for e in entries:
    ...     print(e.timestamp, f"CER={e.cer_percent:.2f}%")
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS runs (
        run_id      TEXT PRIMARY KEY,
        timestamp   TEXT NOT NULL,
        corpus_name TEXT NOT NULL,
        engine_name TEXT NOT NULL,
        cer_mean    REAL,
        wer_mean    REAL,
        doc_count   INTEGER,
        metadata    TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_engine ON runs (engine_name);
    CREATE INDEX IF NOT EXISTS idx_corpus ON runs (corpus_name);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON runs (timestamp);
    """

    def __init__(self, db_path: str = "~/.picarones/history.db") -> None:
        if db_path != ":memory:":
            path = Path(db_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(path)
        else:
            self.db_path = ":memory:"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(self._CREATE_TABLE)
        conn.commit()

    def close(self) -> None:
        """Ferme la connexion SQLite."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Enregistrement
    # ------------------------------------------------------------------

    def record(
        self,
        benchmark_result: "BenchmarkResult",
        run_id: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> str:
        """Enregistre les résultats d'un benchmark dans l'historique.

        Parameters
        ----------
        benchmark_result:
            Résultats à enregistrer (``BenchmarkResult``).
        run_id:
            Identifiant du run (auto-généré si None).
        extra_metadata:
            Métadonnées supplémentaires à stocker.

        Returns
        -------
        str
            L'identifiant du run enregistré.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        timestamp = datetime.now(timezone.utc).isoformat()
        conn = self._connect()

        for report in benchmark_result.engine_reports:
            ranking = benchmark_result.ranking()
            engine_entry = next(
                (r for r in ranking if r["engine"] == report.engine_name),
                None,
            )
            cer_mean = engine_entry["mean_cer"] if engine_entry else None
            wer_mean = engine_entry["mean_wer"] if engine_entry else None

            meta = {
                "engine_version": report.engine_version,
                "engine_config": report.engine_config,
                "picarones_version": benchmark_result.metadata.get("picarones_version", ""),
                **(extra_metadata or {}),
            }

            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, timestamp, corpus_name, engine_name,
                     cer_mean, wer_mean, doc_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{run_id}_{report.engine_name}",
                    timestamp,
                    benchmark_result.corpus_name,
                    report.engine_name,
                    cer_mean,
                    wer_mean,
                    benchmark_result.document_count,
                    json.dumps(meta, ensure_ascii=False),
                ),
            )

        conn.commit()
        logger.info("Benchmark enregistré dans l'historique : run_id=%s", run_id)
        return run_id

    def record_single(
        self,
        run_id: str,
        corpus_name: str,
        engine_name: str,
        cer_mean: Optional[float],
        wer_mean: Optional[float],
        doc_count: int,
        timestamp: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Enregistre manuellement une entrée dans l'historique.

        Utile pour les tests, les imports de données externes, ou pour
        enregistrer des résultats calculés en dehors de Picarones.

        Returns
        -------
        str
            run_id enregistré.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_id, timestamp, corpus_name, engine_name,
                 cer_mean, wer_mean, doc_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp,
                corpus_name,
                engine_name,
                cer_mean,
                wer_mean,
                doc_count,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        return run_id

    # ------------------------------------------------------------------
    # Requêtes
    # ------------------------------------------------------------------

    def query(
        self,
        engine: Optional[str] = None,
        corpus: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[HistoryEntry]:
        """Retourne l'historique des runs, avec filtres optionnels.

        Parameters
        ----------
        engine:
            Filtre sur le nom du moteur.
        corpus:
            Filtre sur le nom du corpus.
        since:
            Date ISO 8601 minimale (``"2025-01-01"``).
        limit:
            Nombre maximum d'entrées retournées.

        Returns
        -------
        list[HistoryEntry]
            Entrées triées par timestamp croissant.
        """
        clauses: list[str] = []
        params: list = []

        if engine:
            clauses.append("engine_name = ?")
            params.append(engine)
        if corpus:
            clauses.append("corpus_name = ?")
            params.append(corpus)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        conn = self._connect()
        rows = conn.execute(
            f"SELECT * FROM runs {where} ORDER BY timestamp ASC LIMIT ?",
            params,
        ).fetchall()

        return [
            HistoryEntry(
                run_id=row["run_id"],
                timestamp=row["timestamp"],
                corpus_name=row["corpus_name"],
                engine_name=row["engine_name"],
                cer_mean=row["cer_mean"],
                wer_mean=row["wer_mean"],
                doc_count=row["doc_count"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def list_engines(self) -> list[str]:
        """Retourne la liste des moteurs présents dans l'historique."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT engine_name FROM runs ORDER BY engine_name"
        ).fetchall()
        return [row[0] for row in rows]

    def list_corpora(self) -> list[str]:
        """Retourne la liste des corpus présents dans l'historique."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT corpus_name FROM runs ORDER BY corpus_name"
        ).fetchall()
        return [row[0] for row in rows]

    def count(self) -> int:
        """Nombre total d'entrées dans l'historique."""
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    # ------------------------------------------------------------------
    # Courbes d'évolution
    # ------------------------------------------------------------------

    def get_cer_curve(
        self,
        engine: str,
        corpus: Optional[str] = None,
    ) -> list[dict]:
        """Retourne les données pour tracer la courbe d'évolution du CER.

        Parameters
        ----------
        engine:
            Nom du moteur.
        corpus:
            Corpus spécifique (None = tous les corpus pour ce moteur).

        Returns
        -------
        list[dict]
            Chaque dict contient ``{"timestamp": str, "cer": float, "run_id": str}``.
        """
        entries = self.query(engine=engine, corpus=corpus, limit=1000)
        return [
            {
                "timestamp": e.timestamp,
                "cer": e.cer_mean,
                "cer_percent": e.cer_percent,
                "run_id": e.run_id,
                "corpus_name": e.corpus_name,
            }
            for e in entries
            if e.cer_mean is not None
        ]

    # ------------------------------------------------------------------
    # Détection de régression
    # ------------------------------------------------------------------

    def detect_regression(
        self,
        engine: str,
        corpus: Optional[str] = None,
        threshold: float = 0.01,
        baseline_run_id: Optional[str] = None,
    ) -> Optional[RegressionResult]:
        """Détecte une régression du CER entre deux runs.

        Compare le run le plus récent à une baseline (le run précédent ou
        un run spécifique).

        Parameters
        ----------
        engine:
            Nom du moteur à surveiller.
        corpus:
            Corpus spécifique (None = tous).
        threshold:
            Seuil de régression en points absolus de CER (ex : 0.01 = 1%).
            Si delta_cer > threshold → régression détectée.
        baseline_run_id:
            run_id de référence. Si None, utilise l'avant-dernier run.

        Returns
        -------
        RegressionResult | None
            None si moins de 2 runs disponibles.
        """
        entries = self.query(engine=engine, corpus=corpus, limit=1000)
        if len(entries) < 2:
            logger.info("Pas assez de runs pour détecter une régression (moteur=%s)", engine)
            return None

        current = entries[-1]

        if baseline_run_id:
            baseline_list = [e for e in entries[:-1] if e.run_id == baseline_run_id]
            baseline = baseline_list[0] if baseline_list else entries[-2]
        else:
            baseline = entries[-2]

        delta = None
        is_regression = False
        if current.cer_mean is not None and baseline.cer_mean is not None:
            delta = current.cer_mean - baseline.cer_mean
            is_regression = delta > threshold

        return RegressionResult(
            engine_name=engine,
            corpus_name=corpus or "tous",
            baseline_run_id=baseline.run_id,
            baseline_timestamp=baseline.timestamp,
            baseline_cer=baseline.cer_mean,
            current_run_id=current.run_id,
            current_timestamp=current.timestamp,
            current_cer=current.cer_mean,
            delta_cer=delta,
            is_regression=is_regression,
            threshold=threshold,
        )

    def detect_all_regressions(
        self,
        threshold: float = 0.01,
    ) -> list[RegressionResult]:
        """Détecte les régressions pour tous les moteurs et corpus connus.

        Parameters
        ----------
        threshold:
            Seuil de régression.

        Returns
        -------
        list[RegressionResult]
            Uniquement les moteurs où une régression est détectée.
        """
        results: list[RegressionResult] = []
        engines = self.list_engines()
        corpora = self.list_corpora()

        for engine in engines:
            for corpus in corpora:
                result = self.detect_regression(engine, corpus, threshold)
                if result and result.is_regression:
                    results.append(result)

        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, output_path: str) -> Path:
        """Exporte l'historique complet en JSON.

        Parameters
        ----------
        output_path:
            Chemin du fichier JSON de sortie.

        Returns
        -------
        Path
            Chemin vers le fichier créé.
        """
        entries = self.query(limit=100_000)
        path = Path(output_path)
        data = {
            "picarones_history": True,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_runs": len(entries),
            "engines": self.list_engines(),
            "corpora": self.list_corpora(),
            "runs": [e.as_dict() for e in entries],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def __repr__(self) -> str:
        return f"BenchmarkHistory(db='{self.db_path}', runs={self.count()})"


# ---------------------------------------------------------------------------
# Données de démonstration longitudinale
# ---------------------------------------------------------------------------

def generate_demo_history(
    db: BenchmarkHistory,
    n_runs: int = 8,
    seed: int = 42,
) -> None:
    """Insère des données fictives de suivi longitudinal pour la démo.

    Simule l'amélioration progressive d'un modèle tesseract sur 8 runs,
    avec une légère régression au run 5.

    Parameters
    ----------
    db:
        Base d'historique à remplir.
    n_runs:
        Nombre de runs à générer.
    seed:
        Graine aléatoire.
    """
    import random
    rng = random.Random(seed)

    engines = ["tesseract", "pero_ocr", "ancien_moteur"]
    corpus = "Chroniques médiévales"

    # Trajectoires de CER simulées (amélioration progressive + bruit)
    base_cers = {
        "tesseract": 0.15,
        "pero_ocr": 0.09,
        "ancien_moteur": 0.28,
    }
    improvements = {
        "tesseract": -0.008,   # améliore de ~0.8% par run
        "pero_ocr": -0.005,    # améliore de ~0.5% par run
        "ancien_moteur": -0.003,
    }

    from datetime import timedelta
    base_date = datetime(2024, 9, 1, tzinfo=timezone.utc)

    for run_idx in range(n_runs):
        run_date = base_date + timedelta(weeks=run_idx * 2)
        run_id = f"demo_run_{run_idx + 1:02d}"

        for engine in engines:
            cer = base_cers[engine] + improvements[engine] * run_idx
            # Ajouter du bruit + régression au run 5
            noise = rng.gauss(0, 0.005)
            if run_idx == 4 and engine == "tesseract":
                noise += 0.02  # régression simulée
            cer = max(0.01, min(0.5, cer + noise))

            wer = cer * 1.8 + rng.gauss(0, 0.01)
            wer = max(0.01, min(0.9, wer))

            db.record_single(
                run_id=f"{run_id}_{engine}",
                corpus_name=corpus,
                engine_name=engine,
                cer_mean=round(cer, 4),
                wer_mean=round(wer, 4),
                doc_count=12,
                timestamp=run_date.isoformat(),
                metadata={
                    "note": f"Run de démonstration #{run_idx + 1}",
                    "engine_version": f"5.{run_idx}.0" if engine == "tesseract" else "0.7.2",
                },
            )
