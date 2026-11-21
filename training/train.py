# File: training/train.py
"""Train the lead-quality XGBoost pipeline and register it with MLflow.

Pipeline: ColumnTransformer(OneHotEncoder | StandardScaler) -> XGBClassifier.
The *whole* sklearn Pipeline is logged as a single pyfunc artifact with
``pyfunc_predict_fn="predict_proba"`` so serving/batch consumers receive
calibrated probabilities from ``model.predict(...)`` directly.

Registry flow: register version under ``--registered-model-name`` and
transition it to **Staging**; promotion to Production is a deliberate,
separate act (see README).

Usage:
    python -m training.train --data data/leads.csv \
        --tracking-uri http://localhost:5000 --registered-model-name lead-quality-classifier
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Final

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_SCHEMA_VERSION,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

logger = logging.getLogger("lead_engine.train")

DEFAULT_DATA: Final[Path] = Path("data") / "leads.csv"
DEFAULT_TRACKING_URI: Final[str] = "http://localhost:5000"
DEFAULT_EXPERIMENT: Final[str] = "lead_scoring"
DEFAULT_MODEL_NAME: Final[str] = "lead-quality-classifier"

# Deliberately small grid: 8 combos x 3 folds keeps CI-friendly runtime.
PARAM_GRID: Final[dict[str, list[int | float]]] = {
    "classifier__n_estimators": [300, 600],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.10],
}
CV_SPLITS: Final[int] = 3
RANDOM_STATE: Final[int] = 42


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Assemble preprocessing + XGBoost into one serialisable pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(NUMERIC_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load features/target from CSV with contract validation."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"target column '{TARGET_COLUMN}' missing from {path}")
    missing = [c for c in (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES) if c not in frame.columns]
    if missing:
        raise ValueError(f"feature contract drift: dataset lacks {missing}")

    features = frame.loc[:, [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]].copy()
    target = frame[TARGET_COLUMN].astype(int)
    if target.nunique() != 2:
        raise ValueError(f"target must be binary, found classes={sorted(target.unique())}")
    return features, target


def _plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Render a labelled confusion matrix PNG (matplotlib only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
    ax.set_yticks([0, 1], labels=["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix @ threshold=0.50")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def evaluate(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute holdout metrics including calibration diagnostics."""
    probabilities = np.asarray(pipeline.predict_proba(x_test))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(y_test, predictions)
    metrics: dict[str, Any] = {
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities)),
        "f1_at_0p5": float(f1_score(y_test, predictions)),
        "precision_at_0p5": float(
            cm[1, 1] / max(cm[1, 1] + cm[0, 1], 1)
        ),
        "recall_at_0p5": float(cm[1, 1] / max(cm[1, 1] + cm[1, 0], 1)),
    }
    return {"metrics": metrics, "confusion_matrix": cm, "probabilities": probabilities}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--registered-model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=CV_SPLITS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-register", action="store_true", help="Log run but skip registry transition.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        features, target = load_dataset(args.data)
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=args.test_size, stratify=target, random_state=args.seed
        )
        logger.info("Loaded %d rows (train=%d test=%d, positives=%.3f)",
                    len(features), len(x_train), len(x_test), float(y_train.mean()))

        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)

        cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
        search = GridSearchCV(
            estimator=build_pipeline(args.seed),
            param_grid=PARAM_GRID,
            scoring="average_precision",
            cv=cv,
            refit=True,
            n_jobs=1,  # XGB already parallelises internally
            verbose=1,
        )

        with mlflow.start_run(run_name=f"lead-xgb-{FEATURE_SCHEMA_VERSION}") as run:
            mlflow.log_params(
                {
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "rows_total": len(features),
                    "rows_train": len(x_train),
                    "rows_test": len(x_test),
                    "positive_rate_train": round(float(y_train.mean()), 6),
                    "test_size": args.test_size,
                    "seed": args.seed,
                }
            )
            search.fit(x_train, y_train)
            best_params = {
                k.replace("classifier__", ""): v for k, v in search.best_params_.items()
            }
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("cv_best_pr_auc", float(search.best_score_))

            result = evaluate(search.best_estimator_, x_test, y_test)
            mlflow.log_metrics(result["metrics"])
            logger.info("Holdout metrics: %s", json.dumps(result["metrics"], indent=2))

            # Artifacts: confusion matrix PNG + CV results table.
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                cm_path = tmp_dir / "confusion_matrix.png"
                _plot_confusion_matrix(result["confusion_matrix"], cm_path)
                mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

                cv_table = pd.DataFrame(search.cv_results_)
                cv_csv = tmp_dir / "cv_results.csv"
                cv_table.to_csv(cv_csv, index=False)
                mlflow.log_artifact(str(cv_csv), artifact_path="evaluation")

            # Signature/example are built on RAW feature rows: the pyfunc
            # contract is "raw lead fields in -> probability out".
            input_example = x_test.head(5)
            signature = infer_signature(input_example, result["probabilities"][:5])
            mlflow.sklearn.log_model(
                sk_model=search.best_estimator_,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                pyfunc_predict_fn="predict_proba",
                registered_model_name=None if args.skip_register else args.registered_model_name,
            )

        if args.skip_register:
            logger.info("Run %s complete (registration skipped).", run.info.run_id)
            return 0

        client = MlflowClient(tracking_uri=args.tracking_uri)
        versions = client.search_model_versions(
            f"name='{args.registered_model_name}' and run_id='{run.info.run_id}'"
        )
        if not versions:
            raise RuntimeError(
                f"no registered version found for run {run.info.run_id}"
            )
        version = max(v.version for v in versions)
        client.transition_model_version_stage(
            name=args.registered_model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        logger.info(
            "Registered %s v%s -> Staging (run=%s)",
            args.registered_model_name, version, run.info.run_id,
        )
        return 0

    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except (ValueError, RuntimeError) as exc:
        logger.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
