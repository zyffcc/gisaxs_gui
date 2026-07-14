"""Multi-algorithm sklearn training and evaluation for Classification."""

from __future__ import annotations

import time
from collections import Counter
from typing import Callable, Optional

import numpy as np

from controllers.classification_models import (
    AlgorithmConfig,
    ClassificationSample,
    ExperimentResult,
    MisclassifiedSample,
    ModelEvaluationResult,
    ProjectionConfig,
    ValidationConfig,
)


ProgressCallback = Optional[Callable[[int, int, str], None]]
CancelCallback = Optional[Callable[[], bool]]


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
]


class ClassificationTrainingService:
    """Build sklearn pipelines and compare selected classifiers fairly."""

    def default_algorithm_configs(self) -> list[AlgorithmConfig]:
        return [
            AlgorithmConfig(
                "logistic_regression",
                "Logistic Regression",
                True,
                {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
                "Linear baseline with calibrated class boundaries.",
                True,
            ),
            AlgorithmConfig(
                "linear_svm",
                "Linear SVM",
                True,
                {"C": 1.0, "class_weight": "balanced", "probability": True},
                "Margin classifier for high-dimensional small datasets.",
                True,
            ),
            AlgorithmConfig(
                "rbf_svm",
                "RBF SVM",
                True,
                {"C": 1.0, "gamma": "scale", "class_weight": "balanced", "probability": True},
                "Nonlinear SVM for curved class boundaries.",
                True,
            ),
            AlgorithmConfig(
                "knn",
                "K-Nearest Neighbors",
                True,
                {"n_neighbors": 5, "weights": "distance", "metric": "minkowski"},
                "Local neighbor voting; useful for compact datasets.",
                True,
            ),
            AlgorithmConfig(
                "random_forest",
                "Random Forest",
                True,
                {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced"},
                "Bagged decision trees; robust and scale-insensitive.",
                False,
            ),
            AlgorithmConfig(
                "extra_trees",
                "Extra Trees",
                False,
                {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced"},
                "Randomized tree ensemble with low variance.",
                False,
            ),
            AlgorithmConfig(
                "gradient_boosting",
                "Gradient Boosting",
                False,
                {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3},
                "Sequential boosted trees for tabular features.",
                False,
            ),
            AlgorithmConfig(
                "adaboost",
                "AdaBoost",
                False,
                {"n_estimators": 120, "learning_rate": 0.5},
                "Boosted weak learners for simple boundaries.",
                False,
            ),
            AlgorithmConfig(
                "lda",
                "LDA",
                True,
                {"solver": "lsqr", "shrinkage": "auto"},
                "Linear discriminant model with shrinkage for small samples.",
                True,
            ),
            AlgorithmConfig(
                "gaussian_nb",
                "Gaussian Naive Bayes",
                False,
                {},
                "Fast probabilistic baseline with Gaussian assumptions.",
                False,
            ),
        ]

    def compare_algorithms(
        self,
        X: np.ndarray,
        y: np.ndarray,
        samples: list[ClassificationSample],
        algorithms: list[AlgorithmConfig],
        validation: ValidationConfig,
        projection: ProjectionConfig,
        ranking_metric: str = "macro_f1",
        progress: ProgressCallback = None,
        is_cancelled: CancelCallback = None,
    ) -> ExperimentResult:
        """Train selected algorithms on shared folds and return comparable metrics."""

        selected = [algorithm for algorithm in algorithms if algorithm.enabled]
        if not selected:
            raise ValueError("Select at least one algorithm.")
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("At least two feature rows are required.")
        labels = sorted(str(label) for label in np.unique(y))
        if len(labels) < 2:
            raise ValueError("At least two classes are required.")

        splits, warnings, effective_validation = self._make_splits(y, validation)
        results: list[ModelEvaluationResult] = []
        for index, algorithm in enumerate(selected, start=1):
            if is_cancelled and is_cancelled():
                warnings.append("Training was cancelled.")
                break
            if progress:
                progress(index - 1, len(selected), algorithm.display_name)
            result = self._evaluate_algorithm(
                algorithm,
                X,
                y,
                samples,
                labels,
                splits,
                projection,
            )
            results.append(result)
            if progress:
                progress(index, len(selected), algorithm.display_name)

        results.sort(
            key=lambda result: (
                result.status == "ok",
                result.score(ranking_metric) if result.status == "ok" else float("-inf"),
            ),
            reverse=True,
        )
        return ExperimentResult(
            results=results,
            ranking_metric=ranking_metric,
            labels=labels,
            sample_ids=[sample.sample_id for sample in samples],
            y_true=np.asarray(y, dtype=object),
            warnings=warnings,
            input_shape=(int(X.shape[0]), int(X.shape[1])),
            projection_config=projection,
            validation_config=effective_validation,
        )

    def build_pipeline(
        self,
        config: AlgorithmConfig,
        projection: ProjectionConfig,
        X_shape: tuple[int, int],
        y_train: Optional[np.ndarray] = None,
    ):
        """Create an sklearn Pipeline for one classifier."""

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        steps = []
        if config.requires_scaling:
            steps.append(("scaler", StandardScaler()))

        projection_step = self._projection_step(projection, X_shape, y_train)
        if projection_step is not None:
            steps.append(projection_step)

        steps.append(("classifier", self._classifier(config, y_train)))
        return Pipeline(steps)

    def _evaluate_algorithm(
        self,
        algorithm: AlgorithmConfig,
        X: np.ndarray,
        y: np.ndarray,
        samples: list[ClassificationSample],
        labels: list[str],
        splits: list[tuple[np.ndarray, np.ndarray]],
        projection: ProjectionConfig,
    ) -> ModelEvaluationResult:
        from sklearn.metrics import classification_report, confusion_matrix

        n_samples = X.shape[0]
        oof_predictions = np.empty(n_samples, dtype=object)
        oof_predictions[:] = None
        probabilities = np.full((n_samples, len(labels)), np.nan, dtype=np.float64)
        decision_scores = np.full((n_samples,), np.nan, dtype=np.float64)
        fold_metrics: list[dict[str, float]] = []
        train_time_total = 0.0
        predict_time_total = 0.0

        try:
            for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
                pipeline = self.build_pipeline(algorithm, projection, X.shape, y[train_idx])
                start = time.perf_counter()
                pipeline.fit(X[train_idx], y[train_idx])
                train_time_total += time.perf_counter() - start

                start = time.perf_counter()
                y_pred = pipeline.predict(X[test_idx])
                predict_time_total += time.perf_counter() - start

                oof_predictions[test_idx] = y_pred
                self._capture_scores(pipeline, X[test_idx], test_idx, labels, probabilities, decision_scores)
                fold_metrics.append(self._metric_dict(y[test_idx], y_pred))
                fold_metrics[-1]["fold"] = float(fold_number)

            valid_mask = np.asarray([pred is not None for pred in oof_predictions], dtype=bool)
            if not np.all(valid_mask):
                missing = int(np.sum(~valid_mask))
                raise ValueError(f"{missing} samples did not receive out-of-fold predictions.")

            metrics_mean = {
                metric: float(np.mean([fold[metric] for fold in fold_metrics]))
                for metric in METRIC_COLUMNS
            }
            metrics_std = {
                metric: float(np.std([fold[metric] for fold in fold_metrics]))
                for metric in METRIC_COLUMNS
            }
            report = classification_report(y, oof_predictions, labels=labels, output_dict=True, zero_division=0)
            cm = confusion_matrix(y, oof_predictions, labels=labels)
            misclassified = self._misclassified(samples, y, oof_predictions, probabilities, decision_scores)

            final_pipeline = self.build_pipeline(algorithm, projection, X.shape, y)
            final_pipeline.fit(X, y)

            return ModelEvaluationResult(
                algorithm_id=algorithm.algorithm_id,
                display_name=algorithm.display_name,
                status="ok",
                metrics_mean=metrics_mean,
                metrics_std=metrics_std,
                fold_metrics=fold_metrics,
                confusion_matrix=cm,
                classification_report=report,
                out_of_fold_predictions=oof_predictions,
                probabilities=probabilities if np.any(np.isfinite(probabilities)) else None,
                decision_scores=decision_scores if np.any(np.isfinite(decision_scores)) else None,
                training_time=train_time_total,
                prediction_time=predict_time_total,
                fitted_pipeline=final_pipeline,
                misclassified_samples=misclassified,
                labels=labels,
            )
        except Exception as exc:
            return ModelEvaluationResult(
                algorithm_id=algorithm.algorithm_id,
                display_name=algorithm.display_name,
                status="failed",
                error_message=str(exc),
                labels=labels,
            )

    def _make_splits(
        self,
        y: np.ndarray,
        validation: ValidationConfig,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str], ValidationConfig]:
        from sklearn.model_selection import (
            RepeatedStratifiedKFold,
            StratifiedKFold,
            StratifiedShuffleSplit,
            train_test_split,
        )

        y = np.asarray(y, dtype=object)
        counts = Counter(y)
        min_count = min(counts.values()) if counts else 0
        warnings: list[str] = []
        effective = ValidationConfig(**validation.__dict__)

        if min_count < 2:
            raise ValueError("Each class needs at least two valid included samples.")

        method = validation.method
        if method == "Stratified train/test split":
            class_count = len(counts)
            test_size = float(validation.test_size)
            min_test_size = class_count / max(1, len(y))
            max_test_size = 1.0 - min_test_size
            if test_size < min_test_size:
                test_size = min_test_size
                warnings.append(
                    f"Test size was increased to {test_size:.2f} so every class can appear in the test split."
                )
            if test_size > max_test_size:
                test_size = max_test_size
                warnings.append(
                    f"Test size was reduced to {test_size:.2f} so every class can remain in the train split."
                )
            if not 0.0 < test_size < 1.0:
                raise ValueError("Not enough samples for a stratified train/test split.")
            effective.test_size = test_size
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=validation.random_state,
            )
            return list(splitter.split(np.zeros(len(y)), y)), warnings, effective

        folds = max(2, int(validation.folds))
        if folds > min_count:
            folds = max(2, min_count)
            effective.folds = folds
            warnings.append(f"Fold count was reduced to {folds} because the smallest class has {min_count} samples.")

        if method == "Repeated stratified K-fold":
            splitter = RepeatedStratifiedKFold(
                n_splits=folds,
                n_repeats=max(1, int(validation.repeats)),
                random_state=validation.random_state,
            )
        else:
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=validation.shuffle,
                random_state=validation.random_state if validation.shuffle else None,
            )
        splits = list(splitter.split(np.zeros(len(y)), y))
        if not splits:
            train_idx, test_idx = train_test_split(
                np.arange(len(y)),
                test_size=validation.test_size,
                random_state=validation.random_state,
                stratify=y,
            )
            splits = [(np.asarray(train_idx), np.asarray(test_idx))]
        return splits, warnings, effective

    def _metric_dict(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    def _projection_step(
        self,
        projection: ProjectionConfig,
        X_shape: tuple[int, int],
        y_train: Optional[np.ndarray],
    ):
        if not projection.enabled or projection.method == "None":
            return None
        n_samples, n_features = X_shape
        if y_train is not None:
            n_samples = len(y_train)
        components = max(1, min(int(projection.n_components), n_samples - 1, n_features))
        if components < 1:
            return None
        if projection.method == "PCA":
            from sklearn.decomposition import PCA

            return ("projection", PCA(n_components=components, random_state=42))
        if projection.method == "UMAP":
            try:
                from umap import UMAP
            except Exception as exc:
                raise RuntimeError("UMAP is not available. Install umap-learn or disable projection.") from exc
            neighbors = max(2, min(int(projection.umap_neighbors), max(2, n_samples - 1)))
            return (
                "projection",
                UMAP(
                    n_components=max(2, components),
                    n_neighbors=neighbors,
                    min_dist=float(projection.umap_min_dist),
                    random_state=42,
                ),
            )
        if projection.method == "t-SNE":
            raise RuntimeError("t-SNE is visualization-only and cannot be used in a prediction pipeline.")
        return None

    def _classifier(self, config: AlgorithmConfig, y_train: Optional[np.ndarray]):
        params = dict(config.parameters)
        random_state = 42

        if config.algorithm_id == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                class_weight=self._none_if_string_none(params.get("class_weight", "balanced")),
                max_iter=int(params.get("max_iter", 1000)),
                random_state=random_state,
            )
        if config.algorithm_id == "linear_svm":
            from sklearn.svm import SVC

            return SVC(
                kernel="linear",
                C=float(params.get("C", 1.0)),
                class_weight=self._none_if_string_none(params.get("class_weight", "balanced")),
                probability=bool(params.get("probability", True)),
                random_state=random_state,
            )
        if config.algorithm_id == "rbf_svm":
            from sklearn.svm import SVC

            return SVC(
                kernel="rbf",
                C=float(params.get("C", 1.0)),
                gamma=params.get("gamma", "scale"),
                class_weight=self._none_if_string_none(params.get("class_weight", "balanced")),
                probability=bool(params.get("probability", True)),
                random_state=random_state,
            )
        if config.algorithm_id == "knn":
            from sklearn.neighbors import KNeighborsClassifier

            configured = int(params.get("n_neighbors", 5))
            n_train = len(y_train) if y_train is not None else configured
            n_neighbors = max(1, min(configured, n_train))
            return KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=params.get("weights", "distance"),
                metric=params.get("metric", "minkowski"),
            )
        if config.algorithm_id == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=self._none_if_string_none(params.get("max_depth", None)),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                class_weight=self._none_if_string_none(params.get("class_weight", "balanced")),
                random_state=random_state,
                n_jobs=-1,
            )
        if config.algorithm_id == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier

            return ExtraTreesClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=self._none_if_string_none(params.get("max_depth", None)),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                class_weight=self._none_if_string_none(params.get("class_weight", "balanced")),
                random_state=random_state,
                n_jobs=-1,
            )
        if config.algorithm_id == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(
                n_estimators=int(params.get("n_estimators", 120)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                max_depth=int(params.get("max_depth", 3)),
                random_state=random_state,
            )
        if config.algorithm_id == "adaboost":
            from sklearn.ensemble import AdaBoostClassifier

            return AdaBoostClassifier(
                n_estimators=int(params.get("n_estimators", 120)),
                learning_rate=float(params.get("learning_rate", 0.5)),
                random_state=random_state,
            )
        if config.algorithm_id == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            solver = params.get("solver", "lsqr")
            shrinkage = params.get("shrinkage", "auto")
            return LinearDiscriminantAnalysis(
                solver=solver,
                shrinkage=self._none_if_string_none(shrinkage),
            )
        if config.algorithm_id == "gaussian_nb":
            from sklearn.naive_bayes import GaussianNB

            return GaussianNB()
        raise ValueError(f"Unknown algorithm: {config.algorithm_id}")

    def _capture_scores(
        self,
        pipeline,
        X_test: np.ndarray,
        test_idx: np.ndarray,
        labels: list[str],
        probabilities: np.ndarray,
        decision_scores: np.ndarray,
    ) -> None:
        classifier = pipeline.named_steps.get("classifier")
        classes = list(getattr(classifier, "classes_", labels))
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X_test)
                for source_col, label in enumerate(classes):
                    if label in labels:
                        target_col = labels.index(label)
                        probabilities[test_idx, target_col] = proba[:, source_col]
                return
            except Exception:
                pass
        if hasattr(pipeline, "decision_function"):
            try:
                scores = pipeline.decision_function(X_test)
                if np.ndim(scores) == 1:
                    decision_scores[test_idx] = np.asarray(scores, dtype=np.float64)
                else:
                    decision_scores[test_idx] = np.max(np.asarray(scores, dtype=np.float64), axis=1)
            except Exception:
                pass

    def _misclassified(
        self,
        samples: list[ClassificationSample],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probabilities: np.ndarray,
        decision_scores: np.ndarray,
    ) -> list[MisclassifiedSample]:
        items: list[MisclassifiedSample] = []
        for index, (truth, pred) in enumerate(zip(y_true, y_pred)):
            if str(truth) == str(pred):
                continue
            confidence = None
            if index < probabilities.shape[0] and np.any(np.isfinite(probabilities[index])):
                confidence = float(np.nanmax(probabilities[index]))
            decision_score = None
            if index < len(decision_scores) and np.isfinite(decision_scores[index]):
                decision_score = float(decision_scores[index])
            sample = samples[index]
            items.append(
                MisclassifiedSample(
                    sample_id=sample.sample_id,
                    file_path=sample.file_path,
                    file_name=sample.file_name,
                    true_label=str(truth),
                    predicted_label=str(pred),
                    confidence=confidence,
                    decision_score=decision_score,
                    data_shape=sample.raw_shape,
                )
            )
        return items

    def _none_if_string_none(self, value):
        if value in ("None", "none", "", None):
            return None
        return value
