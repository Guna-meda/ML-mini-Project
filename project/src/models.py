"""Reference model runners."""

import numpy as np
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .custom_kernel import category_probabilities, mixed_feature_kernel
from .metrics import MyMetric


def reference_kfold() -> StratifiedKFold:
    return StratifiedKFold(n_splits=4, shuffle=True, random_state=1)


def evaluate_standard_model(model_name: str, estimator, X, y: np.ndarray) -> dict:
    metric = MyMetric()
    for train_ix, test_ix in reference_kfold().split(X, y):
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf = estimator()
        clf.fit(train_X, train_y)
        metric.collect_data(test_y, clf.predict(test_X), train_y, clf.predict(train_X))
    return metric.row(model_name)


def run_reference_models(X, X_ordinal: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = [
        evaluate_standard_model(
            "linear_svm_ovr",
            lambda: OneVsRestClassifier(SVC(kernel="linear")),
            X,
            y,
        ),
        evaluate_standard_model(
            "ovo_linear_svm",
            lambda: SVC(kernel="linear", decision_function_shape="ovo"),
            X,
            y,
        ),
        evaluate_standard_model(
            "ovo_rbf_svm",
            lambda: SVC(kernel="rbf", decision_function_shape="ovo"),
            X,
            y,
        ),
        evaluate_custom_kernel_svm(X_ordinal, y),
        evaluate_standard_model(
            "decision_tree",
            lambda: DecisionTreeClassifier(random_state=1),
            X,
            y,
        ),
    ]
    return rows


def evaluate_custom_kernel_svm(X_ordinal: np.ndarray, y: np.ndarray) -> dict:
    category_prob, material_prob, version_prob = category_probabilities(X_ordinal)
    metric = MyMetric()
    for train_ix, test_ix in reference_kfold().split(X_ordinal, y):
        train_X, test_X = X_ordinal[train_ix], X_ordinal[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        train_kernel = mixed_feature_kernel(
            train_X, train_X, category_prob, material_prob, version_prob
        )
        clf = SVC(kernel="precomputed", decision_function_shape="ovo").fit(train_kernel, train_y)
        pred_y = clf.predict(
            mixed_feature_kernel(test_X, train_X, category_prob, material_prob, version_prob)
        )
        pred_y_train = clf.predict(train_kernel)
        metric.collect_data(test_y, pred_y, train_y, pred_y_train)
    return metric.row("custom_kernel_svm")


def run_bagging_experiment(X, y: np.ndarray) -> list[dict]:
    rows = []
    for num_tree in [2, 5, 10, 20, 50, 100, 200]:
        metric = MyMetric()
        for train_ix, test_ix in reference_kfold().split(X, y):
            train_X, test_X = X[train_ix], X[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            clf = BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=num_tree,
                random_state=1,
            )
            clf.fit(train_X, train_y)
            metric.collect_data(test_y, clf.predict(test_X), train_y, clf.predict(train_X))
        row = metric.row(f"bagged_decision_tree_{num_tree}")
        row["num_trees"] = num_tree
        rows.append(row)
    return rows


def run_gradient_boosting_experiment(X, y: np.ndarray) -> dict:
    """Part 2 extension: Gradient Boosting under the reference CV and metrics."""
    # Dense input: the validated Part 2 result was produced on a dense matrix
    # (sparse input takes a different split-search path in scikit-learn).
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return evaluate_standard_model(
        "gradient_boosting",
        lambda: GradientBoostingClassifier(random_state=1),
        X_dense,
        y,
    )
