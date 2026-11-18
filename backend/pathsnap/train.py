import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from pathsnap.core import DECODER, FEATURES, TripStore


SEED = 42


@dataclass
class Example:
    trip: object
    ids: np.ndarray
    features: np.ndarray
    truth: np.ndarray
    route: np.ndarray

    @property
    def labels(self):
        return self.ids == self.truth[:, None]


def load(store, ids, candidate_count, partition, inventory):
    examples = []
    for trip_id in ids:
        trip = store.trip(trip_id)
        route = store.reference(trip_id, len(trip.arcs))
        ids, features = trip.candidates(candidate_count)
        examples.append(Example(trip, ids, features, trip.align_reference(route), route))
        gaps = np.diff(trip.track[:, 2])
        road = coo_matrix((np.ones(len(trip.arcs)), trip.arcs.T), shape=(len(trip.nodes), len(trip.nodes)))
        inventory.append({
            "partition": partition,
            "trip_id": trip_id,
            "observations": len(trip.track),
            "nodes": len(trip.nodes),
            "arcs": len(trip.arcs),
            "route_arcs": len(route),
            "weakly_connected_components": int(connected_components(road, return_labels=False)),
            "isolated_nodes": int(len(trip.nodes) - len(np.unique(trip.arcs))),
            "duplicate_timestamps": int((gaps == 0).sum()),
            "gaps_over_60_seconds": int((gaps > 60).sum()),
            "median_interval_seconds": float(np.median(gaps)) if len(gaps) else None,
            "gps_bounds": [trip.track[:, :2].min(axis=0).tolist(), trip.track[:, :2].max(axis=0).tolist()],
        })
    return examples


def candidates_dataset(examples):
    x = np.concatenate([e.features.reshape(-1, len(FEATURES)) for e in examples])
    y = np.concatenate([e.labels.ravel() for e in examples])
    groups = np.repeat(np.arange(len(examples)), [e.ids.size for e in examples])
    return x, y, groups


def models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=250, random_state=SEED)),
        "gaussian_naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=2, random_state=SEED),
        "extra_trees": ExtraTreesClassifier(n_estimators=60, max_depth=14, n_jobs=2, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=100, random_state=SEED),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=40, random_state=SEED),
        ),
        "xgboost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            tree_method="hist", device="cpu", n_jobs=2, random_state=SEED,
        ),
    }


def cross_validate(model, x, y, folds):
    scores = []
    for train, validation in folds:
        fitted = clone(model).fit(x[train], y[train])
        scores.append(f1_score(y[validation], fitted.predict(x[validation]), average="macro"))
    return scores


def continuity(trip, arcs):
    if len(arcs) < 2:
        return 1.0
    left, right = arcs[:-1], arcs[1:]
    return float(np.mean((left == right) | (trip.arcs[left, 1] == trip.arcs[right, 0])))


def sequence_overlap(reference, prediction):
    # Multiset arc F1 after collapsing consecutive equal predictions.
    from collections import Counter

    predicted = prediction[np.r_[True, prediction[1:] != prediction[:-1]]]
    overlap = sum((Counter(reference) & Counter(predicted)).values())
    return 2 * overlap / (len(reference) + len(predicted))


def evaluate(model, examples, decoder=None):
    labels, scores, top_one, miss, continuity_scores, overlap = [], [], [], [], [], []
    decoded_accuracy, decoded_continuity, decoded_overlap = [], [], []
    start = perf_counter()
    for e in examples:
        prob = model.predict_proba(e.features.reshape(-1, len(FEATURES)))[:, 1].reshape(e.ids.shape)
        direct = e.ids[np.arange(len(e.ids)), prob.argmax(axis=1)]
        labels.append(e.labels.ravel())
        scores.append(prob.ravel())
        top_one.append(direct == e.truth)
        miss.append(~e.labels.any(axis=1))
        continuity_scores.append(continuity(e.trip, direct))
        overlap.append(sequence_overlap(e.route, direct))
        if decoder is not None:
            decoded = e.trip.decode(e.ids, e.features, prob, decoder)
            decoded_accuracy.append(decoded == e.truth)
            decoded_continuity.append(continuity(e.trip, decoded))
            decoded_overlap.append(sequence_overlap(e.route, decoded))
    elapsed = perf_counter() - start
    y, probabilities = np.concatenate(labels), np.concatenate(scores)
    predicted = probabilities > 0.5
    results = {
        "accuracy": accuracy_score(y, predicted),
        "macro_precision": precision_score(y, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(y, predicted, average="macro", zero_division=0),
        "macro_f1": f1_score(y, predicted, average="macro"),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[False, True]).tolist(),
        "log_loss": log_loss(y, probabilities, labels=[False, True]),
        "top1_accuracy": float(np.mean(np.concatenate(top_one))),
        "candidate_miss_rate": float(np.mean(np.concatenate(miss))),
        "route_continuity": float(np.mean(continuity_scores)),
        "arc_overlap_f1": float(np.mean(overlap)),
        "inference_seconds": elapsed,
    }
    if decoder is not None:
        results.update({
            "decoded_accuracy": float(np.mean(np.concatenate(decoded_accuracy))),
            "decoded_continuity": float(np.mean(decoded_continuity)),
            "decoded_arc_overlap_f1": float(np.mean(decoded_overlap)),
            "per_trip": [
                {
                    "classifier_accuracy": float(direct.mean()),
                    "decoded_accuracy": float(decoded.mean()),
                    "classifier_continuity": classifier_continuity,
                    "decoded_continuity": hmm_continuity,
                }
                for direct, decoded, classifier_continuity, hmm_continuity in zip(
                    top_one, decoded_accuracy, continuity_scores, decoded_continuity
                )
            ],
        })
    return results


def bayesian_search(model, x, y, folds, trials):
    spaces = {
        LogisticRegression: ("logisticregression__C", [0.01, 0.1, 1, 10], "logisticregression__class_weight", [None, "balanced"]),
        GaussianNB: ("var_smoothing", [1e-11, 1e-9, 1e-7, 1e-5], "priors", [None, [0.75, 0.25]]),
        RandomForestClassifier: ("max_depth", [6, 10, 14, 18], "min_samples_leaf", [1, 5, 15]),
        ExtraTreesClassifier: ("max_depth", [6, 10, 14, 18], "min_samples_leaf", [1, 5, 15]),
        HistGradientBoostingClassifier: ("max_leaf_nodes", [7, 15, 31, 63], "learning_rate", [0.035, 0.07, 0.14]),
        MLPClassifier: ("mlpclassifier__hidden_layer_sizes", [(16,), (32, 16), (64, 32)], "mlpclassifier__alpha", [0.0001, 0.001, 0.01]),
        XGBClassifier: ("max_depth", [3, 5, 7, 9], "learning_rate", [0.035, 0.07, 0.14]),
    }
    kind = type(model.steps[-1][1]) if hasattr(model, "steps") else type(model)
    first, values_a, second, values_b = spaces[kind]
    grid = [{first: a, second: b} for a, b in product(values_a, values_b)]
    coords = np.array(list(product(np.linspace(0, 1, len(values_a)), np.linspace(0, 1, len(values_b)))))
    rng = np.random.default_rng(SEED)
    available = list(range(len(grid)))
    selected, trials_log = [], []
    for _ in range(min(trials, len(grid))):
        if len(selected) < 2:
            index = int(rng.choice(available))
        else:
            values = np.array([trial["mean_cv_macro_f1"] for trial in trials_log])
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-5, normalize_y=True)
            gp.fit(coords[selected], values)
            mean, std = gp.predict(coords[available], return_std=True)
            improvement = mean - values.max()
            z = improvement / np.maximum(std, 1e-12)
            index = available[int(np.argmax(improvement * norm.cdf(z) + std * norm.pdf(z)))]
        available.remove(index)
        selected.append(index)
        start = perf_counter()
        scores = cross_validate(clone(model).set_params(**grid[index]), x, y, folds)
        trials_log.append({
            "parameters": grid[index], "fold_macro_f1": scores,
            "mean_cv_macro_f1": float(np.mean(scores)), "seconds": perf_counter() - start,
        })
    best = max(trials_log, key=lambda trial: trial["mean_cv_macro_f1"])
    return clone(model).set_params(**best["parameters"]), best, grid, trials_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Test wiring with a small subset; outputs are not final.")
    args = parser.parse_args()
    if args.trials < 3 and not args.smoke:
        parser.error("Use at least 3 trials to include Bayesian selection.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training, testing = (TripStore(args.data, p) for p in ("train", "student_test"))
    train_ids, test_ids = training.ids(), testing.ids()
    if len(train_ids) != 80 or len(test_ids) != 10 or set(train_ids) & set(test_ids):
        parser.error("Expected 80 training and 10 distinct student-test trips.")
    if args.smoke:
        train_ids, test_ids = train_ids[:4], test_ids[:1]
    inventory = []
    train = load(training, train_ids, 8, "train", inventory)
    test = load(testing, test_ids, 8, "student_test", inventory)
    x, y, groups = candidates_dataset(train)
    folds = list(GroupKFold(n_splits=3).split(x, y, groups))
    records = {}
    with threadpool_limits(limits=2):
        for name, model in models().items():
            print(f"Training {name}", flush=True)
            scores = cross_validate(model, x, y, folds)
            start = perf_counter()
            fitted = clone(model).fit(x, y)
            fit_seconds = perf_counter() - start
            records[name] = {
                "parameters": {
                    key: value for key, value in model.get_params().items()
                    if isinstance(value, (str, bool, int, float, tuple, type(None)))
                    and (not isinstance(value, float) or np.isfinite(value))
                },
                "fold_macro_f1": scores,
                "mean_cv_macro_f1": float(np.mean(scores)),
                "fit_seconds": fit_seconds,
                "training_macro_f1": f1_score(y, fitted.predict(x), average="macro"),
                "student_test": evaluate(fitted, test),
            }
        selected = max(records, key=lambda name: records[name]["mean_cv_macro_f1"])
        print(f"Bayesian search for {selected}", flush=True)
        tuned, best, grid, trials = bayesian_search(models()[selected], x, y, folds, 3 if args.smoke else args.trials)
        start = perf_counter()
        tuned.fit(x, y)
        fit_seconds = perf_counter() - start
        decoder_results = evaluate(tuned, test, DECODER)
        tuned_results = evaluate(tuned, test)
    for trip_id, metrics in zip(test_ids, decoder_results["per_trip"]):
        metrics["trip_id"] = trip_id
    artifact = {
        "schema": 1, "model": tuned, "candidate_count": 8, "features": FEATURES,
        "decoder": DECODER, "sklearn_version": sklearn.__version__, "smoke": args.smoke,
    }
    joblib.dump(artifact, output / "model.joblib", compress=3)
    results = {
        "seed": SEED, "smoke": args.smoke, "train_ids": train_ids, "student_test_ids": test_ids,
        "models": records, "selected_model": selected, "best_trial": best, "search_space": grid,
        "objective": "mean grouped-CV candidate macro F1", "trials": trials,
        "tuned_fit_seconds": fit_seconds, "tuned_student_test": tuned_results,
        "decoder_comparison": decoder_results,
        "decoder_settings": artifact["decoder"],
    }
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"Saved model and experiment results to {output}")


if __name__ == "__main__":
    main()
