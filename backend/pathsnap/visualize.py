import argparse
import json
from pathlib import Path

import folium
import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from pathsnap.core import TripStore, predict, route_geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--trip")
    args = parser.parse_args()
    output = Path(args.artifacts)
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    models = results["models"]
    figures, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, title, values in zip(axes.flat, (
        "Grouped CV Macro F1", "Student-test Top-1 accuracy",
        "Student-test Macro F1", "Student-test log loss",
    ), (
        [m["mean_cv_macro_f1"] for m in models.values()],
        [m["student_test"]["top1_accuracy"] for m in models.values()],
        [m["student_test"]["macro_f1"] for m in models.values()],
        [m["student_test"]["log_loss"] for m in models.values()],
    )):
        ax.barh([n.replace("_", " ") for n in models], values)
        ax.set_title(title)
    figures.savefig(output / "model-comparison.png", dpi=150)
    plt.close(figures)

    store = TripStore(args.data, "student_test")
    trip_id = args.trip or results["student_test_ids"][0]
    trip = store.trip(trip_id)
    reference = store.reference(trip_id, len(trip.arcs))
    artifact = joblib.load(output / "model.joblib")
    _, predicted = predict(trip, artifact)
    road = trip.nodes[trip.arcs]
    gps = trip.track[:, :2]
    lo, hi = gps.min(axis=0), gps.max(axis=0)
    margin = np.maximum((hi - lo) * 0.15, 0.0002)
    nearby = ((road >= lo - margin) & (road <= hi + margin)).all(axis=2).any(axis=1)
    road = road[nearby]
    predicted_segments = np.array([
        segment["coordinates"] for segment in route_geometry(trip, predicted, artifact["decoder"])
    ])
    layers = [
        ("Road network", road, "#9ca3af", 1),
        ("Reference route", trip.nodes[trip.arcs[reference]], "#238641", 4),
        ("Predicted route", predicted_segments, "#d1422d", 2),
    ]
    map_view = folium.Map(location=np.median(gps, axis=0)[::-1].tolist())
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    for name, segments, color, weight in layers:
        group = folium.FeatureGroup(name=name)
        if len(segments):
            folium.PolyLine(segments[:, :, ::-1].tolist(), color=color, weight=weight).add_to(group)
        group.add_to(map_view)
        ax.add_collection(LineCollection(segments, colors=color, linewidths=weight * 0.65, label=name))
    points = folium.FeatureGroup(name="GPS observations")
    for lon, lat in gps:
        folium.CircleMarker((lat, lon), radius=2, color="#2867c5", weight=1).add_to(points)
    points.add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    map_view.fit_bounds([lo[::-1].tolist(), hi[::-1].tolist()])
    map_view.save(output / "student-trip-map.html")
    ax.scatter(*gps.T, s=3, color="#2867c5", label="GPS observations")
    ax.set(xlim=(lo[0]-margin[0], hi[0]+margin[0]), ylim=(lo[1]-margin[1], hi[1]+margin[1]),
           xlabel="Longitude", ylabel="Latitude", title=f"Student-test trip {trip_id}")
    ax.legend()
    fig.savefig(output / "student-trip-map.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
