import argparse
import json
from pathlib import Path

import folium
import joblib
import matplotlib.pyplot as plt
import numpy as np
from branca.element import Element
from matplotlib.collections import LineCollection

from pathsnap.core import TripStore, predict, route_geometry


def trip_accuracy_chart(trips, output):
    if "decoded_correct" in trips[0]:
        before = [t["classifier_correct"] / t["observations"] for t in trips]
        after = [t["decoded_correct"] / t["observations"] for t in trips]
    else:
        before = [t["classifier_accuracy"] for t in trips]
        after = [t["decoded_accuracy"] for t in trips]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    rows = np.arange(len(trips))
    ax.barh(rows - 0.18, np.multiply(before, 100), 0.34, label="Classifier only", color="#8999ad")
    ax.barh(rows + 0.18, np.multiply(after, 100), 0.34, label="With HMM/Viterbi", color="#135aa6")
    ax.set(yticks=rows, yticklabels=[t["trip_id"] for t in trips], xlim=(0, 105),
           xlabel="Pointwise accuracy (%)", title="Route decoding results for each student-test trip")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.24)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncols=2, frameon=False)
    fig.savefig(output / "decoder-accuracy-by-trip.png", dpi=170)
    plt.close(fig)


def route_diagnostics(trip, reference, predicted, layers, output, trip_id):
    gps = trip.track[:, :2]
    disagreed = np.flatnonzero(trip.align_reference(reference) != predicted)
    focus = int(disagreed[len(disagreed) // 2]) if len(disagreed) else len(gps) // 2
    detail = gps[max(0, focus - 16):focus + 17]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    for ax, title, points, show_route in zip(
        axes, ("GPS observations", "Matched route", "Detail near a mismatch" if len(disagreed) else "Route detail"),
        (gps, gps, detail), (False, True, True),
    ):
        for index, (name, segments, color, weight) in enumerate(layers):
            if show_route or index == 0:
                ax.add_collection(LineCollection(
                    segments, colors=color, linewidths=weight * 0.65, label=name,
                    linestyles="dashed" if index == 2 else "solid",
                ))
        ax.scatter(*gps.T, s=5, alpha=0.7, color="#2867c5", label="GPS observations")
        lo, hi = points.min(axis=0), points.max(axis=0)
        pad = np.maximum((hi - lo) * 0.12, 0.00015)
        ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]), ylim=(lo[1]-pad[1],hi[1]+pad[1]),
               title=title, xlabel="Longitude", ylabel="Latitude")
        ax.set_aspect(1 / max(np.cos(np.deg2rad(np.mean(points[:, 1]))), 0.1))
        ax.ticklabel_format(useOffset=False)
        ax.grid(alpha=0.14)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    fig.suptitle(f"Map matching on student-test trip {trip_id}")
    fig.savefig(output / "student-trip-route-diagnostics.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--trip")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--leaflet-assets", type=Path, help="Leaflet dist directory for a fully offline HTML map.")
    args = parser.parse_args()
    if args.calibration and not args.model:
        parser.error("--calibration requires the corresponding --model")
    artifacts = Path(args.artifacts)
    output = args.output or artifacts
    output.mkdir(parents=True, exist_ok=True)
    results = json.loads((artifacts / "results.json").read_text(encoding="utf-8"))
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
    accuracy = (
        json.loads(args.calibration.read_text(encoding="utf-8"))["student_test_trips"]
        if args.calibration else results["decoder_comparison"]["per_trip"]
    )
    trip_accuracy_chart(accuracy, output)

    store = TripStore(args.data, "student_test")
    trip_id = args.trip or results["student_test_ids"][0]
    trip = store.trip(trip_id)
    reference = store.reference(trip_id, len(trip.arcs))
    artifact = joblib.load(args.model or artifacts / "model.joblib")
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
    map_view = folium.Map(location=np.median(gps, axis=0)[::-1].tolist(), tiles=None, control_scale=True)
    if args.leaflet_assets:
        map_view.default_js = []
        map_view.default_css = []
        header = map_view.get_root().header
        css = (args.leaflet_assets / "leaflet.css").read_text(encoding="utf-8")
        js = (args.leaflet_assets / "leaflet.js").read_text(encoding="utf-8")
        header.add_child(Element(f"<style>{css}</style>"), name="leaflet_css")
        header.add_child(Element(f"<script>{js}</script>"), name="leaflet_js")
    map_view.get_root().header.add_child(Element("<style>.leaflet-container{background:#f4f7fa}</style>"))
    for name, index in (("roads", 410), ("observations", 420), ("reference", 430), ("prediction", 440)):
        folium.map.CustomPane(name, z_index=index).add_to(map_view)
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    for (name, segments, color, weight), pane in zip(layers, ("roads", "reference", "prediction")):
        group = folium.FeatureGroup(name=name)
        if len(segments):
            folium.PolyLine(
                segments[:, :, ::-1].tolist(), color=color, weight=weight,
                dash_array="5 4" if name == "Predicted route" else None, pane=pane,
            ).add_to(group)
        group.add_to(map_view)
        ax.add_collection(LineCollection(segments, colors=color, linewidths=weight * 0.65, label=name))
    points = folium.FeatureGroup(name="GPS observations")
    for lon, lat in gps:
        folium.CircleMarker(
            (lat, lon), radius=1.5, color="#2867c5", weight=1, opacity=0.55, pane="observations",
        ).add_to(points)
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
    route_diagnostics(trip, reference, predicted, layers, output, trip_id)


if __name__ == "__main__":
    main()
