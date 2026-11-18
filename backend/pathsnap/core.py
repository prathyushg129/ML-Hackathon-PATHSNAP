from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree


FEATURES = ("distance_m", "heading_cosine", "speed_mps", "arc_length_m", "fraction")
TRIP_ID = re.compile(r"[0-9]{8}")


class TripStore:
    """Read either the supplied ZIP or an extracted partition directory."""

    def __init__(self, path, partition):
        self.path = Path(path)
        self.partition = partition

    def ids(self):
        if self.path.is_file():
            with ZipFile(self.path) as archive:
                return sorted({
                    parts[-2] for name in archive.namelist()
                    if len(parts := name.split("/")) >= 3
                    and parts[-3] == self.partition
                    and TRIP_ID.fullmatch(parts[-2])
                    and parts[-1] == f"{parts[-2]}.track"
                })
        directory = self.path / self.partition
        if not directory.is_dir():
            return []
        return sorted(
            p.name for p in directory.iterdir()
            if p.is_dir() and TRIP_ID.fullmatch(p.name) and (p / f"{p.name}.track").is_file()
        )

    def read(self, trip_id, extension, dtype):
        if not TRIP_ID.fullmatch(trip_id):
            raise FileNotFoundError(trip_id)
        name = f"{self.partition}/{trip_id}/{trip_id}.{extension}"
        if self.path.is_file():
            with ZipFile(self.path) as archive:
                for prefix in ("Student folder/", ""):
                    try:
                        with archive.open(prefix + name) as stream:
                            return np.loadtxt(stream, dtype=dtype, ndmin=2)
                    except KeyError:
                        continue
            raise FileNotFoundError(name)
        return np.loadtxt(self.path / name, dtype=dtype, ndmin=2)

    def trip(self, trip_id):
        return Trip(
            self.read(trip_id, "track", float),
            self.read(trip_id, "nodes", float),
            self.read(trip_id, "arcs", int),
        )

    def reference(self, trip_id, arc_count):
        route = self.read(trip_id, "route", int)
        if route.shape[1] != 1 or not len(route) or np.any((route < 0) | (route >= arc_count)):
            raise ValueError("Invalid reference route")
        return route[:, 0]


def unit_vectors(vectors):
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, 1e-9)


@dataclass
class Trip:
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray

    def __post_init__(self):
        if (
            self.track.shape[1] != 3 or self.nodes.shape[1] != 2
            or self.arcs.shape[1] != 2 or not len(self.track) or not len(self.arcs)
            or not np.isfinite(self.track).all() or not np.isfinite(self.nodes).all()
            or np.any((self.arcs < 0) | (self.arcs >= len(self.nodes)))
        ):
            raise ValueError("Invalid GPS observations or road network")
        if np.any(np.abs(self.track[:, :2]) > [180, 90]) or np.any(np.abs(self.nodes) > [180, 90]):
            raise ValueError("Invalid longitude or latitude")
        self.track = self.track[np.argsort(self.track[:, 2], kind="stable")]
        origin = np.median(self.track[:, :2], axis=0)
        scale = 6371008.8 * np.pi / 180 * np.array([np.cos(np.deg2rad(origin[1])), 1])
        self.xy = (self.track[:, :2] - origin) * scale
        xy_nodes = (self.nodes - origin) * scale
        self.start = xy_nodes[self.arcs[:, 0]]
        self.vector = xy_nodes[self.arcs[:, 1]] - self.start
        self.length = np.linalg.norm(self.vector, axis=1)
        self.direction = unit_vectors(self.vector)

    def project(self, points, ids):
        delta = points - self.start[ids]
        fraction = np.clip(
            (delta * self.vector[ids]).sum(axis=-1) / np.maximum(self.length[ids] ** 2, 1e-9),
            0, 1,
        )
        return np.linalg.norm(delta - fraction[..., None] * self.vector[ids], axis=-1), fraction

    def candidates(self, k=8):
        # Index sampled points along segments so long arcs can be found near their ends.
        subdivisions = np.maximum(1, np.ceil(self.length / 30).astype(int))
        arc_ids = np.repeat(np.arange(len(self.arcs)), subdivisions + 1)
        starts = np.repeat(np.cumsum(subdivisions + 1) - subdivisions - 1, subdivisions + 1)
        fraction = (np.arange(len(arc_ids)) - starts) / subdivisions[arc_ids]
        anchors = self.start[arc_ids] + fraction[:, None] * self.vector[arc_ids]
        tree = cKDTree(anchors)
        k = min(k, len(self.arcs))
        nearest = tree.query(self.xy, k=min(len(anchors), max(2, 8 * k)))[1]
        ids = np.empty((len(self.xy), k), dtype=int)
        for row, neighbors in enumerate(np.reshape(nearest, (len(self.xy), -1))):
            pool = np.unique(arc_ids[neighbors])
            if len(pool) < k:
                pool = np.arange(len(self.arcs))
            distance, _ = self.project(self.xy[row], pool)
            ids[row] = pool[np.argsort(distance, kind="stable")[:k]]
        distances, fractions = self.project(self.xy[:, None, :], ids)
        delta = np.diff(self.xy, axis=0, prepend=self.xy[:1])
        seconds = np.diff(self.track[:, 2], prepend=self.track[0, 2])
        speed = np.linalg.norm(delta, axis=1) / np.maximum(seconds, 1)
        heading = (unit_vectors(delta)[:, None, :] * self.direction[ids]).sum(axis=-1)
        heading[(seconds <= 0) | (seconds > 60)] = 0
        features = np.stack(
            (distances, heading, np.broadcast_to(speed[:, None], ids.shape), self.length[ids], fractions),
            axis=-1,
        )
        return ids, np.nan_to_num(features).astype(np.float32)

    def align_reference(self, route):
        """Derive point labels from the ordered route using monotone alignment."""
        previous = np.zeros(len(route))
        back = np.empty((len(self.xy), len(route)), dtype=np.int32)
        ranks = np.arange(len(route))
        for row, point in enumerate(self.xy):
            distance, _ = self.project(point, route)
            # Prefix maxima restrict predecessors to the same or an earlier route position.
            adjusted = previous - 0.5 * ranks
            maxima = np.maximum.accumulate(adjusted)
            back[row] = np.maximum.accumulate(np.where(adjusted == maxima, ranks, 0))
            previous = maxima + 0.5 * ranks - distance
        indexes = np.empty(len(self.xy), dtype=int)
        indexes[-1] = previous.argmax()
        for row in range(len(indexes) - 1, 0, -1):
            indexes[row - 1] = back[row, indexes[row]]
        return route[indexes]

    def road_graph(self):
        graph = [[] for _ in self.nodes]
        for arc_id, (start, end) in enumerate(self.arcs):
            graph[start].append((end, self.length[arc_id], arc_id))
        return graph

    def decode(self, ids, features, probabilities, settings):
        """Use classifier scores as emissions and directed path lengths as transitions."""
        graph = self.road_graph()

        @lru_cache(maxsize=1024)
        def distances(source):
            queue = [(0.0, source)]
            settled = {}
            while queue:
                length, node = heappop(queue)
                if node in settled:
                    continue
                if length > settings["routing_limit_m"]:
                    break
                settled[node] = length
                for target, weight, _ in graph[node]:
                    if target not in settled:
                        heappush(queue, (length + weight, target))
            return settled

        emissions = np.log(np.clip(probabilities, 1e-9, 1))
        heading_weight = settings.get("smoothed_heading_weight", 0)
        if heading_weight:
            # A short window reduces the influence of a noisy single GPS step.
            rows = np.arange(len(ids))
            left = np.maximum(rows - 3, 0)
            right = np.minimum(rows + 3, len(ids) - 1)
            displacement = self.xy[right] - self.xy[left]
            span = self.track[right, 2] - self.track[left, 2]
            moving = (np.linalg.norm(displacement, axis=1) >= 15) & (span <= 30)
            direction = unit_vectors(displacement)
            agreement = (direction[:, None, :] * self.direction[ids]).sum(axis=-1)
            emissions = emissions + heading_weight * agreement * moving[:, None]
        score = emissions[0]
        back = np.zeros(ids.shape, dtype=int)
        for row in range(1, len(ids)):
            before, after = ids[row - 1], ids[row]
            start = self.arcs[after, 0]
            end = self.arcs[before, 1]
            dt = self.track[row, 2] - self.track[row - 1, 2]
            observed = np.linalg.norm(self.xy[row] - self.xy[row - 1])
            if dt > settings["gap_seconds"]:
                # Treat all candidate pairs equally across a long gap.
                transition = np.zeros((len(before), len(after)))
            else:
                between = np.array([
                    [distances(int(node)).get(int(target), np.inf) for target in start]
                    for node in end
                ])
                left = features[row - 1, :, -1]
                right = features[row, :, -1]
                # Include the partial arcs from and to the projected GPS positions.
                route_length = (
                    between + ((1 - left) * self.length[before])[:, None]
                    + (right * self.length[after])[None, :]
                )
                forward = right[None, :] - left[:, None]
                same = (before[:, None] == after) & (forward >= -0.1)
                route_length = np.where(same, np.maximum(forward, 0) * self.length[before, None], route_length)
                transition = np.where(
                    np.isfinite(route_length),
                    -np.abs(route_length - observed) / settings["transition_scale_m"],
                    -settings["disconnect_penalty"],
                )
            # Rows are predecessor candidates and columns are current candidates.
            joint = score[:, None] + transition
            back[row] = joint.argmax(axis=0)
            score = emissions[row] + joint.max(axis=0)
        picks = np.empty(len(ids), dtype=int)
        picks[-1] = score.argmax()
        for row in range(len(picks) - 1, 0, -1):
            picks[row - 1] = back[row, picks[row]]
        return ids[np.arange(len(ids)), picks]


DECODER = {
    "gap_seconds": 60,
    "routing_limit_m": 1500,
    "transition_scale_m": 55,
    "disconnect_penalty": 12,
    "smoothed_heading_weight": 2,
}


def predict(trip, artifact):
    ids, features = trip.candidates(artifact["candidate_count"])
    probability = artifact["model"].predict_proba(features.reshape(-1, len(FEATURES)))[:, 1]
    probability = probability.reshape(ids.shape)
    direct = ids[np.arange(len(ids)), probability.argmax(axis=1)]
    return direct, trip.decode(ids, features, probability, artifact["decoder"])


def route_geometry(trip, predicted, settings):
    graph = trip.road_graph()

    @lru_cache(maxsize=1024)
    def bridge(source, target):
        queue = [(0.0, int(source), -1, -1)]
        settled = {}
        while queue:
            length, node, previous, arc = heappop(queue)
            if node in settled:
                continue
            if length > settings["routing_limit_m"]:
                break
            settled[node] = (previous, arc)
            if node == target:
                path = []
                while node != source:
                    node, incoming = settled[node]
                    path.append(incoming)
                return path[::-1]
            for neighbor, weight, arc_id in graph[node]:
                if neighbor not in settled:
                    heappush(queue, (length + weight, int(neighbor), node, arc_id))
        return []

    path = [predicted[0]]
    for row, (left, right) in enumerate(zip(predicted[:-1], predicted[1:]), 1):
        if left == right:
            continue
        start, end = trip.arcs[left, 1], trip.arcs[right, 0]
        if start != end and trip.track[row, 2] - trip.track[row - 1, 2] <= settings["gap_seconds"]:
            path.extend(bridge(start, end))
        path.append(right)
    return [
        {"arc_id": int(arc), "coordinates": trip.nodes[trip.arcs[arc]].tolist()}
        for arc in path
    ]
