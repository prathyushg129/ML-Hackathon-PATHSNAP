from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from pathsnap.core import FEATURES, TRIP_ID, TripStore, predict, route_geometry


logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app):
    partition = os.getenv("PATHSNAP_PARTITION", "evaluator_test")
    if partition not in {"student_test", "evaluator_test"}:
        raise RuntimeError("Set PATHSNAP_PARTITION to student_test or evaluator_test.")
    store = TripStore(os.getenv("PATHSNAP_DATA", "data"), partition)
    ids = store.ids()
    if len(ids) != 10:
        raise RuntimeError(f"Expected 10 trips in {partition}, found {len(ids)}.")
    model_path = Path(os.getenv("PATHSNAP_MODEL", "artifacts/model.joblib"))
    artifact = joblib.load(model_path)
    if artifact["schema"] != 1 or artifact["features"] != FEATURES:
        raise RuntimeError("The model feature schema is incompatible.")
    if artifact["sklearn_version"] != sklearn.__version__:
        raise RuntimeError(f"Install scikit-learn=={artifact['sklearn_version']} to load this model.")
    if artifact["smoke"] and partition != "student_test":
        raise RuntimeError("Run the full Colab training before evaluator deployment.")
    app.state.store = store
    app.state.ids = ids
    app.state.artifact = artifact
    with threadpool_limits(limits=2):
        yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
origins = [item.strip() for item in os.getenv("PATHSNAP_UI_ORIGINS", "").split(",") if item.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"])


@app.get("/health")
def health(request: Request):
    return {"status": "ready", "trip_count": len(request.app.state.ids)}


@app.get("/trips")
def list_trips(request: Request):
    return {"trip_ids": request.app.state.ids}


@app.post("/trips/{trip_id}/predict")
def predict_trip(trip_id: str, request: Request):
    state = request.app.state
    if not TRIP_ID.fullmatch(trip_id) or trip_id not in state.ids:
        raise HTTPException(status_code=404, detail="Unknown trip ID")
    started = perf_counter()
    logger.info("PathSnap inference received for trip %s", trip_id)
    try:
        trip = state.store.trip(trip_id)
        _, decoded = predict(trip, state.artifact)
        geometry = route_geometry(trip, decoded, state.artifact["decoder"])
        # Load reference labels only after inference, and return aggregate scoring only.
        route = state.store.reference(trip_id, len(trip.arcs))
        reference_points = trip.align_reference(route)
    except (OSError, ValueError) as exc:
        logger.error("PathSnap trip %s failed: %s", trip_id, exc)
        raise HTTPException(status_code=500, detail="Trip data or scoring is unavailable") from exc
    accuracy = float(np.mean(decoded == reference_points))
    elapsed = round(perf_counter() - started, 3)
    logger.info("PathSnap inference complete for trip %s in %.3fs", trip_id, elapsed)
    return {
        "trip_id": trip_id,
        "observation_count": len(decoded),
        "observations": trip.track.tolist(),
        "matched_arc_ids": decoded.tolist(),
        "predicted_route": geometry,
        "metrics": {"pointwise_accuracy": accuracy, "processing_seconds": elapsed},
    }
