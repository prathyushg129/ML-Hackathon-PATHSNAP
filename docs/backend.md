# Run FastAPI on Windows

From the repository root, run the following in PowerShell. Replace the dataset path with the location of your ZIP.

```powershell
Set-Location backend
python -m pip install -r requirements.txt
$env:PATHSNAP_PARTITION = 'student_test'
$env:PATHSNAP_DATA = 'C:\path\to\Student folder.zip'
$env:PATHSNAP_MODEL = '../artifacts/model.joblib'
python -m uvicorn pathsnap.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the endpoints. The model is a serialized Python artifact, so load only an artifact created by your team.

| Endpoint | Result |
| --- | --- |
| `GET /health` | Server readiness and trip count |
| `GET /trips` | Available eight-digit trip IDs under `trip_ids` |
| `POST /trips/{trip_id}/predict` | GPS rows, matched arc IDs, route segment coordinates, pointwise accuracy and processing time |

GPS rows have the order `[longitude, latitude, timestamp]`. Route coordinates use `[longitude, latitude]`. The route is a list of segments so the UI can draw separate lines across gaps. Accuracy is a fraction between zero and one. The server logs the trip ID and elapsed time for each successful inference.

## Two-laptop demo

On demo day, set `PATHSNAP_PARTITION=evaluator_test` and point `PATHSNAP_DATA` directly to the supplied `evaluator_test.zip`. The ZIP layout has been verified and the loader discovers all ten reserved trip IDs without extraction. Keep the ZIP on the backend laptop and use the final `artifacts/model.joblib`.

Run Uvicorn with `--host 0.0.0.0` on the demo LAN. Use laptop 1's LAN IP and port `8000` as the UI's API base URL. Set `PATHSNAP_UI_ORIGINS` on laptop 1 to the browser page's full origin, including scheme and port. Allow inbound TCP 8000 on the demo network if Windows Firewall blocks the connection. Verify `GET /health` from laptop 2 and capture the backend logs during the evaluation.
