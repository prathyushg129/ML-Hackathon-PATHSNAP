# PathSnap evaluator dashboard

Requires Node.js 20.19+ or 22.12+ (verified with Node 24). From the project root:

```sh
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 on the UI laptop. Set **Backend URL** to
`http://<backend-laptop-LAN-IP>:8000` and click **Connect / refresh**.
The setting persists in this browser. Alternatively create `frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://192.168.1.20:8000
```

Restart Vite after changing the environment file. A saved browser setting takes
precedence; use the URL field to change it.

## Windows backend laptop

Follow the [backend setup guide](../docs/backend.md) for the working directory, model, data, and partition. In the
same PowerShell window, use these settings before starting Uvicorn:

```powershell
$env:PATHSNAP_UI_ORIGINS = 'http://localhost:5173,http://127.0.0.1:5173,http://192.168.1.30:5173'
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Here 192.168.1.30 is the **UI laptop's** LAN IP and 192.168.1.20 is the
**backend laptop's** LAN IP. CORS origins must match the browser address exactly
(scheme, host, port; no trailing slash). Localhost is correct when the browser
on the UI laptop opens localhost:5173. Both laptops must use the same LAN.
Allow inbound TCP 8000 for Python/Uvicorn on the backend's Windows private network.
Use `ipconfig` on Windows to find its IPv4 address. Do not use 127.0.0.1 as
the backend URL when the backend runs on the other laptop.

The UI makes direct browser requests: GET /health, GET /trips and bodyless
POST /trips/{encoded-string-ID}/predict. It has no hardcoded trip whitelist or
partition selection. The backend controls student/evaluator availability.
GET timeout is 10 seconds; prediction timeout is 5 minutes. After a timeout,
check backend logs before retrying: cancelling a browser request does not stop
server inference.

## Build and checks

```sh
npm run build
npm run preview
npx playwright install chromium
npm test
```

Preview also uses port 5173, so stop the dev server first.
Leaflet 1.9.4 JS/CSS is installed with npm and bundled into the production
build. Once dependencies are installed, the UI and trip geometry work without
internet access. Only OpenStreetMap basemap tiles need internet; a warning
appears if tiles cannot load, while route and GPS layers remain interactive.
The UI uses system fonts and has no external font dependency.

The dark theme supports keyboard focus, reduced motion and mobile layouts.
Editing the backend URL immediately clears old results and disables prediction
until you reconnect. Invalid JSON, malformed predictions and invalid coordinates
are rejected without showing partial metrics or route geometry. Failed
predictions can be retried manually; POST requests are never retried automatically.

No reference geometry, ground-truth labels, dataset, or trained predictions are
bundled in the frontend. Route segments are separate blue polylines and GPS
observations are orange points. Checkbox layers and Fit trip work independently.

See the [verification notes](../docs/frontend-verification.md) for evidence and remaining live checks.
