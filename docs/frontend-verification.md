# Frontend verification — 18 November 2026

## Current update

- Rechecked README and backend/pathsnap/api.py from commit bd65b8a.
- Backend and training files are unchanged.
- Dark responsive dashboard with keyboard focus and reduced-motion support.
- Leaflet 1.9.4 is bundled locally; only basemap tiles need internet.
- Production build passed under Node 24.22.3.
- All five Chromium browser tests passed (Playwright 1.56.1). Desktop and
  390px mobile screenshots were inspected for layout and overflow.
- Browser tests use real Leaflet with controlled API responses. They check
  leading-zero string IDs, bodyless POST, disabled duplicate submission, separate
  route segments, longitude/latitude conversion, fit bounds across trip changes,
  both layer toggles, metric formatting, clearing, failure/retry, changed backend
  addresses, invalid JSON, invalid geometry/metrics, wrong trip responses,
  empty lists and mobile overflow.
- These browser checks are not live inference. No model, reference labels or
  saved predictions were added to the frontend.

## Remaining live two-laptop checks

1. Start FastAPI with the appropriate data/model/partition, binding 0.0.0.0:8000.
2. Set PATHSNAP_UI_ORIGINS to the actual UI browser origin and permit the backend
   port through the Windows private-network firewall.
3. From the UI laptop, verify health, all trip IDs, and a successful prediction
   using the backend LAN address.
4. Compare displayed metrics and geometry with the live response, switch trips,
   and test backend disconnection/reconnection and prediction retry.
5. Repeat against the reserved evaluator partition and full trained model.
6. Capture the live completed results screen and successful API response for the
   technical report after the LAN checks pass.

Screenshots from controlled browser tests are internal layout checks, not
live-results evidence.
