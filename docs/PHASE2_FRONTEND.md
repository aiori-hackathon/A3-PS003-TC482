# Phase 2 Frontend

## Design
Map-first dark geospatial operations UI with dataset selection, map-layer selection, sensor inspector, forecast explorer, federated topology, communication lab, and experiment reports.

## Data integrity
- Uses METR-LA and PEMS-BAY from `data/`.
- Uses supplied location metadata; no invented sensor coordinates.
- Metric views read saved experiment artifacts when present.
- The topology graphic is a logical architecture illustration and is labelled accordingly.
- Basemap tiles require internet access; underlying dataset processing is local.

## Known implementation boundary
The Streamlit map renders sensor points and a selected metric. It does not yet draw validated road polylines because the supplied distance metadata defines graph distances, not road geometry. Sensor click events from the map are not yet wired to the inspector; the inspector uses an explicit sensor selector.
