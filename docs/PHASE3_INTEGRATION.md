# Phase 3 — Frontend / Research Pipeline Integration

## Delivered in this package

- Added a sidebar experiment launcher to `dashboard.py`.
- The launcher executes the existing `hilofed.pipeline` in a separate Python process, using the selected original dataset (METR-LA or PEMS-BAY), rounds, fast-run row budget, seed, and client partition.
- The pipeline writes real result artifacts to the existing `results/` directory. The dashboard reads those same artifacts in Forecast Explorer, Federated Topology, Communication Lab, and Experiments & Reports.
- Captures and displays the run summary and process output. Invalid configuration, missing datasets, runtime errors, and timeouts are surfaced to the user rather than replaced with made-up metrics.
- Added unit tests for integration input validation.

## Run locally (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

In the dashboard sidebar, expand **Run a real experiment**, select settings, and click **Run HiLo-Fed pipeline**. Start with 300–1200 rows and 1–3 rounds. The fast mode is for interactive smoke runs, not final scientific claims.

## Honest limitations

- The experiment runs synchronously in the Streamlit request and may take several minutes. The integration uses a 30-minute process timeout.
- The dashboard refreshes after a successful run. It displays the outputs emitted by the research pipeline; it does not claim a live packet-level network or live physical clients.
- A completed run is not, by itself, comprehensive scientific validation. Phase 5 must rerun and assess test coverage, baselines, ablations, seeds, and both datasets.
- Map tiles require internet access. The original supplied METR-LA and PEMS-BAY datasets remain the source datasets.
