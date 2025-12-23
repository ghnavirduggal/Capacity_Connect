# Capacity Planning (CAP CONNECT)

A Dash/Flask application for planning, forecasting, and transformation workflows. It combines volume summary (with automatic smoothing/anomaly detection), multi-model forecasting, and transformation adjustments with planning workspaces for business areas.

## Key Features
- **Forecasting Workspace**: Volume summary with auto smoothing/anomaly detection, configurable Phase 2 multi-model forecasts (Prophet, RF, XGBoost, VAR, SARIMAX), accuracy views, normalized ratio charts, and downloads/saves.
- **Transformation Projects**: Apply sequential adjustments (Transform/IA/Marketing) to forecasts, view transposed results, and export or persist final forecasts.
- **Planning Workspace**: Business area planning, plan detail pages, validation layout, and activity logging.
- **Config Management**: Persisted model hyperparameters and general settings with reset-to-defaults.

## Getting Started
1. **Install**: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
2. **Run**: `python main.py` (Dash server on localhost).
3. **Navigate**:
   - `/forecast` to start the forecasting wizard.
   - `/forecast/volume-summary`, `/forecast/forecasting`, `/forecast/transformation-projects`, `/forecast/daily-interval` for step-by-step flows.
   - Planning routes: `/planning`, `/plan/<id>` and related BA roll-up paths.

## Data Inputs
- Volume/History uploads: CSV/XLSX with date + volume (optional category/IQ); auto-aggregated to month-level for large files.
- Smoothing/Forecast: Smoothed series with `ds`/`date`, `Final_Smoothed_Value`, optional `IQ_value` and holidays.
- Transformation: Uses latest forecast file or staged Phase 2 results.

## Outputs & Exports
- Download buttons for smoothed data, seasonality, forecasts, configs, accuracy, and transformation results.
- Files saved under `exports/` by default; respects `latest_forecast_base_dir.txt` when present.

## Notes
- Global loading overlay guards navigation.
- Avoid resetting unrelated changes; the repo may contain user-edited work.
- Network access is restricted in some environments; install dependencies locally when needed.
