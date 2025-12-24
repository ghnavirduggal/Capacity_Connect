from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helper: map internal model keys to display names
# ----------------------------
_MODEL_DISPLAY = {
    "prophet": "Prophet",
    "random_forest": "Rf",
    "rf": "Rf",
    "xgboost": "Xgb",
    "xgb": "Xgb",
    "sarimax": "Sarimax",
    "var": "Var",
    # this one is handled separately as a baseline row
    "final_smoothed_values": "Final_smoothed_values",
}


def process_forecast_results(
    forecast_results: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      combined_forecast_df: long format [Model, Month, Month_Year, Forecast]
      wide_forecast_df:     wide table indexed by Model with Month_Year columns + Avg
      pivot_smoothed_df:    Year x Month baseline (NO Avg column)
    """

    combined_forecast_df = pd.DataFrame()

    # 1) Combine forecasts from model outputs
    skip_keys = {"train", "test", "debug", "final_smoothed_values"}
    for model_name, df_fc in (forecast_results or {}).items():
        if model_name in skip_keys:
            continue
        if df_fc is None or getattr(df_fc, "empty", True):
            continue

        df_temp = df_fc.copy()

        # Normalize expected columns
        # common patterns: Prophet gives ds/yhat, others may already have Month/Forecast
        if "ds" in df_temp.columns:
            df_temp = df_temp.rename(columns={"ds": "Month"})
        if "yhat" in df_temp.columns:
            df_temp = df_temp.rename(columns={"yhat": "Forecast"})

        if "Month" not in df_temp.columns or "Forecast" not in df_temp.columns:
            # Skip unknown format safely
            continue

        display = _MODEL_DISPLAY.get(str(model_name).strip().lower(), str(model_name).title())
        df_temp["Model"] = display

        df_temp["Month"] = pd.to_datetime(df_temp["Month"], errors="coerce")
        df_temp["Forecast"] = pd.to_numeric(df_temp["Forecast"], errors="coerce")
        df_temp = df_temp.dropna(subset=["Month", "Forecast"])

        combined_forecast_df = pd.concat(
            [combined_forecast_df, df_temp[["Model", "Month", "Forecast"]]],
            ignore_index=True,
        )

    wide_forecast_df = pd.DataFrame()
    pivot_smoothed_df = pd.DataFrame()

    # 2) Build wide forecast table for display
    if not combined_forecast_df.empty:
        combined_forecast_df = combined_forecast_df.copy()

        # standardize month to period start + create Mon-YY label
        combined_forecast_df["Month"] = (
            pd.to_datetime(combined_forecast_df["Month"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        combined_forecast_df["Month_Year"] = combined_forecast_df["Month"].dt.strftime("%b-%y")

        combined_forecast_df["Model"] = combined_forecast_df["Model"].astype(str).str.strip()
        combined_forecast_df["Month_Year"] = combined_forecast_df["Month_Year"].astype(str).str.strip()

        # average duplicates (if any)
        combined_forecast_df = (
            combined_forecast_df.groupby(["Model", "Month_Year"], as_index=False)["Forecast"].mean()
        )

        wide_forecast_df = (
            combined_forecast_df.pivot(index="Model", columns="Month_Year", values="Forecast")
            .reset_index()
        )

        # Add Avg column (mean across month columns)
        month_cols = [c for c in wide_forecast_df.columns if c not in ("Model", "Avg")]
        if month_cols:
            wide_forecast_df["Avg"] = wide_forecast_df[month_cols].apply(
                lambda r: pd.to_numeric(r, errors="coerce").mean(), axis=1
            )

        # Put Avg right after Model
        cols = list(wide_forecast_df.columns)
        if "Avg" in cols:
            cols = ["Model", "Avg"] + [c for c in cols if c not in ("Model", "Avg")]
            wide_forecast_df = wide_forecast_df[cols]

    # 3) Build pivot of final_smoothed_values baseline (Year x Month)
    if isinstance(forecast_results, dict) and "final_smoothed_values" in forecast_results:
        base = forecast_results["final_smoothed_values"]
        if base is not None and not getattr(base, "empty", True):
            final_smoothed_df = base.copy()

            # Ensure Date exists and is datetime
            if "Date" in final_smoothed_df.columns:
                final_smoothed_df["Date"] = pd.to_datetime(final_smoothed_df["Date"], errors="coerce")
            else:
                # If Date not present, we can't pivot baseline reliably
                final_smoothed_df["Date"] = pd.NaT

            if "Final_Smoothed_Value" in final_smoothed_df.columns:
                final_smoothed_df["Final_Smoothed_Value"] = pd.to_numeric(
                    final_smoothed_df["Final_Smoothed_Value"], errors="coerce"
                )

            final_smoothed_df = final_smoothed_df.dropna(subset=["Date", "Final_Smoothed_Value"])

            final_smoothed_df["Year"] = final_smoothed_df["Date"].dt.year.astype(int)
            final_smoothed_df["Month"] = final_smoothed_df["Date"].dt.strftime("%b")

            pivot_smoothed_df = (
                final_smoothed_df.pivot(
                    index="Year",
                    columns="Month",
                    values="Final_Smoothed_Value",
                )
                .reset_index()
            )

            # Reorder months to calendar order (NO Avg column here)
            months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            keep = ["Year"] + [m for m in months_order if m in pivot_smoothed_df.columns]
            pivot_smoothed_df = pivot_smoothed_df[keep]

    return combined_forecast_df, wide_forecast_df, pivot_smoothed_df


def fill_final_smoothed_row(wide_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills the 'Final_smoothed_values' row in wide_df using baseline_df (Year x Month).
    This function is robust to the presence/absence of 'Avg' and will NEVER require
    baseline_df to have an 'Avg' column (fixes your melt error).
    """

    if wide_df is None or wide_df.empty or "Model" not in wide_df.columns:
        return wide_df

    if baseline_df is None or baseline_df.empty:
        return wide_df

    wide_df = wide_df.copy()
    base = baseline_df.copy()

    # Find the "final smoothed" row robustly (handles truncations like "Final_smoot")
    model_series = wide_df["Model"].astype(str).str.strip().str.lower()
    mask = model_series.str.startswith("final_smoot")  # matches final_smoot / final_smoothed_values
    if not mask.any():
        # also try exact canonical name
        mask = model_series.eq("final_smoothed_values")
        if not mask.any():
            return wide_df

    # Ensure Year column exists in baseline
    if "Year" not in base.columns:
        if base.index.name and str(base.index.name).lower() == "year":
            base = base.reset_index()
        else:
            return wide_df

    # Determine month columns in baseline (explicitly ignore Avg/Model/etc.)
    full_to_abbrev = {
        "january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr",
        "may": "May", "june": "Jun", "july": "Jul", "august": "Aug",
        "september": "Sep", "october": "Oct", "november": "Nov", "december": "Dec",
    }

    def is_month_col(col) -> bool:
        s = str(col).strip()
        low = s.lower()
        if low in {"year", "model", "avg", "average", "total"}:
            return False
        if re.fullmatch(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", low):
            return True
        if low in full_to_abbrev:
            return True
        return False

    month_cols = [c for c in base.columns if is_month_col(c)]
    if not month_cols:
        return wide_df

    # Normalize full month names -> abbreviations
    rename = {}
    for c in month_cols:
        low = str(c).strip().lower()
        if low in full_to_abbrev:
            rename[c] = full_to_abbrev[low]
    if rename:
        base = base.rename(columns=rename)
        month_cols = [rename.get(c, c) for c in month_cols]

    # Melt baseline to lookup Month-Year -> Smoothed
    # NOTE: this is the key fix: we ONLY melt real month columns, never 'Avg'
    baseline_melted = base.melt(
        id_vars="Year",
        value_vars=month_cols,
        var_name="Month",
        value_name="Smoothed",
    )

    baseline_melted["Smoothed"] = pd.to_numeric(baseline_melted["Smoothed"], errors="coerce")
    baseline_melted = baseline_melted.dropna(subset=["Smoothed"])

    baseline_melted["Month_Year"] = baseline_melted.apply(
        lambda r: f"{str(r['Month']).strip().title()[:3]}-{str(int(r['Year']))[-2:]}",
        axis=1,
    )

    baseline_lookup = (
        baseline_melted.groupby("Month_Year")["Smoothed"]
        .mean()
        .to_dict()
    )

    # Fill wide_df columns that match Month_Year keys
    for col in wide_df.columns:
        if col in ("Model",):
            continue
        if col in baseline_lookup:
            # keep your original scaling behavior:
            # baseline is in percent (e.g., 3.10), wide stores fraction (e.g., 0.031)
            wide_df.loc[mask, col] = baseline_lookup[col] / 100.0

    # Recompute Avg for the final row if Avg column exists
    if "Avg" in wide_df.columns:
        month_cols_in_wide = [c for c in wide_df.columns if c not in ("Model", "Avg")]
        if month_cols_in_wide:
            wide_df.loc[mask, "Avg"] = wide_df.loc[mask, month_cols_in_wide].apply(
                lambda r: pd.to_numeric(r, errors="coerce").mean(), axis=1
            )

    return wide_df
