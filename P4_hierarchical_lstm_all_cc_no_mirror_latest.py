r"""
Motorcycle vibration -> subjective rating prediction using full RPM curves only + hierarchical physical LSTM.

What this script does
---------------------
- Uses the same all-engine-class workbook mappings as the Ridge pipeline.
- Uses the latest domain-expert position sensor mapping:
    Handlebar     -> HG
    Rider step    -> RFP LH, RFP RH
    Rider Seat    -> Rider seat foam
    Petrol tank   -> Thigh LH, Thigh RH
    Pillion Seat  -> Pillion seat frame
    Pillion step  -> PFP LH, PFP RH
    Grab handle   -> Grab
- Mirror is removed completely from the model dataset because its features are redundant.
- Builds one training sample per vehicle-position-RPM-band rating.
- Instead of summary features like mean/max/RMS/P95, it feeds the full RPM-wise
  curve into a hierarchical physical 1D LSTM.
- Uses no auxiliary scalar inputs. The model input is only the full RPM-wise
  curve channels for the relevant position sensors.
- Trains one independent hierarchical LSTM per subjective position.
- Uses leave-one-rated-vehicle-out validation separately per position.
- Reports raw, clipped, and nearest-0.25 rounded prediction errors.
- Saves epoch-wise train/validation loss and MAE so you can diagnose underfitting/overfitting.
- Uses hierarchical sensor-order LSTM branches: each physical sensor-order group is
  processed as an RPM sequence before sensor-level and position-level mixing.

Install requirements
--------------------
pip install pandas numpy scikit-learn openpyxl torch

Example run
-----------
python P4_hierarchical_lstm_all_cc_no_mirror.py ^
  --data-dir "B:\Subjective_Rating\All_CC" ^
  --out-dir "B:\Subjective_Rating\All_CC\model_outputs_hierarchical_lstm" ^
  --epochs 500 ^
  --patience 60

Important notes
---------------
- Do not use random row split for this dataset. This script uses group-aware
  splits by rated vehicle.
- Neural networks may overfit because the number of independent vehicles is
  small. Compare only against the same leave-one-vehicle-out Ridge metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this script. Install it with: pip install torch"
    ) from exc


# =============================================================================
# Configuration
# =============================================================================

HIP_FILES = {
    # 125-160 cc
    "bal_125_v1": "BAL_125 to 150_HIP_DATA_07-05-2026_V1.xlsm",
    "bal_125_v2": "BAL_125 to 150_HIP_DATA_07-05-2026_V2.xlsm",
    "bal_125_v3": "BAL_125 to 150_HIP_DATA_07-05-2026_V3.xlsm",
    "bench_125_v1": "Benchmark_125 to 150_HIP_DATA_07-05-2026_V1.xlsm",
    "bench_125_v2": "Benchmark_125 to 150_HIP_DATA_07-05-2026_V2.xlsm",

    # 200-250 cc
    "bal_200_v1": "BAL_200 to 250_HIP_DATA_07-05-2026_V1.xlsm",
    "bal_200_v2": "BAL_200 to 250_HIP_DATA_07-05-2026_V2.xlsm",
    "bench_200": "Benchmark_200 to 250cc_HIP_DATA_07-05-2026.xlsm",

    # 350+ cc
    "bal_350_v1": "BAL_350+_HIP_DATA_07-05-2026_V1.xlsm",
    "bal_350_v2": "BAL_350+_HIP_DATA_07-05-2026_V2.xlsm",
    "bal_350_v3": "BAL_350+_HIP_DATA_07-05-2026_V3.xlsm",
    "bal_350_v4": "BAL_350+_HIP_DATA_07-05-2026_V4.xlsm",
}

SUBJECTIVE_FILE = "All Subjective Ratings Compiled_15-06-2026.xlsx"

RPM_BANDS = ("<3000", "3000-6000", ">6000")
ORDERS = ("1st", "2nd", "0-400Hz")
DIRECTIONS = ("lat", "long", "vert")
POSITIONS = (
    "Handlebar",
    "Rider step",
    "Rider Seat",
    "Petrol tank",
    "Pillion Seat",
    "Pillion step",
    "Grab handle",
)

# Latest domain-expert mapping.
POSITION_SENSORS = {
    "Handlebar": ("HG",),
    "Rider step": ("RFP LH", "RFP RH"),
    "Rider Seat": ("Rider seat foam",),
    "Petrol tank": ("Thigh LH", "Thigh RH"),
    "Pillion Seat": ("Pillion seat frame",),
    "Pillion step": ("PFP LH", "PFP RH"),
    "Grab handle": ("Grab",),
}

POSITION_ALIASES = {
    "handlebar": "Handlebar",
    "handle bar": "Handlebar",
    "handle bar rating": "Handlebar",
    "rider step": "Rider step",
    "rider foot peg": "Rider step",
    "rider footpeg": "Rider step",
    "rider seat": "Rider Seat",
    "petrol tank": "Petrol tank",
    # Kept only so parse_subjective can explicitly skip Mirror rows.
    "mirror": "Mirror",
    "pillion seat": "Pillion Seat",
    "pillion step": "Pillion step",
    "pillion foot peg": "Pillion step",
    "pillion footpeg": "Pillion step",
    "grab handle": "Grab handle",
}

# Exact engine displacement is not available for every mapped vehicle.
ENGINE_CC_NOMINAL = {
    "125-160cc": 142.5,
    "200-250cc": 225.0,
    "350+cc": 350.0,
}

RATING_STEP = 0.25
RATING_MIN = 1.0
RATING_MAX = 10.0
RAW_LAST_COL = "HS"
MIN_POINTS_PER_CURVE = 3
DEFAULT_N_POINTS = 35


@dataclass(frozen=True)
class VehicleSpec:
    engine_class: str
    subjective_sheet: str
    file_key: str
    sheet: str
    rating_col: str
    name: str

    @property
    def objective_id(self) -> str:
        return f"{self.file_key}::{self.sheet}"

    @property
    def vehicle_id(self) -> str:
        # Same subjective sheet + rating column means same rated vehicle.
        return f"{self.subjective_sheet}::{self.rating_col.upper()}"


VEHICLES: Tuple[VehicleSpec, ...] = (
    # 125-160 cc
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v1", "V1", "E", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v1", "V2", "F", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v1", "V3", "G", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v1", "V4", "H", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v1", "V5", "I", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v1", "V7", "J", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v2", "V1", "K", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v2", "V2", "L", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v2", "V3", "M", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v2", "V4", "N", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v2", "V5", "O", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v2", "V6", "P", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v2", "V7", "AA", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v3", "V2", "Q", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v3", "V3", "R", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bal_125_v3", "V4", "S", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bench_125_v1", "V4", "W", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bench_125_v1", "V5", "V", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bench_125_v1", "V6", "Z", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bench_125_v1", "V7", "Y", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bench_125_v2", "V1", "X", "Hidden"),
    VehicleSpec("125-160cc", "125CC to 160CC", "bench_125_v2", "V2", "U", "Hidden"),

    # 200-250 cc
    VehicleSpec("200-250cc", "200CC to 250CC", "bal_200_v1", "V1", "E", "Hidden"),
    VehicleSpec("200-250cc", "200CC to 250CC", "bal_200_v1", "V2", "O", "Hidden"),
    VehicleSpec("200-250cc", "200CC to 250CC", "bal_200_v1", "V4", "F", "Hidden"),
    VehicleSpec("200-250cc", "200CC to 250CC", "bal_200_v1", "V5", "G", "Hidden"),
    VehicleSpec("200-250cc", "200CC to 250CC", "bal_200_v1", "V6", "I", "Hidden"),
    VehicleSpec("200-250cc", "200CC to 250CC", "bal_200_v2", "V1", "J", "Hidden"),
    VehicleSpec("200-250cc", "200CC to 250CC", "bal_200_v2", "V3", "K", "Hidden"),
    VehicleSpec("200-250cc", "200CC to 250CC", "bal_200_v2", "V4", "P", "Hidden"),
    VehicleSpec("200-250cc", "200CC to 250CC", "bal_200_v2", "V5", "G", "Hidden"),
    VehicleSpec("200-250cc", "200CC to 250CC", "bench_200", "V1", "N", "Hidden"),
    VehicleSpec("200-250cc", "200CC to 250CC", "bench_200", "V2", "L", "Hidden"),

    # 350+ cc
    VehicleSpec("350+cc", "350CC+", "bal_350_v1", "V1", "E", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v1", "V3", "G", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v1", "V4", "F", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v1", "V5", "H", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v1", "V6", "I", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v1", "V7", "J", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v2", "V1", "K", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v2", "V2", "L", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v2", "V3", "M", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v2", "V4", "N", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v2", "V5", "O", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v3", "V1", "P", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v3", "V2", "Q", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v3", "V3", "R", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v3", "V4", "S", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v3", "V5", "T", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v3", "V6", "U", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v3", "V7", "V", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v4", "V1", "W", "Hidden"),
    VehicleSpec("350+cc", "350CC+", "bal_350_v4", "V2", "X", "Hidden"),
)


# =============================================================================
# General helpers
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def excel_col_index(col: str) -> int:
    n = 0
    for ch in col.upper().strip():
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid Excel column: {col!r}")
        n = 26 * n + ord(ch) - ord("A") + 1
    return n - 1


def resolve_file(data_dir: Path, filename: str) -> Path:
    exact = data_dir / filename
    if exact.exists():
        return exact
    p = Path(filename)
    matches = sorted(data_dir.glob(f"{p.stem}(*){p.suffix}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing file: {filename}")


def normalise_words(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def canonical_position(value: object) -> Optional[str]:
    return POSITION_ALIASES.get(normalise_words(value))


def canonical_band(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).replace("\xa0", " ").strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("rpm", "")
    text = text.replace("–", "-").replace("—", "-")
    if text in RPM_BANDS:
        return text
    return None


def parse_rating(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NAN", "NONE", "-"}:
        return np.nan
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([+-])?", text)
    if match is None:
        return np.nan
    rating = float(match.group(1))
    if match.group(2) == "+":
        rating += 0.25
    elif match.group(2) == "-":
        rating -= 0.25
    return rating


def band_mask(rpm: Iterable[float] | pd.Series | np.ndarray, band: str) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(rpm), errors="coerce").to_numpy(float)
    if band == "<3000":
        return numeric < 3000
    if band == "3000-6000":
        return (numeric >= 3000) & (numeric <= 6000)
    if band == ">6000":
        return numeric > 6000
    raise ValueError(f"Unknown RPM band: {band}")


def feature_token(text: str) -> str:
    text = text.lower().replace("0-400hz", "broadband_0_400")
    text = text.replace("1st", "order_1").replace("2nd", "order_2")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def detect_rpm_rows(raw: pd.DataFrame) -> List[int]:
    rpm = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    valid = rpm.between(500, 20000, inclusive="both")
    starts = [i for i in raw.index if i >= 13 and bool(valid.loc[i])]
    if not starts:
        return []
    rows = [starts[0]]
    row_i = starts[0] + 1
    while row_i < len(raw) and bool(valid.iloc[row_i]):
        rows.append(row_i)
        row_i += 1
    return rows


def parse_order(primary: object, fallback: object) -> Optional[str]:
    texts = [str(v).lower() for v in (primary, fallback) if not pd.isna(v)]
    for text in texts:
        if "1st" in text or "1 order" in text:
            return "1st"
        if "2nd" in text or "2 order" in text:
            return "2nd"
        if "0" in text and ("400" in text or "401" in text):
            return "0-400Hz"
    return None


def parse_direction(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).lower()
    for direction in DIRECTIONS:
        if direction in text:
            return direction
    return None


@lru_cache(maxsize=2)
def cached_excel_file(path_string: str) -> pd.ExcelFile:
    return pd.ExcelFile(
        path_string,
        engine="openpyxl",
        engine_kwargs={"read_only": True, "data_only": True},
    )


@lru_cache(maxsize=None)
def workbook_sheet_names(path_string: str) -> Tuple[str, ...]:
    excel = cached_excel_file(path_string)
    return tuple(str(s) for s in excel.sheet_names)



def round_to_rating_step(values: Iterable[float] | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    clipped = np.clip(array, RATING_MIN, RATING_MAX)
    rounded = np.floor(clipped / RATING_STEP + 0.5) * RATING_STEP
    return np.clip(rounded, RATING_MIN, RATING_MAX)


# =============================================================================
# Subjective ratings
# =============================================================================

def parse_subjective(path: Path) -> pd.DataFrame:
    specs_by_sheet: Dict[str, List[VehicleSpec]] = {}
    for spec in VEHICLES:
        specs_by_sheet.setdefault(spec.subjective_sheet, []).append(spec)

    records: List[dict] = []
    for subjective_sheet, specs in specs_by_sheet.items():
        raw = pd.read_excel(path, sheet_name=subjective_sheet, header=None, engine="openpyxl")
        current_position: Optional[str] = None
        for row_i in range(6, len(raw)):
            position = canonical_position(raw.iloc[row_i, 2])
            if position is not None:
                current_position = position
            if current_position is None:
                continue
            if current_position == "Mirror":
                # Mirror ratings are intentionally excluded because mirror features are redundant.
                continue
            band = canonical_band(raw.iloc[row_i, 3])
            if band is None:
                continue
            for spec in specs:
                col_i = excel_col_index(spec.rating_col)
                if col_i >= raw.shape[1]:
                    continue
                rating = parse_rating(raw.iloc[row_i, col_i])
                if np.isnan(rating):
                    continue
                records.append({
                    "vehicle_id": spec.vehicle_id,
                    "objective_id": spec.objective_id,
                    "vehicle_name": spec.name,
                    "engine_class": spec.engine_class,
                    "subjective_sheet": spec.subjective_sheet,
                    "file_key": spec.file_key,
                    "sheet": spec.sheet,
                    "rating_col": spec.rating_col,
                    "position": current_position,
                    "rpm_range": band,
                    "rating": float(rating),
                })

    ratings = pd.DataFrame(records)
    if ratings.empty:
        raise ValueError("No subjective ratings were parsed.")

    duplicate_key = ["objective_id", "position", "rpm_range"]
    if ratings.duplicated(duplicate_key).any():
        bad = ratings[ratings.duplicated(duplicate_key, keep=False)].sort_values(duplicate_key)
        raise ValueError(
            "Duplicate objective-position-band ratings found:\n"
            + bad[duplicate_key + ["rating"]].to_string(index=False)
        )

    return ratings.sort_values(
        ["engine_class", "vehicle_id", "objective_id", "position", "rpm_range"]
    ).reset_index(drop=True)


# =============================================================================
# Objective parsers
# =============================================================================

def parse_hip_sheet(path: Path, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(
        cached_excel_file(str(path)),
        sheet_name=sheet,
        header=None,
        usecols=f"A:{RAW_LAST_COL}",
        nrows=300,
    )
    rpm_rows = detect_rpm_rows(raw)
    columns = ["RPM", "sensor", "order", "direction", "value"]
    if not rpm_rows:
        return pd.DataFrame(columns=columns)

    rpm = pd.to_numeric(raw.iloc[rpm_rows, 1], errors="coerce").to_numpy(float)
    last_col = min(excel_col_index(RAW_LAST_COL), raw.shape[1] - 1)

    current_sensor = ""
    current_order: Optional[str] = None
    pieces: List[pd.DataFrame] = []

    for col_i in range(2, last_col + 1):
        sensor_cell = raw.iloc[10, col_i]
        if not pd.isna(sensor_cell) and str(sensor_cell).strip():
            current_sensor = re.sub(r"\s+", " ", str(sensor_cell).strip())

        order = parse_order(raw.iloc[11, col_i], raw.iloc[8, col_i])
        if order is not None:
            current_order = order
        direction = parse_direction(raw.iloc[3, col_i])

        if not current_sensor or current_order is None or direction is None:
            continue

        values = pd.to_numeric(raw.iloc[rpm_rows, col_i], errors="coerce").to_numpy(float)
        part = pd.DataFrame({
            "RPM": rpm,
            "sensor": current_sensor,
            "order": current_order,
            "direction": direction,
            "value": np.abs(values),
        }).dropna(subset=["RPM", "value"])
        if not part.empty:
            pieces.append(part)

    if not pieces:
        return pd.DataFrame(columns=columns)
    return pd.concat(pieces, ignore_index=True)



# =============================================================================
# Full-curve feature construction
# =============================================================================

def resample_curve(rpm: Iterable[float], values: Iterable[float], n_points: int) -> np.ndarray:
    x = np.asarray(list(rpm) if not isinstance(rpm, np.ndarray) else rpm, dtype=float)
    y = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = np.abs(y[valid])
    if x.size < MIN_POINTS_PER_CURVE:
        return np.full(n_points, np.nan, dtype=np.float32)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if unique_x.size != x.size:
        sums = np.bincount(inverse, weights=y)
        y = sums / counts
        x = unique_x

    if x.size < MIN_POINTS_PER_CURVE or float(x.max() - x.min()) <= 0:
        return np.full(n_points, np.nan, dtype=np.float32)

    target_rpm = np.linspace(float(x.min()), float(x.max()), n_points)
    return np.interp(target_rpm, x, y).astype(np.float32)


def expected_accel_channel_names(sensors: Sequence[str]) -> List[str]:
    names: List[str] = []
    for sensor in sensors:
        for order in ORDERS:
            base = f"{feature_token(sensor)}__{feature_token(order)}"
            for direction in (*DIRECTIONS, "resultant"):
                names.append(f"{base}__{direction}")
    return names



def acceleration_curve_sample(
    hip: pd.DataFrame,
    sensors: Sequence[str],
    band: str,
    n_points: int,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """Build full RPM-curve channels only; no auxiliary scalar features."""
    channel_names = expected_accel_channel_names(sensors)
    curves: List[np.ndarray] = []
    aux: Dict[str, float] = {}

    for sensor in sensors:
        for order in ORDERS:
            base = f"{feature_token(sensor)}__{feature_token(order)}"
            sensor_order = hip[(hip["sensor"] == sensor) & (hip["order"] == order)]

            # Three direct directions.
            for direction in DIRECTIONS:
                group = sensor_order[sensor_order["direction"] == direction]
                group = group[band_mask(group["RPM"], band)] if not group.empty else group
                curve = (
                    resample_curve(group["RPM"], group["value"], n_points)
                    if not group.empty
                    else np.full(n_points, np.nan, dtype=np.float32)
                )
                curves.append(curve)

            # Resultant from lat/long/vert at each RPM.
            if not sensor_order.empty:
                pivot = sensor_order.pivot_table(index="RPM", columns="direction", values="value", aggfunc="mean")
                if all(direction in pivot.columns for direction in DIRECTIONS):
                    complete = pivot.dropna(subset=list(DIRECTIONS))
                    rpm_values = complete.index.to_numpy(float)
                    result = np.sqrt(sum(complete[d].to_numpy(float) ** 2 for d in DIRECTIONS))
                    mask = band_mask(rpm_values, band)
                    curve = resample_curve(rpm_values[mask], result[mask], n_points)
                else:
                    curve = np.full(n_points, np.nan, dtype=np.float32)
            else:
                curve = np.full(n_points, np.nan, dtype=np.float32)
            curves.append(curve)

    # shape: time x channels
    matrix = np.stack(curves, axis=1).astype(np.float32)
    return matrix, channel_names, aux


@dataclass
class CurveDataset:
    curves: np.ndarray
    aux: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    channel_names_by_position: Dict[str, List[str]]
    aux_names_by_position: Dict[str, List[str]]


def build_curve_dataset(
    ratings: pd.DataFrame,
    file_paths: Mapping[str, Path],
    n_points: int,
) -> Tuple[CurveDataset, pd.DataFrame]:
    hip_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    rows: List[dict] = []
    curves: List[np.ndarray] = []
    aux_rows: List[Dict[str, float]] = []
    skipped: List[dict] = []
    channel_names_by_position: Dict[str, List[str]] = {}
    aux_names_by_position: Dict[str, List[str]] = {}

    # parse_subjective already removes Mirror rows, but this guard protects the
    # dataset builder if an external ratings file still contains unsupported positions.
    ordered = ratings[ratings["position"].isin(POSITION_SENSORS)].sort_values(
        ["file_key", "sheet", "position", "rpm_range"]
    )
    unsupported = ratings[~ratings["position"].isin(POSITION_SENSORS)]
    for rating_row in unsupported.itertuples(index=False):
        skipped.append({
            "objective_id": rating_row.objective_id,
            "vehicle_id": rating_row.vehicle_id,
            "vehicle_name": rating_row.vehicle_name,
            "engine_class": rating_row.engine_class,
            "position": rating_row.position,
            "rpm_range": rating_row.rpm_range,
            "reason": "position excluded from model dataset",
        })

    for rating_row in ordered.itertuples(index=False):
        cache_key = (rating_row.file_key, rating_row.sheet)
        try:
            if cache_key not in hip_cache:
                hip_cache[cache_key] = parse_hip_sheet(file_paths[rating_row.file_key], rating_row.sheet)
            hip = hip_cache[cache_key]
            if hip.empty:
                skipped.append({
                    "objective_id": rating_row.objective_id,
                    "vehicle_id": rating_row.vehicle_id,
                    "vehicle_name": rating_row.vehicle_name,
                    "engine_class": rating_row.engine_class,
                    "position": rating_row.position,
                    "rpm_range": rating_row.rpm_range,
                    "reason": "HIP sheet has no detected RPM data",
                })
                continue

            curve, channel_names, aux = acceleration_curve_sample(
                hip,
                POSITION_SENSORS[rating_row.position],
                rating_row.rpm_range,
                n_points,
            )

            if not np.isfinite(curve).any():
                skipped.append({
                    "objective_id": rating_row.objective_id,
                    "vehicle_id": rating_row.vehicle_id,
                    "vehicle_name": rating_row.vehicle_name,
                    "engine_class": rating_row.engine_class,
                    "position": rating_row.position,
                    "rpm_range": rating_row.rpm_range,
                    "reason": "all requested curve channels unavailable; row ignored",
                })
                continue

            position = rating_row.position
            if position not in channel_names_by_position:
                channel_names_by_position[position] = channel_names
                aux_names_by_position[position] = list(aux.keys())
            else:
                if channel_names != channel_names_by_position[position]:
                    raise ValueError(f"Channel-name mismatch for position {position}")
                for key in aux:
                    if key not in aux_names_by_position[position]:
                        aux_names_by_position[position].append(key)

            rows.append({
                "vehicle_id": rating_row.vehicle_id,
                "objective_id": rating_row.objective_id,
                "vehicle_name": rating_row.vehicle_name,
                "engine_class": rating_row.engine_class,
                "subjective_sheet": rating_row.subjective_sheet,
                "file_key": rating_row.file_key,
                "sheet": rating_row.sheet,
                "rating_col": rating_row.rating_col,
                "position": rating_row.position,
                "rpm_range": rating_row.rpm_range,
                "rating": float(rating_row.rating),
                "n_curve_channels": int(curve.shape[1]),
            })
            curves.append(curve)
            aux_rows.append(aux)
        except Exception as exc:
            skipped.append({
                "objective_id": rating_row.objective_id,
                "vehicle_id": rating_row.vehicle_id,
                "vehicle_name": rating_row.vehicle_name,
                "engine_class": rating_row.engine_class,
                "position": rating_row.position,
                "rpm_range": rating_row.rpm_range,
                "reason": f"exception while building curve: {type(exc).__name__}: {exc}",
            })

    metadata = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    if metadata.empty:
        raise ValueError("Curve dataset is empty.")

    curve_object = np.empty(len(curves), dtype=object)
    for i, curve in enumerate(curves):
        curve_object[i] = curve

    aux_object = np.empty(len(aux_rows), dtype=object)
    for i, row in enumerate(aux_rows):
        aux_object[i] = row

    return CurveDataset(
        curves=curve_object,
        aux=aux_object,
        y=metadata["rating"].to_numpy(float),
        metadata=metadata,
        channel_names_by_position=channel_names_by_position,
        aux_names_by_position=aux_names_by_position,
    ), skipped_df


def position_arrays(curve_data: CurveDataset, position: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str], List[str]]:
    mask = (curve_data.metadata["position"] == position).to_numpy(bool)
    meta = curve_data.metadata.loc[mask].reset_index(drop=True)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return np.empty((0, 0, 0)), np.empty((0, 0)), np.empty((0,)), meta, [], []

    channel_names = curve_data.channel_names_by_position[position]
    aux_names = curve_data.aux_names_by_position[position]
    curves = np.stack([curve_data.curves[i] for i in indices], axis=0).astype(np.float32)

    aux_matrix = np.full((len(indices), len(aux_names)), np.nan, dtype=np.float32)
    for row_i, source_i in enumerate(indices):
        aux_dict = curve_data.aux[source_i]
        for col_i, name in enumerate(aux_names):
            aux_matrix[row_i, col_i] = aux_dict.get(name, np.nan)

    y = curve_data.y[indices].astype(np.float32)
    return curves, aux_matrix, y, meta, channel_names, aux_names


# =============================================================================
# Scaling
# =============================================================================

def fit_curve_aux_scaler(X_curve: np.ndarray, X_aux: np.ndarray) -> dict:
    # Channel-wise scaling over samples and time. Shape: N, T, C.
    curve_mean = np.nanmean(X_curve, axis=(0, 1))
    curve_std = np.nanstd(X_curve, axis=(0, 1))
    curve_mean = np.where(np.isfinite(curve_mean), curve_mean, 0.0).astype(np.float32)
    curve_std = np.where(np.isfinite(curve_std) & (curve_std > 1e-12), curve_std, 1.0).astype(np.float32)

    # No auxiliary scalar features are used in this version, but keep an empty
    # aux scaler so the training/prediction pipeline stays generic.
    if X_aux.shape[1] == 0:
        aux_mean = np.empty((0,), dtype=np.float32)
        aux_std = np.empty((0,), dtype=np.float32)
    else:
        aux_mean = np.nanmean(X_aux, axis=0)
        aux_std = np.nanstd(X_aux, axis=0)
        aux_mean = np.where(np.isfinite(aux_mean), aux_mean, 0.0).astype(np.float32)
        aux_std = np.where(np.isfinite(aux_std) & (aux_std > 1e-12), aux_std, 1.0).astype(np.float32)

    return {
        "curve_mean": curve_mean,
        "curve_std": curve_std,
        "aux_mean": aux_mean,
        "aux_std": aux_std,
    }


def transform_curve_aux(X_curve: np.ndarray, X_aux: np.ndarray, scaler: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    curve = (X_curve - scaler["curve_mean"].reshape(1, 1, -1)) / scaler["curve_std"].reshape(1, 1, -1)
    if X_aux.shape[1] == 0:
        aux = np.empty((X_aux.shape[0], 0), dtype=np.float32)
    else:
        aux = (X_aux - scaler["aux_mean"].reshape(1, -1)) / scaler["aux_std"].reshape(1, -1)
        aux = np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # NaNs in curves become zero after scaling, meaning "training mean".
    curve = np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return curve, aux


# =============================================================================
# LSTM model
# =============================================================================

class SensorOrderLSTMBranch(nn.Module):
    """Process one physically coherent sensor-order group over RPM using LSTM.

    Input group examples:
        hg__order_1 -> lat, long, vert, resultant
        thigh_lh__order_2 -> lat, long, vert, resultant

    For each sensor-order group, the LSTM reads the RPM sequence:
        step 1: [lat, long, vert, resultant] at RPM point 1
        step 2: [lat, long, vert, resultant] at RPM point 2
        ...

    This keeps the same physical hierarchy as the LSTM version:
        sensor-order sequence -> sensor representation -> position representation -> rating
    """

    def __init__(
        self,
        n_group_channels: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.25,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.output_features = int(hidden_size) * (2 if bidirectional else 1)

        # Dropout inside nn.LSTM is applied only between stacked LSTM layers.
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(n_group_channels),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bool(bidirectional),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, curve_group: torch.Tensor) -> torch.Tensor:
        # Input shape: N, T, group_channels.
        # LSTM with batch_first=True also expects N, T, features.
        outputs, _ = self.lstm(curve_group)

        # Mean pooling over RPM/time points gives one representation of the full curve.
        # This is usually more stable than taking only the final time step because the
        # whole RPM band matters, not only the last resampled RPM point.
        pooled = outputs.mean(dim=1)
        return self.dropout(pooled)


def parse_channel_hierarchy(channel_names: Sequence[str]) -> Tuple[OrderedDict, OrderedDict]:
    """Create physical hierarchy from channel names.

    Acceleration channel format:
        sensor__order__component
        e.g. thigh_lh__order_1__lat


    Returns:
        order_groups: group_key -> list of channel indices
        sensor_to_order_groups: sensor_key -> list of group_keys
    """
    order_groups: OrderedDict[str, List[int]] = OrderedDict()
    sensor_to_order_groups: OrderedDict[str, List[str]] = OrderedDict()

    for index, name in enumerate(channel_names):
        parts = str(name).split("__")
        if len(parts) >= 3:
            sensor_key = parts[0]
            order_key = parts[1]
        elif len(parts) == 2:
            sensor_key = parts[0]
            order_key = "disp"
        else:
            sensor_key = "unknown_sensor"
            order_key = "unknown_order"

        group_key = f"{sensor_key}__{order_key}"
        if group_key not in order_groups:
            order_groups[group_key] = []
        order_groups[group_key].append(index)

        if sensor_key not in sensor_to_order_groups:
            sensor_to_order_groups[sensor_key] = []
        if group_key not in sensor_to_order_groups[sensor_key]:
            sensor_to_order_groups[sensor_key].append(group_key)

    return order_groups, sensor_to_order_groups


class HierarchicalCurveLSTM(nn.Module):
    """Hierarchical physical LSTM for full RPM curves.

    Hierarchy used:
        sensor-order RPM sequence, e.g. thigh_lh/order_1
        -> LSTM branch representation
        -> sensor representation, e.g. thigh_lh
        -> position representation, e.g. petrol tank
        -> subjective rating

    This prevents the model from directly mixing unrelated channels like
    1st-order channels with 2nd-order channels or LH with RH at the first level.
    """

    def __init__(
        self,
        channel_names: Sequence[str],
        order_features: int = 8,
        sensor_features: int = 16,
        head_features: int = 32,
        dropout: float = 0.25,
        lstm_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.channel_names = list(channel_names)
        self.order_groups, self.sensor_to_order_groups = parse_channel_hierarchy(self.channel_names)
        if not self.order_groups:
            raise ValueError("No channel groups were created for the hierarchical LSTM.")

        branch_output_features = int(order_features) * (2 if bool(bidirectional) else 1)

        self.order_branches = nn.ModuleDict()
        for group_key, indices in self.order_groups.items():
            self.order_branches[group_key] = SensorOrderLSTMBranch(
                n_group_channels=len(indices),
                hidden_size=int(order_features),
                num_layers=int(lstm_layers),
                dropout=float(dropout),
                bidirectional=bool(bidirectional),
            )

        self.sensor_combiners = nn.ModuleDict()
        for sensor_key, group_keys in self.sensor_to_order_groups.items():
            self.sensor_combiners[sensor_key] = nn.Sequential(
                nn.Linear(len(group_keys) * branch_output_features, int(sensor_features)),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        n_sensors = len(self.sensor_to_order_groups)
        self.head = nn.Sequential(
            nn.Linear(n_sensors * int(sensor_features), int(head_features)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(head_features), 1),
        )

    def forward(self, curve: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        # curve shape: N, T, C. We slice C according to physical groups.
        order_outputs: Dict[str, torch.Tensor] = {}
        for group_key, indices in self.order_groups.items():
            group_curve = curve[:, :, indices]
            order_outputs[group_key] = self.order_branches[group_key](group_curve)

        sensor_outputs: List[torch.Tensor] = []
        for sensor_key, group_keys in self.sensor_to_order_groups.items():
            sensor_input = torch.cat([order_outputs[group_key] for group_key in group_keys], dim=1)
            sensor_outputs.append(self.sensor_combiners[sensor_key](sensor_input))

        position_features = torch.cat(sensor_outputs, dim=1)
        return self.head(position_features).squeeze(1)

def split_train_validation_groups(
    train_indices: np.ndarray,
    groups: pd.Series,
    seed: int,
    val_fraction: float = 0.20,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    train_groups = pd.Series(groups.iloc[train_indices].to_numpy()).drop_duplicates().to_numpy()
    if len(train_groups) < 4:
        return train_indices, None

    rng = np.random.default_rng(seed)
    shuffled = np.array(train_groups, copy=True)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_groups = set(shuffled[:n_val])

    mask_val = groups.iloc[train_indices].isin(val_groups).to_numpy(bool)
    inner_train = train_indices[~mask_val]
    inner_val = train_indices[mask_val]
    if len(inner_train) == 0 or len(inner_val) == 0:
        return train_indices, None
    return inner_train, inner_val


def make_loader(
    X_curve: np.ndarray,
    X_aux: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X_curve, dtype=torch.float32),
        torch.tensor(X_aux, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_lstm_model(
    X_curve: np.ndarray,
    X_aux: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    val_indices: Optional[np.ndarray],
    config: Mapping[str, object],
    device: torch.device,
    channel_names: Sequence[str],
) -> Tuple[HierarchicalCurveLSTM, dict, dict]:
    scaler = fit_curve_aux_scaler(X_curve[train_indices], X_aux[train_indices])
    X_train_curve, X_train_aux = transform_curve_aux(X_curve[train_indices], X_aux[train_indices], scaler)
    y_train = y[train_indices].astype(np.float32)

    if val_indices is not None and len(val_indices) > 0:
        X_val_curve, X_val_aux = transform_curve_aux(X_curve[val_indices], X_aux[val_indices], scaler)
        y_val = y[val_indices].astype(np.float32)
        has_real_val = True
    else:
        X_val_curve = X_train_curve
        X_val_aux = X_train_aux
        y_val = y_train
        has_real_val = False

    model = HierarchicalCurveLSTM(
        channel_names=channel_names,
        order_features=int(config["order_features"]),
        sensor_features=int(config["sensor_features"]),
        head_features=int(config["head_features"]),
        dropout=float(config["dropout"]),
        lstm_layers=int(config.get("lstm_layers", 1)),
        bidirectional=bool(config.get("bidirectional_lstm", False)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.SmoothL1Loss(beta=0.5)

    train_loader = make_loader(
        X_train_curve,
        X_train_aux,
        y_train,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    train_curve_tensor = torch.tensor(X_train_curve, dtype=torch.float32, device=device)
    train_aux_tensor = torch.tensor(X_train_aux, dtype=torch.float32, device=device)
    train_y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    val_curve_tensor = torch.tensor(X_val_curve, dtype=torch.float32, device=device)
    val_aux_tensor = torch.tensor(X_val_aux, dtype=torch.float32, device=device)
    val_y_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_state = None
    patience_left = int(config["patience"])
    best_epoch = 0
    epoch_history: List[dict] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_curve, batch_aux, batch_y in train_loader:
            batch_curve = batch_curve.to(device)
            batch_aux = batch_aux.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_curve, batch_aux)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            batch_n = int(batch_y.shape[0])
            train_loss_sum += float(loss.item()) * batch_n
            train_count += batch_n

        train_loss_epoch = train_loss_sum / max(train_count, 1)

        model.eval()
        with torch.no_grad():
            train_pred = model(train_curve_tensor, train_aux_tensor)
            val_pred = model(val_curve_tensor, val_aux_tensor)
            train_eval_loss = float(criterion(train_pred, train_y_tensor).item())
            val_loss = float(criterion(val_pred, val_y_tensor).item())

        train_pred_np = train_pred.detach().cpu().numpy().astype(float)
        val_pred_np = val_pred.detach().cpu().numpy().astype(float)
        train_actual_np = y_train.astype(float)
        val_actual_np = y_val.astype(float)
        train_rounded = round_to_rating_step(train_pred_np)
        val_rounded = round_to_rating_step(val_pred_np)

        epoch_history.append({
            "epoch": int(epoch),
            "train_loss_batch_avg": float(train_loss_epoch),
            "train_loss_eval": float(train_eval_loss),
            "val_loss": float(val_loss),
            "train_MAE_raw": float(mean_absolute_error(train_actual_np, train_pred_np)),
            "val_MAE_raw": float(mean_absolute_error(val_actual_np, val_pred_np)),
            "train_MAE_rounded_0_25": float(mean_absolute_error(train_actual_np, train_rounded)),
            "val_MAE_rounded_0_25": float(mean_absolute_error(val_actual_np, val_rounded)),
            "generalization_gap_loss": float(val_loss - train_eval_loss),
            "generalization_gap_MAE_raw": float(mean_absolute_error(val_actual_np, val_pred_np) - mean_absolute_error(train_actual_np, train_pred_np)),
            "has_real_inner_validation": bool(has_real_val),
        })

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = int(config["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "best_val_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "trained_epochs": int(epoch),
        "n_train_rows": int(len(train_indices)),
        "n_val_rows": int(0 if val_indices is None else len(val_indices)),
        "has_real_inner_validation": bool(has_real_val),
        "epoch_history": epoch_history,
    }
    return model, scaler, info

def predict_lstm(
    model: HierarchicalCurveLSTM,
    scaler: Mapping[str, np.ndarray],
    X_curve: np.ndarray,
    X_aux: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    X_curve_t, X_aux_t = transform_curve_aux(X_curve[indices], X_aux[indices], scaler)
    dummy_y = np.zeros(len(indices), dtype=np.float32)
    loader = make_loader(X_curve_t, X_aux_t, dummy_y, batch_size=batch_size, shuffle=False)
    preds: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_curve, batch_aux, _ in loader:
            batch_curve = batch_curve.to(device)
            batch_aux = batch_aux.to(device)
            pred = model(batch_curve, batch_aux).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds).astype(float) if preds else np.array([], dtype=float)


# =============================================================================
# Inference on a new, unrated HIP workbook using already-trained models
# =============================================================================
#
# Everything above this point is used to TRAIN and validate one model per
# position. The functions below let you SCORE a brand-new vehicle's HIP
# workbook using models that were already trained and saved by
# train_final_models() / main() -- i.e. the contents of
# "hier_lstm_models_by_position.pt". Nothing here retrains anything: each
# position's frozen state_dict, scaler, and channel manifest are reloaded
# exactly as they were saved, so predictions reproduce the trained model
# exactly. This model uses no auxiliary scalar inputs (see the module
# docstring), so unlike the Ridge-fusion version of this pipeline there is
# no engine-cc or RPM-band context to supply here -- only the vibration
# curves matter.


def rebuild_position_model(
    info: Mapping[str, object],
    device: Optional[torch.device] = None,
) -> HierarchicalCurveLSTM:
    """Recreate the exact architecture saved by train_final_models() and load its weights.

    train_final_models() saves, per position: state_dict, channel_names, and
    the training config used for that position (which includes every
    architecture hyperparameter: order_features, sensor_features,
    head_features, dropout, lstm_layers, bidirectional_lstm). Reconstructing
    the module from those fields -- instead of retraining -- is what makes
    scoring a new workbook fast: the LSTM weights are frozen exactly as they
    were at the end of training, with no leakage and no group-aware
    refitting.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = dict(info.get("config", {}) or {})
    model = HierarchicalCurveLSTM(
        channel_names=info["channel_names"],
        order_features=int(config.get("order_features", 8)),
        sensor_features=int(config.get("sensor_features", 16)),
        head_features=int(config.get("head_features", 32)),
        dropout=float(config.get("dropout", 0.25)),
        lstm_layers=int(config.get("lstm_layers", 1)),
        bidirectional=bool(config.get("bidirectional_lstm", False)),
    ).to(device)
    model.load_state_dict(info["state_dict"])
    model.eval()
    return model


def predict_workbook(
    path: Path,
    models: Mapping[str, dict],
    device: Optional[torch.device] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score every V1-V7 sheet of one new, unrated HIP workbook.

    Use this together with a saved ``hier_lstm_models_by_position.pt``
    bundle so a new vehicle can be scored without retraining anything:

        bundle = torch.load("hier_lstm_models_by_position.pt", weights_only=False)
        predictions, skipped = predict_workbook(
            path=Path("NewVehicle_HIP_DATA.xlsm"),
            models=bundle["models"],
        )

    Parameters
    ----------
    path:
        Workbook containing one or more V1-V7 HIP sheets for the new vehicle.
        Uses the same parse_hip_sheet() reader as training, so the sheet must
        follow the same layout (RPM column, sensor/order/direction header
        rows) as every training workbook.
    models:
        The ``"models"`` dictionary saved inside
        ``hier_lstm_models_by_position.pt``, i.e. exactly
        ``torch.load(pt_path, weights_only=False)["models"]``. Keyed by
        position name, each value holds that position's frozen state_dict,
        scaler, channel_names, and training config.
    device:
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    (predictions, skipped):
        ``predictions`` has one row per sheet/position/RPM-band that could be
        scored, with raw/clipped/nearest-0.25-rounded ratings. ``skipped``
        logs every sheet/position/band that could not be scored and why
        (missing HIP data, missing sensor for that position, etc.), the same
        way build_curve_dataset() logs skipped training rows.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sheets = [
        sheet
        for sheet in workbook_sheet_names(str(path))
        if re.fullmatch(r"V[1-7]", sheet.strip())
    ]

    # Rebuild every position's frozen model once up front; it does not depend
    # on which sheet/band is being scored.
    rebuilt_models: Dict[str, HierarchicalCurveLSTM] = {}
    for position, info in models.items():
        if position not in POSITION_SENSORS:
            continue
        rebuilt_models[position] = rebuild_position_model(info, device)

    hip_cache: Dict[str, pd.DataFrame] = {}
    predictions: List[dict] = []
    skipped: List[dict] = []

    for sheet in sheets:
        if sheet not in hip_cache:
            hip_cache[sheet] = parse_hip_sheet(path, sheet)
        hip = hip_cache[sheet]

        if hip.empty:
            for position in rebuilt_models:
                for band in RPM_BANDS:
                    skipped.append({
                        "workbook": path.name,
                        "sheet": sheet,
                        "position": position,
                        "rpm_range": band,
                        "reason": "HIP sheet has no detected RPM data",
                    })
            continue

        for position, model in rebuilt_models.items():
            info = models[position]
            sensors = POSITION_SENSORS[position]
            channel_names = list(info["channel_names"])
            n_points = int(dict(info.get("config", {}) or {}).get("n_points", DEFAULT_N_POINTS))

            for band in RPM_BANDS:
                curve, curve_channel_names, _aux = acceleration_curve_sample(
                    hip, sensors, band, n_points
                )

                if curve_channel_names != channel_names:
                    raise ValueError(
                        f"Channel mismatch for {position!r}: this workbook produced "
                        f"{curve_channel_names!r} but the trained model expects "
                        f"{channel_names!r}. Check the sensor mapping/version used "
                        "to train this model bundle."
                    )

                if not np.isfinite(curve).any():
                    skipped.append({
                        "workbook": path.name,
                        "sheet": sheet,
                        "position": position,
                        "rpm_range": band,
                        "reason": "all requested curve channels unavailable; row ignored",
                    })
                    continue

                # No auxiliary scalar inputs in this model: aux_batch always
                # has zero columns, and transform_curve_aux() / the saved
                # scaler already know how to handle that.
                curve_batch = curve[np.newaxis, :, :].astype(np.float32)
                aux_batch = np.empty((1, 0), dtype=np.float32)
                curve_scaled, aux_scaled = transform_curve_aux(
                    curve_batch, aux_batch, info["scaler"]
                )

                curve_tensor = torch.tensor(curve_scaled, dtype=torch.float32, device=device)
                aux_tensor = torch.tensor(aux_scaled, dtype=torch.float32, device=device)

                with torch.no_grad():
                    raw_prediction = float(model(curve_tensor, aux_tensor).cpu().item())

                clipped_prediction = float(np.clip(raw_prediction, RATING_MIN, RATING_MAX))
                rounded_prediction = float(round_to_rating_step(raw_prediction).item())

                predictions.append({
                    "model": "hierarchical_physical_lstm_curve_only",
                    "workbook": path.name,
                    "sheet": sheet,
                    "position": position,
                    "rpm_range": band,
                    "predicted_rating_raw": raw_prediction,
                    "predicted_rating_clipped": clipped_prediction,
                    "predicted_rating_rounded_0_25": rounded_prediction,
                })

    return pd.DataFrame(predictions), pd.DataFrame(skipped)


def load_model_bundle(pt_path: Path) -> dict:
    """Load a hier_lstm_models_by_position.pt bundle saved by this script's main().

    Thin, documented wrapper around torch.load so callers do not need to
    remember the required ``weights_only=False`` flag: the bundle stores
    plain Python dicts/dataclasses and metadata alongside tensors, not only
    tensors, so PyTorch's newer weights-only-by-default loading must be
    disabled. Only load bundles you trained yourself or otherwise trust.
    """
    return torch.load(str(pt_path), map_location="cpu", weights_only=False)


# =============================================================================
# Validation and metrics
# =============================================================================

def regression_metrics(actual: Iterable[float], predicted_raw: Iterable[float]) -> Dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    raw = np.asarray(predicted_raw, dtype=float)
    clipped = np.clip(raw, RATING_MIN, RATING_MAX)
    rounded = round_to_rating_step(raw)
    out = {
        "MAE_raw": float(mean_absolute_error(actual_array, raw)),
        "MAE_clipped": float(mean_absolute_error(actual_array, clipped)),
        "MAE_rounded_0_25": float(mean_absolute_error(actual_array, rounded)),
        "RMSE_raw": float(math.sqrt(mean_squared_error(actual_array, raw))),
        "RMSE_clipped": float(math.sqrt(mean_squared_error(actual_array, clipped))),
        "RMSE_rounded_0_25": float(math.sqrt(mean_squared_error(actual_array, rounded))),
        "exact_match_rate_0_25": float(np.mean(np.isclose(actual_array, rounded))),
        "within_0_25_rate": float(np.mean(np.abs(actual_array - rounded) <= 0.25)),
        "within_0_50_rate": float(np.mean(np.abs(actual_array - rounded) <= 0.50)),
    }
    if np.unique(actual_array).size > 1:
        out.update({
            "R2_raw": float(r2_score(actual_array, raw)),
            "R2_clipped": float(r2_score(actual_array, clipped)),
            "R2_rounded_0_25": float(r2_score(actual_array, rounded)),
        })
    else:
        out.update({"R2_raw": np.nan, "R2_clipped": np.nan, "R2_rounded_0_25": np.nan})
    return out


def lovo_by_position(
    curve_data: CurveDataset,
    config: Mapping[str, object],
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: List[dict] = []
    metric_rows: List[dict] = []
    epoch_history_rows: List[dict] = []

    for position in POSITIONS:
        position_config = dict(config)
        X_curve, X_aux, y, meta, channel_names, aux_names = position_arrays(curve_data, position)
        if len(meta) == 0 or meta["vehicle_id"].nunique() < 3:
            continue

        print(
            f"\nPosition: {position} | rows={len(meta)} | vehicles={meta['vehicle_id'].nunique()} "
            f"| channels={X_curve.shape[2]}",
            flush=True,
        )
        groups = meta["vehicle_id"]
        logo = LeaveOneGroupOut()
        position_predictions: List[dict] = []
        fold_best_epochs: List[int] = []
        fold_trained_epochs: List[int] = []

        fold_no = 0
        for train_index, test_index in logo.split(X_curve, y, groups):
            fold_no += 1
            held_out_vehicle_id = ",".join(sorted(meta.iloc[test_index]["vehicle_id"].unique()))
            inner_train, inner_val = split_train_validation_groups(
                np.asarray(train_index), groups, seed=int(position_config["seed"]) + fold_no
            )
            set_seed(int(position_config["seed"]) + fold_no)
            model, scaler, train_info = train_lstm_model(
                X_curve,
                X_aux,
                y,
                inner_train,
                inner_val,
                position_config,
                device,
                channel_names,
            )
            fold_best_epochs.append(int(train_info["best_epoch"]))
            fold_trained_epochs.append(int(train_info["trained_epochs"]))

            for hist in train_info.get("epoch_history", []):
                epoch_history_rows.append({
                    "scope": "lovo_inner_training",
                    "position": position,
                    "fold_no": int(fold_no),
                    "held_out_vehicle_id": held_out_vehicle_id,
                    "n_outer_train_rows": int(len(train_index)),
                    "n_outer_test_rows": int(len(test_index)),
                    "n_inner_train_rows": int(len(inner_train)),
                    "n_inner_val_rows": int(0 if inner_val is None else len(inner_val)),
                    **hist,
                })

            preds = predict_lstm(
                model,
                scaler,
                X_curve,
                X_aux,
                np.asarray(test_index),
                device,
                batch_size=int(position_config["batch_size"]),
            )

            for local_i, source_index in enumerate(test_index):
                source = meta.iloc[source_index]
                raw_prediction = float(preds[local_i])
                clipped_prediction = float(np.clip(raw_prediction, RATING_MIN, RATING_MAX))
                rounded_prediction = float(round_to_rating_step(raw_prediction).item())
                item = {
                    "model": "hierarchical_physical_lstm_curve_only",
                    "position": position,
                    "held_out_vehicle_id": source.vehicle_id,
                    "objective_id": source.objective_id,
                    "vehicle_name": source.vehicle_name,
                    "engine_class": source.engine_class,
                    "rpm_range": source.rpm_range,
                    "actual": float(source.rating),
                    "predicted_raw": raw_prediction,
                    "predicted_clipped": clipped_prediction,
                    "predicted_rounded_0_25": rounded_prediction,
                    "absolute_error_raw": float(abs(source.rating - raw_prediction)),
                    "absolute_error_rounded_0_25": float(abs(source.rating - rounded_prediction)),
                    "n_curve_channels": int(X_curve.shape[2]),
                    "n_aux_features": int(X_aux.shape[1]),
                    "fold_best_epoch": train_info["best_epoch"],
                    "fold_trained_epochs": train_info["trained_epochs"],
                }
                prediction_rows.append(item)
                position_predictions.append(item)

        if position_predictions:
            position_df = pd.DataFrame(position_predictions)
            metrics = regression_metrics(position_df["actual"], position_df["predicted_raw"])
            metric_row = {
                "model": "hierarchical_physical_lstm_curve_only",
                "position": position,
                "n_rows": len(position_df),
                "n_rated_vehicles": meta["vehicle_id"].nunique(),
                "n_objective_recordings": meta["objective_id"].nunique(),
                "n_curve_channels": int(X_curve.shape[2]),
                "n_aux_features": int(X_aux.shape[1]),
                "median_best_epoch": float(np.median(fold_best_epochs)) if fold_best_epochs else np.nan,
                "p75_best_epoch": float(np.percentile(fold_best_epochs, 75)) if fold_best_epochs else np.nan,
                "median_trained_epochs": float(np.median(fold_trained_epochs)) if fold_trained_epochs else np.nan,
                **metrics,
            }
            metric_rows.append(metric_row)

            print(f"\nCompleted validation for position: {position}", flush=True)
            show_cols = [
                "position", "n_rows", "n_rated_vehicles", "MAE_raw", "MAE_clipped",
                "MAE_rounded_0_25", "RMSE_raw", "RMSE_rounded_0_25",
                "within_0_25_rate", "within_0_50_rate", "R2_raw",
                "median_best_epoch", "p75_best_epoch", "median_trained_epochs",
            ]
            print(pd.DataFrame([metric_row])[show_cols].to_string(index=False), flush=True)
            print(
                "Epoch guide: use the saved epoch-history CSV to plot train_MAE_raw and val_MAE_raw. "
                "A practical starting max epoch for this position is near p75_best_epoch, "
                "while early stopping still protects against overfitting.",
                flush=True,
            )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    epoch_history = pd.DataFrame(epoch_history_rows)

    class_rows: List[dict] = []
    if not predictions.empty:
        for (position, engine_class), group in predictions.groupby(["position", "engine_class"]):
            class_rows.append({
                "model": "hierarchical_physical_lstm_curve_only",
                "position": position,
                "engine_class": engine_class,
                "n_rows": len(group),
                "n_rated_vehicles": group["held_out_vehicle_id"].nunique(),
                **regression_metrics(group["actual"], group["predicted_raw"]),
            })

    return predictions, metrics, pd.DataFrame(class_rows), epoch_history


def train_final_models(
    curve_data: CurveDataset,
    config: Mapping[str, object],
    device: torch.device,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    models: dict = {}
    summary_rows: List[dict] = []
    epoch_history_rows: List[dict] = []

    for position in POSITIONS:
        position_config = dict(config)
        X_curve, X_aux, y, meta, channel_names, aux_names = position_arrays(curve_data, position)
        if len(meta) == 0 or meta["vehicle_id"].nunique() < 3:
            continue

        all_indices = np.arange(len(meta))
        train_idx, val_idx = split_train_validation_groups(
            all_indices,
            meta["vehicle_id"],
            seed=int(position_config["seed"]) + 1000,
        )
        set_seed(int(position_config["seed"]) + 1000)
        model, scaler, train_info = train_lstm_model(
            X_curve,
            X_aux,
            y,
            train_idx,
            val_idx,
            position_config,
            device,
            channel_names,
        )

        for hist in train_info.get("epoch_history", []):
            epoch_history_rows.append({
                "scope": "final_model_training",
                "position": position,
                "n_rows": int(len(meta)),
                "n_train_rows": int(len(train_idx)),
                "n_val_rows": int(0 if val_idx is None else len(val_idx)),
                **hist,
            })

        train_info_for_model = {k: v for k, v in train_info.items() if k != "epoch_history"}
        models[position] = {
            "model_type": "hierarchical_physical_lstm_curve_only",
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "n_channels": int(X_curve.shape[2]),
            "n_aux": int(X_aux.shape[1]),
            "channel_names": channel_names,
            "aux_names": aux_names,
            "order_groups": {k: list(v) for k, v in model.order_groups.items()},
            "sensor_to_order_groups": {k: list(v) for k, v in model.sensor_to_order_groups.items()},
            "scaler": {k: np.asarray(v) for k, v in scaler.items()},
            "n_rows": int(len(meta)),
            "n_rated_vehicles": int(meta["vehicle_id"].nunique()),
            "n_objective_recordings": int(meta["objective_id"].nunique()),
            "train_info": train_info_for_model,
            "config": dict(position_config),
        }
        summary_rows.append({
            "model": "hierarchical_physical_lstm_curve_only",
            "position": position,
            "n_rows": len(meta),
            "n_rated_vehicles": meta["vehicle_id"].nunique(),
            "n_objective_recordings": meta["objective_id"].nunique(),
            "n_curve_channels": int(X_curve.shape[2]),
            "n_aux_features": int(X_aux.shape[1]),
            "best_epoch": train_info["best_epoch"],
            "trained_epochs": train_info["trained_epochs"],
            "best_val_loss": train_info["best_val_loss"],
        })

    return models, pd.DataFrame(summary_rows), pd.DataFrame(epoch_history_rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "model_outputs_hierarchical_lstm_curve_only_no_mirror")
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--order-features", type=int, default=8, help="LSTM hidden size learned per sensor-order group.")
    parser.add_argument("--lstm-layers", type=int, default=1, help="Number of LSTM layers inside each sensor-order branch.")
    parser.add_argument("--bidirectional-lstm", action="store_true", help="Use bidirectional LSTMs in each sensor-order branch.")
    parser.add_argument("--sensor-features", type=int, default=16, help="Features learned per physical sensor.")
    parser.add_argument("--head-features", type=int, default=32, help="Hidden units in the final rating head.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use cuda only if PyTorch with GPU support is installed.",
    )
    args = parser.parse_args()

    if args.n_points < 16:
        parser.error("--n-points should be at least 16.")
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        parser.error("--epochs, --patience, and --batch-size must be positive.")
    if args.order_features <= 0 or args.sensor_features <= 0 or args.head_features <= 0:
        parser.error("--order-features, --sensor-features, and --head-features must be positive.")
    if args.lstm_layers <= 0:
        parser.error("--lstm-layers must be positive.")

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config = {
        "n_points": int(args.n_points),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "order_features": int(args.order_features),
        "sensor_features": int(args.sensor_features),
        "head_features": int(args.head_features),
        "lstm_layers": int(args.lstm_layers),
        "bidirectional_lstm": bool(args.bidirectional_lstm),
        "seed": int(args.seed),
        "branch_architecture": "hierarchical_sensor_order_lstm",
        "mirror_removed": True,
    }

    file_paths = {key: resolve_file(args.data_dir, filename) for key, filename in HIP_FILES.items()}
    subjective_path = resolve_file(args.data_dir, SUBJECTIVE_FILE)

    print("Parsing subjective ratings...")
    ratings = parse_subjective(subjective_path)
    ratings.to_csv(args.out_dir / "parsed_ratings.csv", index=False)
    print(
        f"Ratings after removing Mirror: {len(ratings)} | rated vehicles: {ratings.vehicle_id.nunique()} "
        f"| objective recordings: {ratings.objective_id.nunique()}"
    )

    print("\nBuilding full-curve dataset...")
    curve_data, skipped = build_curve_dataset(ratings, file_paths, n_points=args.n_points)
    curve_data.metadata.to_csv(args.out_dir / "curve_dataset_index.csv", index=False)
    skipped.to_csv(args.out_dir / "skipped_training_rows.csv", index=False)
    print(f"Curve rows: {len(curve_data.metadata)} | skipped rows: {len(skipped)}")
    print(curve_data.metadata.groupby("position").size().to_string())

    channel_manifest = {
        position: {
            "channel_names": curve_data.channel_names_by_position.get(position, []),
            "aux_names": curve_data.aux_names_by_position.get(position, []),
        }
        for position in POSITIONS
    }
    (args.out_dir / "curve_channel_manifest.json").write_text(
        json.dumps(channel_manifest, indent=2),
        encoding="utf-8",
    )

    metrics = None
    if not args.skip_validation:
        print("\nRunning leave-one-rated-vehicle-out hierarchical LSTM validation...")
        predictions, metrics, class_metrics, epoch_history = lovo_by_position(
            curve_data,
            config,
            device,
        )
        predictions.to_csv(args.out_dir / "hier_lstm_lovo_predictions.csv", index=False)
        metrics.to_csv(args.out_dir / "hier_lstm_lovo_metrics_by_position.csv", index=False)
        class_metrics.to_csv(args.out_dir / "hier_lstm_lovo_metrics_by_position_and_engine_class.csv", index=False)
        epoch_history.to_csv(args.out_dir / "hier_lstm_lovo_epoch_history.csv", index=False)

        print("\nHierarchical LSTM leave-one-vehicle-out metrics")
        print(metrics.to_string(index=False))

    print("\nTraining final hierarchical LSTM models on all available rows...")
    final_models, summary, final_epoch_history = train_final_models(curve_data, config, device)
    summary.to_csv(args.out_dir / "hier_lstm_final_model_summary.csv", index=False)
    final_epoch_history.to_csv(args.out_dir / "hier_lstm_final_epoch_history.csv", index=False)
    torch.save(
        {
            "models": final_models,
            "vehicles": [asdict(v) for v in VEHICLES],
            "position_sensors": POSITION_SENSORS,
            "rpm_bands": RPM_BANDS,
            "positions": POSITIONS,
            "config": config,
            "rating_step": RATING_STEP,
        },
        args.out_dir / "hier_lstm_models_by_position.pt",
    )
    print("\nFinal hierarchical LSTM model summary")
    print(summary.to_string(index=False))

    run_config = {
        "model_type": "hierarchical_physical_lstm_curve_only_no_mirror",
        "position_sensors": POSITION_SENSORS,
        "positions": POSITIONS,
        "n_rating_rows": int(len(ratings)),
        "n_curve_rows": int(len(curve_data.metadata)),
        "n_skipped_training_rows": int(len(skipped)),
        "n_rated_vehicles": int(ratings.vehicle_id.nunique()),
        "n_objective_recordings": int(ratings.objective_id.nunique()),
        "skip_validation": bool(args.skip_validation),
        "device": str(device),
        **config,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print("\nSaved outputs in:", args.out_dir)


if __name__ == "__main__":
    main()
