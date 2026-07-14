import io
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import base64
from html import escape
from uuid import uuid4

try:
    from scipy.stats import pearsonr
    SCIPY_IMPORT_ERROR = None
except Exception as e:
    pearsonr = None
    SCIPY_IMPORT_ERROR = str(e)


# ============================================================
# STREAMLIT APP
# ============================================================
#
# Contents:
#   1) Bundled file helpers
#      Same-folder/GitHub-repo file handling for:
#      - OMNI data file
#      - background image
#      - Algorithm parameters PDF
#
#   2) Data structures
#      Internal row representation and raw OMNI dataframe conversion.
#
#   3) Stage 1: storm detection
#      Detects disturbed episodes and storm minima.
#
#   4) Stage 2: main phase filtering
#      Uses Stage 1 storms to find t_start and filter valid main phases.
#
#   5) Stage 3: Data-complete storms + metrics
#      Checks main phase solar wind data completeness and computes metrics.
#
#   6) Additional outputs, table formatting, sorting, plots, and downloads
#      Builds correlations, Extras outputs, display tables, plots, and CSV/ZIP outputs.
#
#   7) Streamlit rendering functions
#      Defines the display functions for Overview, stage tables, correlations,
#      downloads, and Extras.
#
#   8) Streamlit app layout and display
#      Builds the visible Streamlit page, controls, run button, and selected section.
#
# ============================================================

st.set_page_config(page_title="Magnetic Storm Detection Program", layout="wide")


# ---------------------------
# 1) Bundled file helpers
# ---------------------------

OMNI_DATA_FILENAME = "1964-may 2026.txt"
BACKGROUND_IMAGE_FILENAMES = (
    "background.png",
    "background.jpg",
    "background.jpeg",
    "background.webp",
)
ALGORITHM_PARAMETERS_PDF_FILENAMES = (
    "parameters.pdf",
)


def app_base_dir() -> Path:
    """Return the folder where the app file is running."""
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def bundled_file_candidate_paths(filenames: List[str]) -> List[Path]:
    candidate_paths: List[Path] = []
    for folder in [app_base_dir(), Path.cwd()]:
        for filename in filenames:
            candidate = folder / filename
            if candidate not in candidate_paths:
                candidate_paths.append(candidate)
    return candidate_paths


def find_bundled_file(filenames: List[str]) -> Optional[Path]:
    """Find the first existing bundled file among the allowed filenames."""
    for path in bundled_file_candidate_paths(filenames):
        if path.exists() and path.is_file():
            return path
    return None


def find_background_image() -> Optional[Path]:
    return find_bundled_file(list(BACKGROUND_IMAGE_FILENAMES))


def load_background_image_data_url() -> Optional[str]:
    image_file = find_background_image()
    if image_file is None:
        return None

    suffix = image_file.suffix.lower()
    mime = {
        ".png": "png",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".webp": "webp",
    }.get(suffix, "png")

    encoded = base64.b64encode(image_file.read_bytes()).decode()
    return f"data:image/{mime};base64,{encoded}"


def find_algorithm_parameters_pdf() -> Optional[Path]:
    return find_bundled_file(list(ALGORITHM_PARAMETERS_PDF_FILENAMES))


def load_algorithm_parameters_pdf() -> Tuple[Optional[Path], Optional[bytes]]:
    pdf_path = find_algorithm_parameters_pdf()
    if pdf_path is None:
        return None, None
    return pdf_path, pdf_path.read_bytes()


@st.cache_data(show_spinner=False)
def load_local_omni_text(filename: str = OMNI_DATA_FILENAME) -> str:
    """Load the OMNI data file bundled next to app.py in the GitHub repo."""
    candidate_paths = bundled_file_candidate_paths([filename])
    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore")

    searched = "\n".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f'OMNI data file "{filename}" was not found. Upload it to the same folder as the running Python file.\n\nSearched:\n{searched}'
    )


# ---------------------------
# 2) Data structures
# ---------------------------
# Row is the cleaned internal representation of one hourly OMNI data row.
# Dataframes are created from these rows only after parsing and date filtering.

@dataclass
class Row:
    t: datetime
    imf: float
    bz: float
    vsw: float
    ey: float
    dst: float


def rows_to_df(rows: List[Row]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "t_utc": r.t,
                "IMF": r.imf,
                "Bz": r.bz,
                "Vsw": r.vsw,
                "Ey": r.ey,
                "Dst": r.dst,
            }
            for r in rows
        ]
    )


def dst_int(x: float) -> int:
    return int(round(float(x)))


def fmt_dt(dt) -> str:
    if dt is None or pd.isna(dt):
        return "NA"
    return pd.to_datetime(dt, utc=True).strftime("%Y-%m-%d %H:%M")


def parse_line_new8(line: str) -> Optional[Tuple[int, int, int, float, float, float, float, float]]:
    # Only accept the expected 8-column OMNI format:
    # Year DOY Hour IMF Bz Vsw Ey Dst.
    parts = line.strip().split()
    if not parts:
        return None
    if not parts[0].lstrip("+-").isdigit():
        return None
    if len(parts) != 8:
        return None
    try:
        y = int(parts[0])
        doy = int(parts[1])
        hour = int(parts[2])
        imf = float(parts[3])
        bz = float(parts[4])
        vsw = float(parts[5])
        ey = float(parts[6])
        dst = float(parts[7])
        return y, doy, hour, imf, bz, vsw, ey, dst
    except ValueError:
        return None


def to_time_utc(year: int, doy: int, hour: int, shift_plus_1_hour: bool) -> datetime:
    t = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1, hours=hour)
    if shift_plus_1_hour:
        t += timedelta(hours=1)
    return t


def parse_user_datetime(s: str, is_end: bool) -> datetime:
    s = str(s).strip()
    formats = ["%Y-%m-%d %H:%M", "%Y-%m-%d"]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            if fmt == "%Y-%m-%d":
                if is_end:
                    dt = dt + timedelta(hours=23)
            return dt
        except ValueError:
            pass
    raise ValueError(f"Invalid date format: {s!r}. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'.")


def filter_rows_by_date(rows: List[Row], date_start: str, date_end: str) -> List[Row]:
    t_start = parse_user_datetime(date_start, is_end=False)
    t_end = parse_user_datetime(date_end, is_end=True)
    if t_end < t_start:
        raise ValueError("DATE_END must be >= DATE_START.")
    return [r for r in rows if t_start <= r.t <= t_end]


def read_rows_new8_from_text(text: str, shift_plus_1_hour: bool) -> Tuple[List[Row], int]:
    rows: List[Row] = []
    bad = 0
    for line in text.splitlines():
        p = parse_line_new8(line)
        if p is None:
            bad += 1
            continue
        y, doy, hour, imf, bz, vsw, ey, dst = p
        rows.append(
            Row(
                t=to_time_utc(y, doy, hour, shift_plus_1_hour),
                imf=imf,
                bz=bz,
                vsw=vsw,
                ey=ey,
                dst=dst,
            )
        )
    rows.sort(key=lambda r: r.t)
    return rows, bad


# ---------------------------
# 3) Stage 1: storm detection
# ---------------------------

def build_episodes_contiguous(rows: List[Row], episode_level: float) -> List[Tuple[int, int]]:
    """Build raw disturbed episodes using EPISODE_LEVEL."""
    episodes: List[Tuple[int, int]] = []
    in_ep = False
    start = 0

    for i, r in enumerate(rows):
        qualifies = r.dst <= episode_level

        if not in_ep:
            if qualifies:
                in_ep = True
                start = i
            continue

        prev = rows[i - 1]
        dt_hours = (r.t - prev.t).total_seconds() / 3600.0
        if (not qualifies) or (dt_hours > 1.0 + 1e-9):
            episodes.append((start, i - 1))
            in_ep = False
            if qualifies:
                in_ep = True
                start = i

    if in_ep:
        episodes.append((start, len(rows) - 1))
    return episodes


def apply_left_censored_stage1_boundary_rule(
    rows: List[Row],
    episodes_raw: List[Tuple[int, int]],
    storm_level: float,
) -> Tuple[List[Tuple[int, int]], int]:
    """Reject a first raw episode if the selected interval starts inside an active storm_level depression."""
    left_censored_boundary_episodes = 0
    if episodes_raw and rows and float(rows[0].dst) <= float(storm_level):
        first_a = episodes_raw[0][0]
        if int(first_a) == 0:
            episodes_raw = episodes_raw[1:]
            left_censored_boundary_episodes = 1
    return episodes_raw, left_censored_boundary_episodes


def find_episode_min_idx(rows: List[Row], a: int, b: int) -> int:
    """Return the row index of the deepest Dst value inside one episode."""
    imin = a
    for i in range(a, b + 1):
        if rows[i].dst < rows[imin].dst:
            imin = i
    return imin


def is_local_min_radius_stage1(rows: List[Row], i: int, a: int, b: int, radius_h: int) -> bool:
    """Check whether row i is a local Dst minimum within its episode-limited radius window."""
    lo = max(a, i - radius_h)
    hi = min(b, i + radius_h)
    v = rows[i].dst

    min_val = rows[lo].dst
    for j in range(lo + 1, hi + 1):
        if rows[j].dst < min_val:
            min_val = rows[j].dst

    if v != min_val:
        return False

    for j in range(lo, i):
        if rows[j].dst == min_val:
            return False
    return True


def storms_from_episodes(
    rows: List[Row],
    episodes_raw: List[Tuple[int, int]],
    storm_level: float,
    local_min_radius_hours: int,
    minima_split_factor: float,
    storm_limit: float,
) -> Tuple[List[Dict], List[Tuple[int, int]], List[Dict]]:
    """Convert raw disturbed episodes into retained storm_level episodes and Stage 1 storms."""
    storms: List[Dict] = []
    episodes_kept: List[Tuple[int, int]] = []
    multi_storm_details: List[Dict] = []

    new_ep_id = 0

    for (a, b) in episodes_raw:
        # First decide whether the minimum Dst value inside the episode reaches STORM_LEVEL.
        # Episodes where Dst never reaches STORM_LEVEL are not kept as storm episodes.
        imin_global = find_episode_min_idx(rows, a, b)
        global_min = rows[imin_global].dst

        if global_min > storm_level:
            continue

        new_ep_id += 1
        episodes_kept.append((a, b))

        # After the storm_level check, find local storm_level minima inside the kept episode.
        cand = []
        for i in range(a, b + 1):
            if rows[i].dst <= storm_level and is_local_min_radius_stage1(rows, i, a, b, local_min_radius_hours):
                cand.append(i)

        if not cand:
            cand = [imin_global]

        # Multiple candidate minima are separated only if the inter-minimum peak height between them is large enough.
        accepted = [cand[0]]

        for nxt in cand[1:]:
            prev = accepted[-1]
            lo = min(prev, nxt)
            hi = max(prev, nxt)

            if hi - lo <= 1:
                if rows[nxt].dst < rows[prev].dst:
                    accepted[-1] = nxt
                continue

            inter_minimum_peak = max(rows[j].dst for j in range(lo + 1, hi))
            m1 = rows[prev].dst
            m2 = rows[nxt].dst
            deeper_min = min(m1, m2)
            shallower_min = max(m1, m2)
            peak_height = inter_minimum_peak - shallower_min
            separation_threshold = (-deeper_min) * minima_split_factor

            if peak_height >= separation_threshold:
                accepted.append(nxt)
            else:
                if rows[nxt].dst < rows[prev].dst:
                    accepted[-1] = nxt

        minima_indices = [idx for idx in accepted if rows[idx].dst <= storm_level]
        if not minima_indices:
            minima_indices = [imin_global]

        # STORM_LIMIT imposes a lower bound on accepted Dst minima.
        minima_indices = [idx for idx in minima_indices if rows[idx].dst >= storm_limit]
        if not minima_indices:
            continue

        if len(minima_indices) >= 2:
            multi_storm_details.append(
                {
                    "episode_id": new_ep_id,
                    "start": rows[a].t,
                    "end": rows[b].t,
                    "global_min": global_min,
                    "minima": [(rows[i].t, rows[i].dst, i) for i in minima_indices],
                }
            )

        for idx in minima_indices:
            r = rows[idx]
            storms.append(
                {
                    "episode_id": new_ep_id,
                    "t_min": r.t,
                    "Dst_min": r.dst,
                    "row_index": idx,
                }
            )

    storms.sort(key=lambda s: s["t_min"])
    for sid, s in enumerate(storms, start=1):
        s["storm_id"] = sid

    return storms, episodes_kept, multi_storm_details


def storms_to_df(storms: List[Dict]) -> pd.DataFrame:
    """Convert the Stage 1 storm catalogue into a dataframe for display/download."""
    if not storms:
        return pd.DataFrame(columns=["storm_id", "episode_id", "Dst_min", "t_min"])
    out = []
    for s in storms:
        out.append(
            {
                "storm_id": s["storm_id"],
                "episode_id": s["episode_id"],
                "Dst_min": s["Dst_min"],
                "t_min": s["t_min"],
            }
        )
    return pd.DataFrame(out)


def episodes_to_df(rows: List[Row], episodes: List[Tuple[int, int]], storms: List[Dict]) -> pd.DataFrame:
    """Convert retained storm_level episodes into the Stage 1 episode table."""
    storm_count_by_episode: Dict[int, int] = {}
    for s in storms:
        ep_id = int(s.get("episode_id", 0))
        if ep_id > 0:
            storm_count_by_episode[ep_id] = storm_count_by_episode.get(ep_id, 0) + 1

    out = []
    for ep_id, (a, b) in enumerate(episodes, start=1):
        t0 = rows[a].t
        t1 = rows[b].t
        dur_h = (t1 - t0).total_seconds() / 3600.0 + 1.0
        imin = find_episode_min_idx(rows, a, b)
        global_min = rows[imin].dst
        out.append(
            {
                "episode_id": ep_id,
                "start_utc": t0,
                "end_utc": t1,
                "duration_hours": dur_h,
                "storm_count": int(storm_count_by_episode.get(ep_id, 0)),
                "global_min": dst_int(global_min),
                "t_min": rows[imin].t,
            }
        )
    return pd.DataFrame(out, columns=["episode_id", "start_utc", "end_utc", "duration_hours", "storm_count", "global_min", "t_min"])


def multi_storms_to_df(multi_storms: List[Dict]) -> pd.DataFrame:
    """Convert multi-storm episode details into one row per storm minimum."""
    columns = [
        "episode_id",
        "start_utc",
        "end_utc",
        "duration_hours",
        "storm_count",
        "global_min",
        "storm_number",
        "Dst_min",
        "t_min",
    ]
    out = []
    for d in multi_storms:
        start_utc = d["start"]
        end_utc = d["end"]
        duration_hours = (end_utc - start_utc).total_seconds() / 3600.0 + 1.0
        storm_count = len(d.get("minima", []))
        for k, (t, v, _idx) in enumerate(d["minima"], start=1):
            out.append(
                {
                    "episode_id": d["episode_id"],
                    "start_utc": start_utc,
                    "end_utc": end_utc,
                    "duration_hours": duration_hours,
                    "storm_count": int(storm_count),
                    "global_min": dst_int(d["global_min"]),
                    "storm_number": k,
                    "Dst_min": dst_int(v),
                    "t_min": t,
                }
            )
    return pd.DataFrame(out, columns=columns)


def stage1_outputs_to_summary(
    storms_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    multi_df: pd.DataFrame,
    left_censored_boundary_episodes: int,
) -> Dict:
    """Collect Stage 1 overview counts so the UI does not recalculate them."""
    multi_storm_episode_count = int(multi_df["episode_id"].nunique()) if (not multi_df.empty and "episode_id" in multi_df.columns) else 0
    return {
        "total_stage1_storms": int(len(storms_df)),
        "total_episodes": int(len(episodes_df)),
        "multi_storm_episodes": multi_storm_episode_count,
        "left_censored_boundary_episodes": int(left_censored_boundary_episodes),
    }


# ---------------------------
# 4) Stage 2: main phase filtering
# ---------------------------
# Stage 2 uses the Stage 1 storm minima and finds a main phase start for each storm.

def is_local_max_radius(rows: List[Row], i: int, radius_h: int) -> bool:
    """Check whether row i is a local Dst maximum; ties keep the latest maximum before Dst_min."""
    lo = max(0, i - radius_h)
    hi = min(len(rows) - 1, i + radius_h)
    v = float(rows[i].dst)

    max_val = float(rows[lo].dst)
    for j in range(lo + 1, hi + 1):
        if float(rows[j].dst) > max_val:
            max_val = float(rows[j].dst)

    if v != max_val:
        return False

    for j in range(i + 1, hi + 1):
        if float(rows[j].dst) == max_val:
            return False
    return True


def is_local_min_radius_stage2(rows: List[Row], i: int, radius_h: int) -> bool:
    lo = max(0, i - radius_h)
    hi = min(len(rows) - 1, i + radius_h)
    v = float(rows[i].dst)

    min_val = float(rows[lo].dst)
    for j in range(lo + 1, hi + 1):
        if float(rows[j].dst) < min_val:
            min_val = float(rows[j].dst)

    if v != min_val:
        return False

    for j in range(i + 1, hi + 1):
        if float(rows[j].dst) == min_val:
            return False
    return True


def calculate_weak_mark_factor(strong_storm_threshold: float, strong_mark: float) -> float:
    if abs(float(strong_storm_threshold)) < 1e-12:
        return float("nan")
    return float(strong_mark) / float(strong_storm_threshold)


def compute_mark(Dst_min: float, strong_storm_threshold: float, strong_mark: float) -> float:
    """Return the Dst reference mark used before accepting a local maximum as t_start."""
    weak_mark_factor = calculate_weak_mark_factor(strong_storm_threshold, strong_mark)
    if Dst_min <= strong_storm_threshold:
        return strong_mark
    return Dst_min * weak_mark_factor


def find_tstart_simple(
    rows: List[Row],
    imin: int,
    Dst_min: float,
    radius_max: int,
    strong_storm_threshold: float,
    strong_mark: float,
) -> Tuple[Optional[int], float]:
    """Search backward from Dst_min and return the row index of t_start, if found."""
    mark = compute_mark(Dst_min, strong_storm_threshold, strong_mark)

    if imin <= 0:
        return None, mark

    reached_mark = False

    # Backward search: start before Dst_min and move toward earlier times.
    for i in range(imin - 1, -1, -1):
        v = float(rows[i].dst)
        if not reached_mark:
            if v >= mark:
                reached_mark = True
            else:
                continue
        if is_local_max_radius(rows, i, radius_max):
            return i, mark

    return None, mark


def stage2_filter(
    rows: List[Row],
    storms: List[Dict],
    multi_storm_episodes: List[Dict],
    strong_storm_threshold: float,
    strong_mark: float,
    local_max_radius_hours: int,
    disturbance_filter: bool,
    disturbance_level: float,
    disturb_dip_count: int,
    disturb_dip_radius_hours: int,
    stage1_left_censored_boundary: bool = False,
):
    """Apply Stage 2 exclusions and return filtered storms plus diagnostic tables."""
    multi_storm_episode_ids = {d["episode_id"] for d in multi_storm_episodes}
    storms_ordered = sorted(storms, key=lambda s: (s["t_min"], int(s["row_index"]), int(s["storm_id"])))

    filtered_storms = []
    excluded = []
    disturbances = []
    previous_all_storms = []

    previous_storm_excluded_count = 0
    first_storm_no_tstart_boundary_count = 0

    for storm_pos, s in enumerate(storms_ordered):
        imin = int(s["row_index"])
        Dst_min = float(s["Dst_min"])
        t_min = s["t_min"]
        ep_id = int(s["episode_id"])
        is_multi_storm = ep_id in multi_storm_episode_ids

        istart, mark = find_tstart_simple(
            rows,
            imin,
            Dst_min,
            local_max_radius_hours,
            strong_storm_threshold,
            strong_mark,
        )

        
        
        
        # If no reliable t_start is found, the storm cannot enter Stage 2 outputs.
        if istart is None:
            if storm_pos == 0 and not bool(stage1_left_censored_boundary):
                first_storm_no_tstart_boundary_count += 1
            previous_all_storms.append(s)
            continue

        tstart = rows[istart].t
        Dst_start = float(rows[istart].dst)
        Mp_duration = (t_min - tstart).total_seconds() / 3600.0

        # Exclude a storm if the previous storm minimum falls inside its candidate main phase.
        blocking_storm = None
        for ps in previous_all_storms:
            prev_t_min = ps["t_min"]
            if tstart <= prev_t_min < t_min:
                blocking_storm = ps
                break

        if blocking_storm is not None:
            previous_storm_excluded_count += 1
            excluded.append(
                {
                    **s,
                    "is_multi_storm": is_multi_storm,
                    "mark": mark,
                    "t_start": tstart,
                    "Dst_start": Dst_start,
                    "Mp_duration": Mp_duration,
                    "blocking_storm_id": blocking_storm["storm_id"],
                    "blocking_storm_t_min": blocking_storm["t_min"],
                    "reason": "excluded_previous_storm_between_tstart_and_t_min",
                }
            )
            previous_all_storms.append(s)
            continue

        # Optional disturbance filter for weaker storms with repeated secondary dips before t_start.
        if disturbance_filter and Dst_min > disturbance_level:
            secondary_dip_count = 0
            disturb_dip_index = None
            disturb_dip_dst = None
            disturb_dip_time = None

            for i in range(imin - 1, istart, -1):
                if is_local_min_radius_stage2(rows, i, disturb_dip_radius_hours):
                    secondary_dip_count += 1
                    if disturb_dip_index is None:
                        disturb_dip_index = i
                        disturb_dip_dst = float(rows[i].dst)
                        disturb_dip_time = rows[i].t

            if secondary_dip_count >= disturb_dip_count:
                disturbances.append(
                    {
                        **s,
                        "is_multi_storm": is_multi_storm,
                        "mark": mark,
                        "t_start": tstart,
                        "Dst_start": Dst_start,
                        "Mp_duration": Mp_duration,
                        "disturb_dip_count": secondary_dip_count,
                        "disturb_dip_index": disturb_dip_index,
                        "disturb_dip_utc": disturb_dip_time,
                        "disturb_dip_dst": disturb_dip_dst,
                        "reason": f"disturbance_dipcount>={disturb_dip_count}_before_tstart_dipradius{disturb_dip_radius_hours}h",
                    }
                )
                previous_all_storms.append(s)
                continue

        filtered_storms.append(
            {
                **s,
                "is_multi_storm": is_multi_storm,
                "mark": mark,
                "t_start": tstart,
                "Dst_start": Dst_start,
                "Mp_duration": Mp_duration,
            }
        )
        previous_all_storms.append(s)

    summary = {
        "excluded_previous_storm": previous_storm_excluded_count,
        "first_storm_no_tstart_boundary": first_storm_no_tstart_boundary_count,
        "avg_mainphase_filtered": float(np.mean([x["Mp_duration"] for x in filtered_storms])) if filtered_storms else np.nan,
    }

    return filtered_storms, disturbances, excluded, summary


def stage2_outputs_to_df(
    filtered_storms: List[Dict],
    disturbances: List[Dict],
    excluded: List[Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert Stage 2 list outputs into Stage 2 dataframes."""
    filtered_df = pd.DataFrame(filtered_storms)
    disturbances_df = pd.DataFrame(disturbances)
    excluded_df = pd.DataFrame(excluded)
    return filtered_df, disturbances_df, excluded_df


# ---------------------------
# 5) Stage 3: Data-complete storms + metrics
# ---------------------------
# Stage 3 uses Stage 2 main phase intervals and checks solar-wind data completeness + computes metrics

def is_missing(val) -> bool:
    """Recognize OMNI missing-value sentinels used by the relevant parameters."""
    if val is None:
        return True
    try:
        v = float(val)
    except Exception:
        return True
    return (
        abs(v - 999.9) < 1e-6
        or abs(v - 999.99) < 1e-6
        or abs(v - 9999.0) < 1e-6
    )


def compute_data_complete_metrics(filtered_df: pd.DataFrame, omni_df: pd.DataFrame):
    """Build Stage 3 data-complete storms, main phase rows, peak delays, and metrics."""
    if filtered_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, pd.DataFrame(), pd.DataFrame(), {
            "avg_delay_Eyp": np.nan,
            "err_delay_Eyp": np.nan,
            "avg_delay_Bzp": np.nan,
            "err_delay_Bzp": np.nan,
        }

    omni = omni_df.copy()
    filtered = filtered_df.copy()

    filtered["t_start"] = pd.to_datetime(filtered["t_start"], utc=True)
    filtered["t_min"] = pd.to_datetime(filtered["t_min"], utc=True)
    omni["t_utc"] = pd.to_datetime(omni["t_utc"], utc=True)

    for c in ["IMF", "Bz", "Vsw", "Ey", "Dst"]:
        omni[c] = pd.to_numeric(omni[c], errors="coerce")

    data_complete_storms = []
    data_complete_rows = []
    replaced_values_log = []
    rejected_quality_log = []

    for _, s in filtered.iterrows():
        sid = int(s["storm_id"])
        tstart = s["t_start"]
        t_min = s["t_min"]

        # Stage 3 does not refind t_start; it re-extracts the OMNI rows inside the Stage 2 interval.
        mp = omni[(omni["t_utc"] >= tstart) & (omni["t_utc"] <= t_min)].copy()
        if mp.empty:
            continue
        mp.sort_values("t_utc", inplace=True)

        reject = False
        had_single_missing_replaced = False
        storm_replacements = []
        storm_missing_records = []

        # A storm remains data-complete if missing values are absent or only isolated single values can be replaced.
        for col in ["IMF", "Bz", "Vsw", "Ey"]:
            values = mp[col].to_numpy(copy=True)
            times = mp["t_utc"].to_numpy(copy=True)
            missing_idx = [i for i, v in enumerate(values) if is_missing(v)]
            for i in missing_idx:
                storm_missing_records.append(
                    {
                        "storm_id": sid,
                        "episode_id": int(s["episode_id"]),
                        "Dst_min": float(s["Dst_min"]) if pd.notna(s["Dst_min"]) else np.nan,
                        "parameter": col,
                        "t_utc": pd.to_datetime(times[i], utc=True),
                        "missing_value": values[i],
                        "row_index": i,
                    }
                )

            col_reject = False
            if missing_idx:
                runs = 1
                for i in range(1, len(missing_idx)):
                    if missing_idx[i] == missing_idx[i - 1] + 1:
                        runs += 1
                        if runs >= 2:
                            reject = True
                            col_reject = True
                            break
                    else:
                        runs = 1

            if not col_reject:
                for i in missing_idx:
                    if i == 0 or i == len(values) - 1:
                        reject = True
                        col_reject = True
                        break
                    prev = values[i - 1]
                    nxt = values[i + 1]
                    if is_missing(prev) or is_missing(nxt):
                        reject = True
                        col_reject = True
                        break
                    replaced_val = (float(prev) + float(nxt)) / 2.0
                    storm_replacements.append(
                        {
                            "storm_id": sid,
                            "episode_id": int(s["episode_id"]),
                            "parameter": col,
                            "t_utc": pd.to_datetime(times[i], utc=True),
                            "orig_value": values[i],
                            "replaced_value": replaced_val,
                            "row_index": i,
                        }
                    )
                    values[i] = replaced_val
                    had_single_missing_replaced = True

            if not col_reject:
                mp[col] = values

        if reject:
            rejected_quality_log.append(
                {
                    "storm_id": sid,
                    "Dst_min": float(s["Dst_min"]) if pd.notna(s["Dst_min"]) else np.nan,
                    "t_min": pd.to_datetime(s["t_min"], utc=True) if "t_min" in s else pd.NaT,
                    "missing_value_hours": len({pd.to_datetime(r["t_utc"], utc=True) for r in storm_missing_records}) if storm_missing_records else 0,
                    "failed_parameters": ",".join(sorted({r["parameter"] for r in storm_missing_records})) if storm_missing_records else "",
                }
            )
            continue

        data_complete_storms.append(s.to_dict())
        if had_single_missing_replaced:
            replaced_values_log.extend(storm_replacements)

        mp["storm_id"] = sid
        mp["episode_id"] = int(s["episode_id"])
        data_complete_rows.append(mp)

    data_complete_storms_df = pd.DataFrame(data_complete_storms) if data_complete_storms else pd.DataFrame(columns=filtered.columns)
    data_complete_rows_df = pd.concat(data_complete_rows, ignore_index=True) if data_complete_rows else pd.DataFrame()
    replacements_df = pd.DataFrame(replaced_values_log)
    rejected_quality_df = pd.DataFrame(rejected_quality_log)

    if not data_complete_storms_df.empty:
        data_complete_storms_df.sort_values("Dst_min", inplace=True)
        data_complete_storms_df["t_start"] = pd.to_datetime(data_complete_storms_df["t_start"], utc=True)
        data_complete_storms_df["t_min"] = pd.to_datetime(data_complete_storms_df["t_min"], utc=True)

    if data_complete_rows_df.empty:
        return data_complete_storms_df, data_complete_rows_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), rejected_quality_df, {
            "avg_delay_Eyp": np.nan,
            "err_delay_Eyp": np.nan,
            "avg_delay_Bzp": np.nan,
            "err_delay_Bzp": np.nan,
        }

    data_complete_rows_df["t_utc"] = pd.to_datetime(data_complete_rows_df["t_utc"], utc=True)
    for c in ["Dst", "IMF", "Bz", "Vsw", "Ey"]:
        data_complete_rows_df[c] = pd.to_numeric(data_complete_rows_df[c], errors="coerce")

    rows_by_storm = {int(k): g.sort_values("t_utc") for k, g in data_complete_rows_df.groupby("storm_id")}

    # Metrics are calculated only for storms that passed the data-completeness check.
    peak_delays_out = []
    peaks_integrals_out = []

    for _, s in data_complete_storms_df.iterrows():
        sid = int(s["storm_id"])
        ep = int(s["episode_id"])
        tstart = pd.to_datetime(s["t_start"], utc=True)
        t_min = pd.to_datetime(s["t_min"], utc=True)
        Dst_min = float(s["Dst_min"]) if pd.notna(s["Dst_min"]) else np.nan

        g = rows_by_storm.get(sid)
        if g is None or g.empty:
            continue
        mp = g[(g["t_utc"] >= tstart) & (g["t_utc"] <= t_min)].copy()
        if mp.empty:
            continue

        Mp_duration = (t_min - tstart).total_seconds() / 3600.0
        mp["Ey_inj"] = mp["Ey"].clip(lower=0)

        if mp["IMF"].notna().any():
            i_IMFp = mp["IMF"].idxmax()
            IMFp = float(mp.loc[i_IMFp, "IMF"])
            t_IMFp = mp.loc[i_IMFp, "t_utc"]
        else:
            IMFp = np.nan
            t_IMFp = pd.NaT

        if mp["Bz"].notna().any():
            i_Bzp = mp["Bz"].idxmin()
            Bzp = float(mp.loc[i_Bzp, "Bz"])
            t_Bzp = mp.loc[i_Bzp, "t_utc"]
            delay_bz_h = (t_min - t_Bzp).total_seconds() / 3600.0
        else:
            Bzp = np.nan
            t_Bzp = pd.NaT
            delay_bz_h = np.nan

        if mp["Vsw"].notna().any():
            i_Vp = mp["Vsw"].idxmax()
            Vp = float(mp.loc[i_Vp, "Vsw"])
            t_Vp = mp.loc[i_Vp, "t_utc"]
        else:
            Vp = np.nan
            t_Vp = pd.NaT

        if mp["Ey_inj"].notna().any():
            i_Eyp = mp["Ey_inj"].idxmax()
            Eyp = float(mp.loc[i_Eyp, "Ey_inj"])
            t_Eyp = mp.loc[i_Eyp, "t_utc"]
            delay_ey_h = (t_min - t_Eyp).total_seconds() / 3600.0
        else:
            Eyp = np.nan
            t_Eyp = pd.NaT
            delay_ey_h = np.nan

        if pd.notna(t_Bzp) and pd.notna(t_Eyp):
            abs_bz_ey_h = abs((t_Bzp - t_Eyp).total_seconds() / 3600.0)
        else:
            abs_bz_ey_h = np.nan

        peak_delays_out.append(
            {
                "storm_id": sid,
                "episode_id": ep,
                "t_start": tstart,
                "t_min": t_min,
                "Dst_min": Dst_min,
                "Mp_duration": Mp_duration,
                "Eyp": Eyp,
                "t_Eyp": t_Eyp,
                "delay_Eyp_to_Dst_min_hours": delay_ey_h,
                "Bzp": Bzp,
                "t_Bzp": t_Bzp,
                "delay_Bzp_to_Dst_min_hours": delay_bz_h,
                "t_min_minus_t_Bzp": delay_bz_h,
                "t_min_minus_t_Eyp": delay_ey_h,
                "abs_t_Bzp_minus_t_Eyp": abs_bz_ey_h,
                "mainphase_rows": len(mp),
            }
        )

        peaks_integrals_out.append(
            {
                "storm_id": sid,
                "episode_id": ep,
                "t_start": tstart,
                "t_min": t_min,
                "Dst_min": Dst_min,
                "abs_Dst_min": abs(Dst_min) if pd.notna(Dst_min) else np.nan,
                "Mp_duration": Mp_duration,
                "IMFp": IMFp,
                "t_IMFp": t_IMFp,
                "Bzp": Bzp,
                "t_Bzp": t_Bzp,
                "Vp": Vp,
                "t_Vp": t_Vp,
                "Eyp": Eyp,
                "t_Eyp": t_Eyp,
                "Eyi": mp["Ey_inj"].sum(skipna=True),
                "delay_Eyp_to_Dst_min_hours": delay_ey_h,
                "delay_Bzp_to_Dst_min_hours": delay_bz_h,
                "mainphase_rows": len(mp),
            }
        )

    df_delays = pd.DataFrame(peak_delays_out)
    df_metrics = pd.DataFrame(peaks_integrals_out)

    if not df_metrics.empty:
        df_metrics["class"] = df_metrics["Dst_min"].apply(class_name)
    else:
        df_metrics["class"] = pd.Series(dtype="object")

    # Summary values are returned once so the UI can display them without recalculating.
    summary = {
        "avg_delay_Eyp": float(df_delays["delay_Eyp_to_Dst_min_hours"].mean()) if not df_delays.empty else np.nan,
        "err_delay_Eyp": sem(df_delays["delay_Eyp_to_Dst_min_hours"]) if not df_delays.empty else np.nan,
        "avg_delay_Bzp": float(df_delays["delay_Bzp_to_Dst_min_hours"].mean()) if not df_delays.empty else np.nan,
        "err_delay_Bzp": sem(df_delays["delay_Bzp_to_Dst_min_hours"]) if not df_delays.empty else np.nan,
    }

    return data_complete_storms_df, data_complete_rows_df, df_delays, df_metrics, replacements_df, rejected_quality_df, summary


# ---------------------------
# 6) Additional outputs, table formatting, sorting, plots, and downloads
# ---------------------------


def pearson_safe(x: pd.Series, y: pd.Series):
    df = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    n = len(df)
    if n < 3:
        return np.nan, np.nan, n
    if df.iloc[:, 0].nunique() < 2 or df.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, n
    r, p = pearsonr(df.iloc[:, 0].values, df.iloc[:, 1].values)
    return float(r), float(p), n


def sem(series) -> float:
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    n = len(s)
    if n <= 1:
        return np.nan
    return float(s.std(ddof=1) / np.sqrt(n))


def class_name(Dst_min: float) -> str:
    if pd.isna(Dst_min):
        return "other"
    if Dst_min <= -250.0:
        return "super-storm (Dst_min <= -250)"
    if Dst_min <= -100.0:
        return "intense (-250 < Dst_min <= -100)"
    if Dst_min <= -50.0:
        return "moderate (-100 < Dst_min <= -50)"
    return "other"


def class_order_key(name: str) -> int:
    if str(name).startswith("moderate"):
        return 0
    if str(name).startswith("intense"):
        return 1
    if str(name).startswith("super-storm"):
        return 2
    return 99


def count_by_class(df: pd.DataFrame, count_label: str) -> pd.DataFrame:
    classes = [
        "moderate (-100 < Dst_min <= -50)",
        "intense (-250 < Dst_min <= -100)",
        "super-storm (Dst_min <= -250)",
    ]
    out = pd.DataFrame({"class": classes})
    if df.empty or "Dst_min" not in df.columns:
        out[count_label] = 0
        return out

    tmp = df.copy()
    tmp["class"] = tmp["Dst_min"].apply(class_name)
    tmp = tmp[tmp["class"] != "other"]

    if tmp.empty:
        out[count_label] = 0
        return out

    grp = tmp.groupby("class", dropna=False).size().reset_index(name=count_label)
    out = out.merge(grp, on="class", how="left")
    out[count_label] = out[count_label].fillna(0).astype(int)
    out["__order"] = out["class"].map(class_order_key)
    out = out.sort_values(["__order", "class"]).drop(columns="__order")
    return out


def summarize_by_class(df: pd.DataFrame, count_label: str, avg_cols: List[Tuple[str, str]]) -> pd.DataFrame:
    classes = [
        "moderate (-100 < Dst_min <= -50)",
        "intense (-250 < Dst_min <= -100)",
        "super-storm (Dst_min <= -250)",
    ]
    out = pd.DataFrame({"class": classes})

    if df.empty or "class" not in df.columns:
        out[count_label] = 0
        for _, label in avg_cols:
            out[label] = 0
        return out

    tmp = df.copy()
    tmp = tmp[tmp["class"] != "other"]

    if tmp.empty:
        out[count_label] = 0
        for _, label in avg_cols:
            out[label] = 0
        return out

    grouped = tmp.groupby("class", dropna=False)
    grp = grouped.size().reset_index(name=count_label)
    out = out.merge(grp, on="class", how="left")

    for src_col, label in avg_cols:
        if src_col in tmp.columns:
            means = grouped[src_col].mean().reset_index(name=label)
            out = out.merge(means, on="class", how="left")
        else:
            out[label] = 0

    out[count_label] = out[count_label].fillna(0).astype(int)
    for _, label in avg_cols:
        out[label] = out[label].fillna(0)

    out["__order"] = out["class"].map(class_order_key)
    out = out.sort_values(["__order", "class"]).drop(columns="__order")
    return out


def correlations_selected(df_metrics: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        ("Bzp", "Bzp"),
        ("Eyp", "Eyp"),
        ("Eyi", "Eyi"),
    ]
    rows = []
    if df_metrics.empty:
        return pd.DataFrame(
            [{"Parameter vs |Dst_min|": label, "R": 0, "N": 0} for _, label in wanted]
        )
    for src, label in wanted:
        if src not in df_metrics.columns:
            rows.append({"Parameter vs |Dst_min|": label, "R": 0, "N": 0})
            continue
        r, _, n = pearson_safe(df_metrics[src], df_metrics["abs_Dst_min"])
        rows.append({"Parameter vs |Dst_min|": label, "R": 0 if pd.isna(r) else r, "N": n})
    return pd.DataFrame(rows)


def format_val(label, val):
    try:
        label_l = str(label).lower()
        
        if any(k in label_l for k in ["count", "total storms", "storms", "episodes"]):
            return "—" if pd.isna(val) else str(int(round(float(val))))
        
        if isinstance(val, (int,)) or (isinstance(val, float) and float(val).is_integer()):
            return "—" if pd.isna(val) else str(int(val))
        return f"{float(val):.3f}"
    except:
        return str(val)


def format_mean_pm_error(mean_val, err_val, digits: int = 2) -> str:
    if mean_val is None or pd.isna(mean_val):
        return "—"
    if err_val is None or pd.isna(err_val):
        return f"{float(mean_val):.{digits}f}"
    return f"{float(mean_val):.{digits}f} ± {float(err_val):.{digits}f}"
def pretty_value(x):
    if x is None:
        return "—"
    if isinstance(x, (pd.Timestamp, datetime)):
        return fmt_dt(x)
    if isinstance(x, str):
        xs = x.strip()
        if xs:
            dt = pd.to_datetime(xs, utc=True, errors="coerce")
            if not pd.isna(dt):
                return fmt_dt(dt)
        return x
    if pd.isna(x):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        if abs(float(x) - round(float(x))) < 1e-9:
            return f"{int(round(float(x))):,}"
        return f"{float(x):.3f}"
    return x


def build_stage3_peak_delays_table(peak_delays_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["storm_id", "Dst_min", "t_min", "Mp_duration", "t_min_minus_t_Bzp", "t_min_minus_t_Eyp", "abs_t_Bzp_minus_t_Eyp"]
    display_names = {
        "t_min_minus_t_Bzp": "t_min − t_Bzp",
        "t_min_minus_t_Eyp": "t_min − t_Eyp",
        "abs_t_Bzp_minus_t_Eyp": "|t_Bzp − t_Eyp|",
    }
    if peak_delays_df is None or peak_delays_df.empty:
        return pd.DataFrame(columns=[display_names.get(c, c) for c in cols])

    df = peak_delays_df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].copy().rename(columns=display_names)


def build_stage3_full_peak_data_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["storm_id", "Dst_min", "t_start", "t_min", "t_IMFp", "t_Bzp", "t_Vp", "t_Eyp"]
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(columns=cols)

    df = metrics_df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NaT if c.startswith("t_") else np.nan
    return df[cols].copy()


def build_stage3_rejected_quality_table(rejected_quality_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["storm_id", "Dst_min", "t_min", "missing_value_hours", "failed_parameters"]
    if rejected_quality_df is None or rejected_quality_df.empty:
        return pd.DataFrame(columns=cols)
    df = rejected_quality_df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c == "failed_parameters" else np.nan
    return df[cols].copy()

def build_stage3_replacements_table(replacements_df: pd.DataFrame, data_complete_storms_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["storm_id", "Dst_min", "parameter", "t_utc", "error_value", "replaced_value"]
    if replacements_df is None or replacements_df.empty:
        return pd.DataFrame(columns=cols)

    df = replacements_df.copy()

    
    rename_map = {}

    
    if "tutc" in df.columns:
        rename_map["tutc"] = "t_utc"

    
    if "param" in df.columns:
        rename_map["param"] = "parameter"

    
    if "orig_value" in df.columns:
        rename_map["orig_value"] = "error_value"
    elif "original_value" in df.columns:
        rename_map["original_value"] = "error_value"

    
    if "new_value" in df.columns:
        rename_map["new_value"] = "replaced_value"

    
    if "idx" in df.columns:
        rename_map["idx"] = "row_index"

    df = df.rename(columns=rename_map)

    
    if "Dst_min" not in df.columns and "storm_id" in df.columns:
        if data_complete_storms_df is not None and not data_complete_storms_df.empty:
            df = df.merge(data_complete_storms_df[["storm_id", "Dst_min"]], on="storm_id", how="left")

    
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NaT if c == "t_utc" else None

    return df[cols].copy()


def build_data_complete_storms_table(data_complete_storms_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    if data_complete_storms_df is None or data_complete_storms_df.empty:
        return pd.DataFrame(columns=[
            "storm_id", "Dst_min", "t_min", "Dst_start", "t_start", "Mp_duration",
            "IMFp", "Bzp", "Vp", "Eyp", "Eyi"
        ])

    base = data_complete_storms_df.copy()

    keep_base = [c for c in ["storm_id", "Dst_min", "t_min", "Dst_start", "t_start", "Mp_duration"] if c in base.columns]
    base = base[keep_base].copy()

    if metrics_df is not None and not metrics_df.empty:
        metrics_keep = [c for c in ["storm_id", "IMFp", "Bzp", "Vp", "Eyp", "Eyi"] if c in metrics_df.columns]
        metrics_part = metrics_df[metrics_keep].copy()
        merged = base.merge(metrics_part, on="storm_id", how="left")
    else:
        merged = base.copy()
        for c in ["IMFp", "Bzp", "Vp", "Eyp", "Eyi"]:
            if c not in merged.columns:
                merged[c] = np.nan

    desired = ["storm_id", "Dst_min", "t_min", "Dst_start", "t_start", "Mp_duration",
               "IMFp", "Bzp", "Vp", "Eyp", "Eyi"]
    desired = [c for c in desired if c in merged.columns]
    rest = [c for c in merged.columns if c not in desired]
    return merged[desired + rest]

def format_corr_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()

    
    desired_order = ["IMFp", "Bzp", "Eyp", "Eyi", "Vp", "Mp_duration"]
    order_map = {k.lower(): i for i, k in enumerate(desired_order)}
    if df.shape[1] > 0:
        df["__order"] = df.iloc[:, 0].map(lambda x: order_map.get(str(x).strip().lower(), 999))
        df = df.sort_values("__order", kind="stable").drop(columns="__order")
    
    df = df[~df.iloc[:,0].astype(str).str.contains("delay", case=False, na=False)]
    if df.empty:
        return df
    show_df = df.copy()
    if "R" in show_df.columns:
        show_df["R"] = show_df["R"].map(lambda v: "—" if pd.isna(v) else f"{float(v):.3f}")
    if "p_value" in show_df.columns:
        show_df["p_value"] = show_df["p_value"].map(lambda v: "—" if pd.isna(v) else f"{float(v):.2e}")
    if "N" in show_df.columns:
        show_df["N"] = show_df["N"].map(lambda v: "—" if pd.isna(v) else str(int(v)))
    for col in show_df.columns:
        if col not in {"R", "p_value", "N"}:
            show_df[col] = show_df[col].map(pretty_value)
    return show_df


def _static_table_cell_html(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "&mdash;"
    text_value = escape(str(value))
    return text_value.replace("\n", "<br>")


def render_static_scroll_table(df: pd.DataFrame, max_height: int = 420, key: Optional[str] = None, fit_to_container: bool = False, equal_col_widths: bool = False):
    if df is None or df.empty:
        st.info("No data.")
        return

    table_id = key or f"static_table_{uuid4().hex}"
    table_width_css = "100%" if fit_to_container else "max-content"
    table_min_width_css = "100%" if fit_to_container else "100%"
    table_layout_css = "fixed" if (fit_to_container and equal_col_widths) else "auto"
    col_count = max(1, len(df.columns))
    col_width_css = f"{100.0 / col_count:.6f}%" if (fit_to_container and equal_col_widths) else "auto"
    header_html = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)

    body_rows = []
    for row in df.itertuples(index=False, name=None):
        cells = "".join(f"<td>{_static_table_cell_html(value)}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body_html = "".join(body_rows)

    html = f"""
    <style>
    div#{table_id}_outer {{
        width: 100%;
        max-width: 100%;
        background: #ffffff;
        border: 1px solid #000000;
        border-radius: 10px;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.05);
        padding: 0;
        overflow: hidden;
    }}

    div#{table_id}_scroll {{
        width: 100%;
        max-width: 100%;
        background: transparent;
        border-radius: 7px;
        overflow-x: auto;
        overflow-y: auto;
        max-height: {max_height}px;
    }}

    table#{table_id} {{
        border-collapse: separate;
        border-spacing: 0;
        width: {table_width_css};
        min-width: {table_min_width_css};
        table-layout: {table_layout_css};
        font-size: 0.95rem;
        background: white;
    }}

    table#{table_id} thead th {{
        position: sticky;
        top: 0;
        z-index: 2;
        background: white;
        color: black;
        font-weight: 600;
        cursor: default !important;
        pointer-events: none !important;
        user-select: none !important;
    }}

    table#{table_id} th,
    table#{table_id} td {{
        width: {col_width_css};
        max-width: {col_width_css};
        padding: 0.45rem 0.65rem;
        text-align: left;
        white-space: nowrap;
        border-bottom: 1px solid #e5e7eb;
        border-right: 1px solid #f3f4f6;
        background: white;
    }}

    table#{table_id} th:last-child,
    table#{table_id} td:last-child {{
        border-right: none;
    }}

    table#{table_id} tbody tr:last-child td {{
        border-bottom: 1px solid #e5e7eb;
    }}

    table#{table_id} tbody tr:nth-child(even) td {{
        background: #fafafa;
    }}
    </style>
    <div id="{table_id}_outer">
        <div id="{table_id}_scroll">
            <table id="{table_id}">
                <thead>
                    <tr>{header_html}</tr>
                </thead>
                <tbody>
                    {body_html}
                </tbody>
            </table>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.markdown('<div style="height:30px"></div>', unsafe_allow_html=True)


def render_table(title: str, df: pd.DataFrame, one_decimal_columns: Optional[set] = None):
    if df.empty:
        st.info("No data.")
    else:
        show_df = df.copy()
        one_decimal_columns = {str(c).lower() for c in (one_decimal_columns or set())}
        for col in show_df.columns:
            col_l = str(col).lower()
            if col_l in one_decimal_columns and pd.api.types.is_numeric_dtype(show_df[col]):
                show_df[col] = show_df[col].map(lambda x: "—" if pd.isna(x) else f"{float(x):.1f}")
            elif "avg" in col_l and pd.api.types.is_numeric_dtype(show_df[col]):
                show_df[col] = show_df[col].map(lambda x: "—" if pd.isna(x) else f"{float(x):.2f}")
            else:
                show_df[col] = show_df[col].map(pretty_value)
        render_static_scroll_table(show_df, key=f"{title.lower().replace(' ', '_')}_{uuid4().hex}")


def render_data_complete_storms_table(title: str, df: pd.DataFrame):
    if df.empty:
        st.info("No data.")
    else:
        show_df = df.copy()
        float_cols_2dec = {"Bzp", "IMFp", "Eyp", "Eyi"}
        for col in show_df.columns:
            col_l = str(col).lower()
            if pd.api.types.is_datetime64_any_dtype(show_df[col]):
                show_df[col] = pd.to_datetime(show_df[col], utc=True, errors="coerce").map(fmt_dt)
            elif str(col) in float_cols_2dec:
                show_df[col] = show_df[col].map(lambda x: "—" if pd.isna(x) else f"{float(x):.2f}")
            elif col_l == "vp":
                show_df[col] = show_df[col].map(lambda x: "—" if pd.isna(x) else str(int(round(float(x)))))
            elif col == "storm_id":
                show_df[col] = show_df[col].map(lambda x: "—" if pd.isna(x) else f"{int(x)}")
            else:
                show_df[col] = show_df[col].map(pretty_value)
        render_static_scroll_table(show_df, key=f"{title.lower().replace(' ', '_')}_{uuid4().hex}")


def sort_stage1_storm_minima(df: pd.DataFrame, order_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    order_choice = str(order_choice).strip()
    out = df.copy()
    if order_choice.lower() in ("time", "date") and "t_min" in out.columns:
        return out.sort_values("t_min", ascending=True).reset_index(drop=True)
    if order_choice == "Dst_min" and "Dst_min" in out.columns:
        
        return out.sort_values("Dst_min", ascending=True).reset_index(drop=True)
    return out.reset_index(drop=True)


def sort_stage1_episodes(df: pd.DataFrame, order_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    order_choice = str(order_choice).strip()
    out = df.copy()

    if order_choice.lower() in ("time", "date", "data"):
        time_col = None
        for candidate in ["start_utc", "start", "episode_start", "start_time"]:
            if candidate in out.columns:
                time_col = candidate
                break
        if time_col:
            return out.sort_values(by=time_col, ascending=True).reset_index(drop=True)
        return out.reset_index(drop=True)

    if order_choice.lower() in ("duration", "duration_hours"):
        duration_col = None
        for candidate in ["duration_hours", "duration_h", "duration"]:
            if candidate in out.columns:
                duration_col = candidate
                break
        if duration_col:
            return out.sort_values(by=duration_col, ascending=False).reset_index(drop=True)
        return out.reset_index(drop=True)

    if order_choice == "global_min" and "global_min" in out.columns:
        sort_cols = ["global_min"]
        ascending = [True]
        for candidate in ["start_utc", "start", "episode_start", "start_time"]:
            if candidate in out.columns:
                sort_cols.append(candidate)
                ascending.append(True)
                break
        return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return out.reset_index(drop=True)


def sort_stage1_multi_storm_minima(df: pd.DataFrame, order_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    order_choice = str(order_choice).strip()
    out = df.copy()
    time_col = "t_min" if "t_min" in out.columns else None

    if order_choice.lower() in ("time", "date") and time_col:
        return out.sort_values(time_col, ascending=True).reset_index(drop=True)

    if order_choice.lower() in ("duration", "duration_hours"):
        duration_col = None
        for candidate in ["duration_hours", "duration_h", "duration"]:
            if candidate in out.columns:
                duration_col = candidate
                break
        if duration_col:
            sort_cols = [duration_col]
            ascending = [False]
            if time_col:
                sort_cols.append(time_col)
                ascending.append(True)
            return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        return out.reset_index(drop=True)

    if order_choice == "global_min" and "global_min" in out.columns:
        
        sort_cols = ["global_min"]
        ascending = [True]
        if time_col:
            sort_cols.append(time_col)
            ascending.append(True)
        return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return out.reset_index(drop=True)


def sort_stage2_table(df: pd.DataFrame, order_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    order_choice = str(order_choice).strip()
    out = df.copy()

    if order_choice.lower() in ("time", "date") and "t_min" in out.columns:
        return out.sort_values("t_min", ascending=True).reset_index(drop=True)

    if order_choice == "Dst_min" and "Dst_min" in out.columns:
        
        sort_cols = ["Dst_min"]
        ascending = [True]
        if "t_min" in out.columns:
            sort_cols.append("t_min")
            ascending.append(True)
        return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    if order_choice == "Mp_duration":
        duration_col = None
        for candidate in ["Mp_duration"]:
            if candidate in out.columns:
                duration_col = candidate
                break
        if duration_col:
            sort_cols = [duration_col]
            ascending = [False]
            if "t_min" in out.columns:
                sort_cols.append("t_min")
                ascending.append(True)
            return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return out.reset_index(drop=True)


def sort_by_time_generic(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for col in ["t_min", "t_start", "utc", "time", "datetime"]:
        if col in df.columns:
            return df.sort_values(by=col, ascending=True).reset_index(drop=True)
    return df

def sort_data_complete_storms(df: pd.DataFrame, order_choice: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    order_choice = str(order_choice).strip()
    normalized_order_choice = {
        "date": "date",
        "time": "date",
        "dst_min": "Dst_min",
        "imfp": "IMFp",
        "bzp": "Bzp",
        "eyp": "Eyp",
        "vp": "Vp",
        "eyi": "Eyi",
    }.get(order_choice.lower(), order_choice)
    out = df.copy()

    if normalized_order_choice == "date" and "t_min" in out.columns:
        return out.sort_values("t_min", ascending=True).reset_index(drop=True)

    if normalized_order_choice == "Dst_min" and "Dst_min" in out.columns:
        sort_cols = ["Dst_min"]
        ascending = [True]
        if "t_min" in out.columns:
            sort_cols.append("t_min")
            ascending.append(True)
        return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    if normalized_order_choice == "Mp_duration":
        for candidate in ["Mp_duration"]:
            if candidate in out.columns:
                sort_cols = [candidate]
                ascending = [False]
                if "t_min" in out.columns:
                    sort_cols.append("t_min")
                    ascending.append(True)
                return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    
    
    
    order_map = {
        "IMFp": ("IMFp", False),
        "Bzp": ("Bzp", True),
        "Vp": ("Vp", False),
        "Eyp": ("Eyp", False),
        "Eyi": ("Eyi", False),
    }
    if normalized_order_choice in order_map:
        col, asc = order_map[normalized_order_choice]
        if col in out.columns:
            sort_cols = [col]
            ascending = [asc]
            if "t_min" in out.columns:
                sort_cols.append("t_min")
                ascending.append(True)
            return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return out.reset_index(drop=True)


def reorder_stage2_display(df: pd.DataFrame, excluded: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    desired = ["storm_id", "Dst_min", "t_min", "Dst_start", "t_start", "Mp_duration"]
    front = [c for c in desired if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    if excluded and len(rest) >= 3:
        tail = rest[-3:]
        middle = rest[:-3]
        return df[front + middle + tail]
    return df[front + rest]


def compute_correlations(df_metrics: pd.DataFrame) -> pd.DataFrame:
    
    
    
    metrics = [
        "IMFp",
        "Bzp",
        "Eyp",
        "Eyi",
        "Vp",
        "Mp_duration",
        "delay_Eyp_to_Dst_min_hours",
        "delay_Bzp_to_Dst_min_hours",
    ]
    out = []
    if df_metrics.empty:
        return pd.DataFrame([{"Parameter vs |Dst_min|": m, "R": np.nan, "p_value": np.nan, "N": 0} for m in metrics])
    for m in metrics:
        if m not in df_metrics.columns:
            out.append({"Parameter vs |Dst_min|": m, "R": np.nan, "p_value": np.nan, "N": 0})
            continue
        r, p, n = pearson_safe(df_metrics[m], df_metrics["abs_Dst_min"])
        out.append({
            "Parameter vs |Dst_min|": m,
            "R": r,
            "p_value": p,
            "N": n
        })
    return pd.DataFrame(out)


def render_correlation_plots(metrics_df: pd.DataFrame):
    plot_specs = [
        ("IMFp", "IMFp"),
        ("Bzp", "Bzp"),
        ("Eyp", "Eyp"),
        ("Eyi", "Eyi"),
        ("Vp", "Vp"),
        ("Mp_duration", "Mp_duration"),
    ]

    
    
    plot_figsize = (4.5, 3.2)

    if metrics_df is None or metrics_df.empty or "abs_Dst_min" not in metrics_df.columns:
        st.info("No correlation plots available.")
        return

    plots = []
    for metric_col, label in plot_specs:
        if metric_col not in metrics_df.columns:
            continue

        plot_df = metrics_df[["abs_Dst_min", metric_col]].dropna().copy()
        if plot_df.empty:
            continue

        plots.append((metric_col, label, plot_df))

    for i in range(0, len(plots), 2):
        cols = st.columns(2)

        for j in range(2):
            if i + j >= len(plots):
                break

            metric_col, label, plot_df = plots[i + j]
            x = plot_df["abs_Dst_min"].to_numpy(dtype=float)
            y = plot_df[metric_col].to_numpy(dtype=float)

            fig, ax = plt.subplots(figsize=plot_figsize)
            fig.set_layout_engine(None)
            ax.set_position([0.16, 0.18, 0.78, 0.68])  
            ax.scatter(x, y, color='red', marker='o', s=20)

            r, _, n = pearson_safe(plot_df[metric_col], plot_df["abs_Dst_min"])
            title = f"{label} vs |Dst_min|"
            if pd.notna(r):
                title += f" (R={float(r):.3f}, N={int(n)})"
            ax.set_title(title)
            ax.set_xlabel("|Dst_min|")
            ax.set_ylabel(label)

            if pd.notna(r) and abs(float(r)) > 0.5 and len(plot_df) >= 2:
                try:
                    m, b = np.polyfit(x, y, 1)
                    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                    yfit = m * xfit + b
                    ax.plot(xfit, yfit, color='black')

                    
                    xeq = xfit[int(len(xfit) * 0.60)]
                    yeq = m * xeq + b
                    sign = "+" if b >= 0 else "-"
                    eq_text = f"y = {m:.3f}x {sign} {abs(b):.3f}"

                    ax.text(xeq, yeq, eq_text)
                except Exception:
                    pass

            cols[j].pyplot(fig, clear_figure=True)


def validate_parameter_mainphase_values(mp: pd.DataFrame, col: str) -> Tuple[bool, Optional[np.ndarray]]:
    """Return validated values for one solar-wind parameter during one storm main phase.

    The rule matches the Stage 3 data-completeness check for one parameter: a single internal missing value is replaced by the average of
    its two neighbours; edge missing values or consecutive missing values fail.
    """
    if mp is None or mp.empty or col not in mp.columns:
        return False, None

    values = pd.to_numeric(mp[col], errors="coerce").to_numpy(copy=True)
    missing_idx = [i for i, v in enumerate(values) if is_missing(v)]

    if not missing_idx:
        return True, values

    run_len = 1
    for i in range(1, len(missing_idx)):
        if missing_idx[i] == missing_idx[i - 1] + 1:
            run_len += 1
            if run_len >= 2:
                return False, None
        else:
            run_len = 1

    for i in missing_idx:
        if i == 0 or i == len(values) - 1:
            return False, None
        prev = values[i - 1]
        nxt = values[i + 1]
        if is_missing(prev) or is_missing(nxt):
            return False, None
        values[i] = (float(prev) + float(nxt)) / 2.0

    return True, values


PARAMETER_PEAK_TABLE_CONFIGS = {
    "imf": {"source_col": "IMF", "value_col": "IMFp", "mode": "max", "columns": ["storm_id", "Dst_min", "IMFp"]},
    "bz": {"source_col": "Bz", "value_col": "Bzp", "mode": "min", "columns": ["storm_id", "Dst_min", "Bzp"]},
    "speed": {"source_col": "Vsw", "value_col": "Vp", "mode": "max", "columns": ["storm_id", "Dst_min", "Vp"]},
}


def build_parameter_peak_table(filtered_df: pd.DataFrame, omni_df: pd.DataFrame, parameter_key: str) -> pd.DataFrame:
    """Build one per-parameter Data-complete peak table for the Extras section.

    This is intentionally parameter-specific so the Extras tab can calculate only
    the selected table instead of building all long tables on every analysis run.
    """
    cfg = PARAMETER_PEAK_TABLE_CONFIGS.get(parameter_key)
    if cfg is None:
        return pd.DataFrame(columns=["storm_id", "Dst_min"])

    out_columns = cfg["columns"]
    if filtered_df is None or filtered_df.empty or omni_df is None or omni_df.empty:
        return pd.DataFrame(columns=out_columns)

    filtered = filtered_df.copy()
    omni = omni_df.copy()

    if "t_start" not in filtered.columns or "t_min" not in filtered.columns or "t_utc" not in omni.columns:
        return pd.DataFrame(columns=out_columns)

    filtered["t_start"] = pd.to_datetime(filtered["t_start"], utc=True, errors="coerce")
    filtered["t_min"] = pd.to_datetime(filtered["t_min"], utc=True, errors="coerce")
    omni["t_utc"] = pd.to_datetime(omni["t_utc"], utc=True, errors="coerce")

    source_col = cfg["source_col"]
    if source_col not in omni.columns:
        return pd.DataFrame(columns=out_columns)

    for c in [source_col, "Dst"]:
        if c in omni.columns:
            omni[c] = pd.to_numeric(omni[c], errors="coerce")

    out_rows = []
    value_col = cfg["value_col"]
    mode = cfg["mode"]

    for _, s in filtered.iterrows():
        try:
            sid = int(s["storm_id"])
        except Exception:
            continue

        tstart = pd.to_datetime(s.get("t_start"), utc=True, errors="coerce")
        t_min = pd.to_datetime(s.get("t_min"), utc=True, errors="coerce")
        if pd.isna(tstart) or pd.isna(t_min):
            continue

        mp = omni[(omni["t_utc"] >= tstart) & (omni["t_utc"] <= t_min)].copy()
        if mp.empty:
            continue
        mp.sort_values("t_utc", inplace=True)

        ok, values = validate_parameter_mainphase_values(mp, source_col)
        if not ok or values is None or len(values) == 0:
            continue

        if mode == "min":
            value = np.nanmin(values)
        else:
            value = np.nanmax(values)

        if pd.isna(value):
            continue

        Dst_min = float(s["Dst_min"]) if "Dst_min" in s and pd.notna(s["Dst_min"]) else np.nan
        out_rows.append({"storm_id": sid, "Dst_min": Dst_min, value_col: float(value)})

    df = pd.DataFrame(out_rows, columns=out_columns)
    if not df.empty:
        df.sort_values(["storm_id"], ascending=[True], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def sort_parameter_peak_table(df: pd.DataFrame, peak_col: str, order_choice: str) -> pd.DataFrame:
    """Sort an Extras parameter-Data-complete table by storm or by its peak strength."""
    if df is None or df.empty:
        return df

    out = df.copy()

    if order_choice == "peak value" and peak_col in out.columns:
        
        ascending = True if peak_col == "Bzp" else False
        return out.sort_values([peak_col, "storm_id"], ascending=[ascending, True]).reset_index(drop=True)

    if "storm_id" in out.columns:
        return out.sort_values("storm_id", ascending=True).reset_index(drop=True)

    return out.reset_index(drop=True)


def format_parameter_peak_correlation(df: pd.DataFrame, value_col: str, value_label: str = "parameter value") -> str:
    """Return a compact Pearson correlation text for |Dst_min| vs one Extras value column."""
    if df is None or df.empty or "Dst_min" not in df.columns or value_col not in df.columns:
        return f"Correlation |Dst_min| vs {value_label}: R = NA | p = NA | N = 0"

    abs_dst_min = pd.to_numeric(df["Dst_min"], errors="coerce").abs()
    r, p, n = pearson_safe(abs_dst_min, df[value_col])

    if pd.isna(r):
        return f"Correlation |Dst_min| vs {value_label}: R = NA | p = NA | N = {n}"

    return f"Correlation |Dst_min| vs {value_label}: R = {r:.3f} | p = {p:.3g} | N = {n}"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    return out.to_csv(index=False).encode("utf-8")


def zip_outputs(files: Dict[str, pd.DataFrame]) -> bytes:
    import zipfile

    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in files.items():
            zf.writestr(name, to_csv_bytes(df))
    bio.seek(0)
    return bio.read()


@st.cache_data(show_spinner=False, max_entries=8)
def run_storm_analysis_cached(
    raw_text: str,
    shift_plus_1_hour: bool,
    date_start: str,
    date_end: str,
    episode_level: float,
    storm_level: float,
    local_min_radius_hours: int,
    minima_split_factor: float,
    storm_limit: float,
    strong_storm_threshold: float,
    strong_mark: float,
    local_max_radius_hours: int,
    disturbance_filter: bool,
    disturbance_level: float,
    disturb_dip_count: int,
    disturb_dip_radius_hours: int,
) -> Dict[str, object]:
    """Run the expensive storm analysis once per unique input/settings set.

    Sorting controls still trigger a normal Streamlit rerun, but this cached
    function returns the already computed dataframes instead of recalculating
    storm detection, filtering, Stage 3 data-completeness checks, and correlations.
    Extras parameter peak tables are calculated lazily and kept in session state
    after the user selects them for the first time.
    """
    all_rows, bad_rows = read_rows_new8_from_text(raw_text, bool(shift_plus_1_hour))
    if not all_rows:
        raise ValueError(f"No valid data rows were read from the OMNI data file: {OMNI_DATA_FILENAME}")

    analysis_rows = filter_rows_by_date(all_rows, date_start, date_end)
    if not analysis_rows:
        raise ValueError("No rows remain after the date filter.")

    weak_mark_factor_cached = calculate_weak_mark_factor(float(strong_storm_threshold), float(strong_mark))
    if pd.isna(weak_mark_factor_cached):
        raise ValueError("Invalid filtering settings: STRONG_STORM_THRESHOLD must not be 0.")

    episodes_raw = build_episodes_contiguous(analysis_rows, float(episode_level))
    episodes_raw, left_censored_boundary_episodes = apply_left_censored_stage1_boundary_rule(
        analysis_rows,
        episodes_raw,
        float(storm_level),
    )

    storms, episodes, multi_storms = storms_from_episodes(
        analysis_rows,
        episodes_raw,
        float(storm_level),
        int(local_min_radius_hours),
        float(minima_split_factor),
        float(storm_limit),
    )

    omni_df = rows_to_df(analysis_rows)
    storms_df = storms_to_df(storms)
    episodes_df = episodes_to_df(analysis_rows, episodes, storms)
    multi_df = multi_storms_to_df(multi_storms)
    stage1_summary = stage1_outputs_to_summary(
        storms_df,
        episodes_df,
        multi_df,
        int(left_censored_boundary_episodes),
    )

    run_info = {
        "rows_read": int(len(all_rows)),
        "rows_analyzed": int(len(analysis_rows)),
        "bad_rows": int(bad_rows),
        "full_span_start": all_rows[0].t,
        "full_span_end": all_rows[-1].t,
        "analyzed_span_start": analysis_rows[0].t,
        "analyzed_span_end": analysis_rows[-1].t,
        "left_censored_boundary_episodes": int(left_censored_boundary_episodes),
        "left_censored_boundary_note": (
            "! There is a storm event preceding the selected start date (first Dst value is below STORM_LEVEL so whole episode gets rejected)"
        ) if left_censored_boundary_episodes else "",
    }

    filtered_storms, disturbances, excluded, stage2_summary = stage2_filter(
        analysis_rows,
        storms,
        multi_storms,
        float(strong_storm_threshold),
        float(strong_mark),
        int(local_max_radius_hours),
        bool(disturbance_filter),
        float(disturbance_level),
        int(disturb_dip_count),
        int(disturb_dip_radius_hours),
        bool(left_censored_boundary_episodes),
    )

    filtered_df, disturbances_df, excluded_df = stage2_outputs_to_df(
        filtered_storms,
        disturbances,
        excluded,
    )

    data_complete_storms_df, data_complete_rows_df, peak_delays_df, metrics_df, replacements_df, rejected_quality_df, stage3_summary = compute_data_complete_metrics(
        filtered_df, omni_df
    )

    corr_all_df = compute_correlations(metrics_df)

    
    
    stage2_cols_to_remove = ["bz", "imf", "vsw", "ey", "is_multi_storm", "mark", "disturb_dip_count", "disturb_dip_index", "disturb_dip_utc", "disturb_dip_dst", "episode_id", "row_index"]
    filtered_display_df = reorder_stage2_display(
        filtered_df.drop(columns=["reason", *stage2_cols_to_remove], errors="ignore"),
        excluded=False,
    )
    disturbances_display_df = reorder_stage2_display(
        disturbances_df.drop(columns=["reason", *stage2_cols_to_remove], errors="ignore"),
        excluded=False,
    )
    excluded_display_df = reorder_stage2_display(
        excluded_df.drop(columns=["reason", *stage2_cols_to_remove], errors="ignore"),
        excluded=True,
    )

    stage3_all_data_complete_storms_df = build_data_complete_storms_table(data_complete_storms_df, metrics_df)
    stage3_peak_delays_df = build_stage3_peak_delays_table(peak_delays_df)
    stage3_full_peak_data_df = build_stage3_full_peak_data_table(metrics_df)
    stage3_replacements_df = build_stage3_replacements_table(replacements_df, data_complete_storms_df)
    stage3_rejected_quality_df = build_stage3_rejected_quality_table(rejected_quality_df)

    return {
        "run_info": run_info,
        "omni_df": omni_df,
        "storms_df": storms_df,
        "episodes_df": episodes_df,
        "multi_df": multi_df,
        "stage1_summary": stage1_summary,
        "filtered_df": filtered_df,
        "disturbances_df": disturbances_df,
        "excluded_df": excluded_df,
        "stage2_summary": stage2_summary,
        "data_complete_storms_df": data_complete_storms_df,
        "metrics_df": metrics_df,
        "stage3_summary": stage3_summary,
        "corr_all_df": corr_all_df,
        "filtered_display_df": filtered_display_df,
        "disturbances_display_df": disturbances_display_df,
        "excluded_display_df": excluded_display_df,
        "stage3_all_data_complete_storms_df": stage3_all_data_complete_storms_df,
        "stage3_peak_delays_df": stage3_peak_delays_df,
        "stage3_full_peak_data_df": stage3_full_peak_data_df,
        "stage3_replacements_df": stage3_replacements_df,
        "stage3_rejected_quality_df": stage3_rejected_quality_df,
    }



# ---------------------------
# 7) Streamlit rendering functions
# ---------------------------
# This section defines UI helper functions and the output-page renderers.
# The final app layout section below builds the page and calls one renderer.

def _remember_widget_value(widget_key: str, persistent_key: str) -> None:
    st.session_state[persistent_key] = st.session_state[widget_key]


def persistent_radio(label, options, persistent_key: str, default=None, **kwargs):
    options = list(options)
    if not options:
        raise ValueError("persistent_radio requires at least one option")
    if default is None:
        default = options[0]
    if default not in options:
        default = options[0]

    saved_value = st.session_state.get(persistent_key, default)
    if saved_value not in options:
        saved_value = default
        st.session_state[persistent_key] = saved_value

    widget_key = f"_{persistent_key}_widget"
    index = options.index(saved_value)
    return st.radio(
        label,
        options,
        index=index,
        key=widget_key,
        on_change=_remember_widget_value,
        args=(widget_key, persistent_key),
        **kwargs,
    )


def render_table_title(title: str):
    """Render table section titles with larger text."""
    clean_title = str(title).replace("**", "")
    st.markdown(
        f'<div style="font-size:1.25rem; font-weight:700; margin:0.45rem 0 0.20rem 0; line-height:1.25;">{escape(clean_title)}</div>',
        unsafe_allow_html=True,
    )


def render_sort_radio(options, persistent_key: str, default=None):
    """Render sorting controls with the label above the radio options."""
    st.markdown(
        '<div style="margin-bottom:0.15rem; white-space:nowrap; font-weight:400;">Sort by:</div>',
        unsafe_allow_html=True,
    )
    return persistent_radio(
        "",
        options,
        persistent_key=persistent_key,
        default=default if default is not None else options[0],
        label_visibility="collapsed",
        horizontal=True,
    )


def unpack_render_context(ctx: Dict):
    return (
        ctx["analysis_cache_key"],
        ctx["run_info"],
        ctx["omni_df"],
        ctx["storms_df"],
        ctx["episodes_df"],
        ctx["multi_df"],
        ctx["stage1_summary"],
        ctx["filtered_df"],
        ctx["disturbances_df"],
        ctx["excluded_df"],
        ctx["stage2_summary"],
        ctx["data_complete_storms_df"],
        ctx["metrics_df"],
        ctx["stage3_summary"],
        ctx["corr_all_df"],
        ctx["filtered_display_df"],
        ctx["disturbances_display_df"],
        ctx["excluded_display_df"],
        ctx["stage3_all_data_complete_storms_df"],
        ctx["stage3_peak_delays_df"],
        ctx["stage3_full_peak_data_df"],
        ctx["stage3_replacements_df"],
        ctx["stage3_rejected_quality_df"],
    )

# Keep each visible output page in its own render function.
# This separates Overview, stage tables, correlations, downloads, and Extras without changing the calculations.

def render_overview_section(ctx: Dict):
    (
        analysis_cache_key,
        run_info,
        omni_df,
        storms_df,
        episodes_df,
        multi_df,
        stage1_summary,
        filtered_df,
        disturbances_df,
        excluded_df,
        stage2_summary,
        data_complete_storms_df,
        metrics_df,
        stage3_summary,
        corr_all_df,
        filtered_display_df,
        disturbances_display_df,
        excluded_display_df,
        stage3_all_data_complete_storms_df,
        stage3_peak_delays_df,
        stage3_full_peak_data_df,
        stage3_replacements_df,
        stage3_rejected_quality_df,
    ) = unpack_render_context(ctx)

    stage1_class_df = count_by_class(storms_df, "storm count")

    filtered_summary_source = filtered_df.copy()
    if not filtered_summary_source.empty:
        filtered_summary_source["class"] = filtered_summary_source["Dst_min"].apply(class_name)
    filtered_class_df = summarize_by_class(
        filtered_summary_source,
        "filtered storm count",
        [("Mp_duration", "avg Mp duration (h)")],
    )

    data_complete_summary_source = metrics_df.copy()
    data_complete_class_df = summarize_by_class(
        data_complete_summary_source,
        "Data-complete storm count",
        [
            ("Mp_duration", "avg Mp duration (h)"),
            ("delay_Eyp_to_Dst_min_hours", "avg Eyp delay (h)"),
            ("delay_Bzp_to_Dst_min_hours", "avg Bzp delay (h)"),
        ],
    )
    data_complete_corr_df = correlations_selected(data_complete_summary_source)

    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.markdown(
                """
                <h3 style="margin:0 0 0.5rem 0; padding:0;">Results of storm detection</h3>
                """,
                unsafe_allow_html=True,
            )
            summary_left_df = pd.DataFrame(
                [
                    {"": "Total storms", " ": stage1_summary["total_stage1_storms"]},
                    {"": "Storm episodes", " ": stage1_summary["total_episodes"]},
                    {"": "Multistorm episodes", " ": stage1_summary["multi_storm_episodes"]},
                ]
            )
            for _, row in summary_left_df.iterrows():
                st.markdown(f"{row['']}: <span style='font-weight:bold'>&nbsp;&nbsp;{format_val(row[''], row[' '])}</span>", unsafe_allow_html=True)
        with right:
            render_static_scroll_table(stage1_class_df.copy(), key="summary_stage1_class_table", fit_to_container=True, equal_col_widths=True)

    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.markdown(
                """
                <h3 style="margin:0 0 0.5rem 0; padding:0;">Results after filtering</h3>
                """,
                unsafe_allow_html=True,
            )
            previous_storm_exclusions = int(stage2_summary["excluded_previous_storm"])
            filtered_left_df = pd.DataFrame(
                [
                    {"": "Filtered storms", " ": len(filtered_df)},
                    {"": "Excluded as disturbances", " ": len(disturbances_df)},
                    {"": "Excluded due to previous-storm overlap", " ": previous_storm_exclusions},
                    {"": "Average main phase duration (hours)", " ": format_mean_pm_error(stage2_summary["avg_mainphase_filtered"], sem(pd.to_numeric(filtered_df["Mp_duration"], errors="coerce")) if ("Mp_duration" in filtered_df.columns and not filtered_df.empty) else np.nan)},
                ]
            )
            for _, row in filtered_left_df.iterrows():
                st.markdown(f"{row['']}: <span style='font-weight:bold'>&nbsp;&nbsp;{format_val(row[''], row[' '])}</span>", unsafe_allow_html=True)
        with right:
            filtered_class_show_df = filtered_class_df.copy()



            filtered_mp_error_by_class = {}
            if (
                filtered_summary_source is not None
                and not filtered_summary_source.empty
                and "class" in filtered_summary_source.columns
                and "Mp_duration" in filtered_summary_source.columns
            ):
                for class_label, class_group in filtered_summary_source.groupby("class", dropna=False):
                    filtered_mp_error_by_class[str(class_label)] = sem(
                        pd.to_numeric(class_group["Mp_duration"], errors="coerce")
                    )

            if "avg Mp duration (h)" in filtered_class_show_df.columns:
                filtered_class_show_df["avg Mp duration (h)"] = filtered_class_show_df.apply(
                    lambda r: format_mean_pm_error(
                        r.get("avg Mp duration (h)", np.nan),
                        filtered_mp_error_by_class.get(str(r.get("class", "")), np.nan),
                    ),
                    axis=1,
                )


            if "class" in filtered_class_show_df.columns:
                filtered_class_show_df["class"] = filtered_class_show_df["class"].replace({
                    "moderate (-100 < Dst_min <= -50)": "moderate",
                    "intense (-250 < Dst_min <= -100)": "intense",
                    "super-storm (Dst_min <= -250)": "super-storm",
                })
            render_static_scroll_table(filtered_class_show_df, key="summary_filtered_class_table", fit_to_container=True, equal_col_widths=True)

    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:8px; margin:0 0 0.5rem 0;">
                    <h3 style="margin:0; padding:0;">Results of Data-complete storms</h3>
                    <button
                        type="button"
                        title="Data-complete storms are filtered storms for which the solar wind parameters do not contain consecutive missing OMNIWeb values during the main phase. Filtered storms with isolated missing values are also retained, with those values replaced using single-point linear interpolation."
                        style="
                            width:24px; height:24px; border-radius:0; border:1px solid #000000;
                            background:#F8E463; color:#111827; font-weight:700; cursor:help;
                            line-height:22px; text-align:center; padding:0;
                        "
                        aria-label="Data-complete storms hint"
                    >?</button>
                </div>
                """,
                unsafe_allow_html=True,
            )


            data_complete_counts = {
                "moderate": 0,
                "intense": 0,
                "super-storm": 0,
            }
            if data_complete_class_df is not None and not data_complete_class_df.empty:
                count_col = "Data-complete storm count"
                for _, class_row in data_complete_class_df.iterrows():
                    class_label = str(class_row.get("class", "")).lower()
                    try:
                        class_count = int(class_row.get(count_col, 0))
                    except Exception:
                        class_count = 0
                    if class_label.startswith("moderate"):
                        data_complete_counts["moderate"] = class_count
                    elif class_label.startswith("intense"):
                        data_complete_counts["intense"] = class_count
                    elif class_label.startswith("super-storm"):
                        data_complete_counts["super-storm"] = class_count

            data_complete_left_rows = [
                {"label": "Data-complete storms", "value": len(data_complete_storms_df), "kind": "normal"},
                {"label": "class_counts", "value": data_complete_counts, "kind": "class_counts"},
                {"label": "Average min Dst delay from Eyp (hours)", "value": format_mean_pm_error(stage3_summary.get("avg_delay_Eyp", np.nan), stage3_summary.get("err_delay_Eyp", np.nan)), "kind": "normal"},
                {"label": "Average min Dst delay from Bzp (hours)", "value": format_mean_pm_error(stage3_summary.get("avg_delay_Bzp", np.nan), stage3_summary.get("err_delay_Bzp", np.nan)), "kind": "normal"},
            ]
            for row in data_complete_left_rows:
                if row["kind"] == "class_counts":
                    counts = row["value"]
                    st.markdown(
                        "Moderate:&nbsp;&nbsp;&nbsp;<span style='font-weight:bold'>{}</span>,&nbsp;&nbsp;&nbsp;"
                        "Intense:&nbsp;&nbsp;&nbsp;<span style='font-weight:bold'>{}</span>,&nbsp;&nbsp;&nbsp;"
                        "Super-storm:&nbsp;&nbsp;&nbsp;<span style='font-weight:bold'>{}</span>".format(
                            format_val("storms", counts["moderate"]),
                            format_val("storms", counts["intense"]),
                            format_val("storms", counts["super-storm"]),
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"{row['label']}: <span style='font-weight:bold'>&nbsp;&nbsp;{format_val(row['label'], row['value'])}</span>", unsafe_allow_html=True)
        with right:
            data_complete_corr_show_df = data_complete_corr_df.copy()
            if "R" in data_complete_corr_show_df.columns:
                data_complete_corr_show_df["R"] = pd.to_numeric(data_complete_corr_show_df["R"], errors="coerce").map(
                    lambda x: "—" if pd.isna(x) else f"{float(x):.3f}"
                )
            render_static_scroll_table(data_complete_corr_show_df, key="summary_Data-complete_correlations_table", fit_to_container=True, equal_col_widths=True)


def render_storm_detection_section(ctx: Dict):
    (
        analysis_cache_key,
        run_info,
        omni_df,
        storms_df,
        episodes_df,
        multi_df,
        stage1_summary,
        filtered_df,
        disturbances_df,
        excluded_df,
        stage2_summary,
        data_complete_storms_df,
        metrics_df,
        stage3_summary,
        corr_all_df,
        filtered_display_df,
        disturbances_display_df,
        excluded_display_df,
        stage3_all_data_complete_storms_df,
        stage3_peak_delays_df,
        stage3_full_peak_data_df,
        stage3_replacements_df,
        stage3_rejected_quality_df,
    ) = unpack_render_context(ctx)

    render_section_heading("Storm detection", ["Dst: nT"])
    render_table_title("Total storms")

    stage1_storm_minima_order = render_sort_radio(
        ["Date", "Dst_min"],
        persistent_key="stage1_storm_minima_order",
        default="Date",
    )

    stage1_storm_minima_df = sort_stage1_storm_minima(storms_df, stage1_storm_minima_order)
    render_table("Storm minima", stage1_storm_minima_df)

    if int(run_info.get("left_censored_boundary_episodes", 0)) > 0:
        st.warning(run_info.get("left_censored_boundary_note", "! There is a storm event preceeding the selected date start"))


    render_table_title("Episodes")

    stage1_episodes_order = render_sort_radio(
        ["Date", "duration", "global_min"],
        persistent_key="stage1_episodes_order",
        default="Date",
    )

    episodes_sorted_df = sort_stage1_episodes(episodes_df, stage1_episodes_order)
    render_table("Episodes", episodes_sorted_df)
    render_table_title("Multi-storm episodes")

    stage1_multi_order = render_sort_radio(
        ["Date", "duration", "global_min"],
        persistent_key="stage1_multi_order",
        default="Date",
    )

    multi_sorted_df = sort_stage1_multi_storm_minima(multi_df, stage1_multi_order)
    render_table("Multi-storm episode minima", multi_sorted_df)


def render_main_phase_filtering_section(ctx: Dict):
    (
        analysis_cache_key,
        run_info,
        omni_df,
        storms_df,
        episodes_df,
        multi_df,
        stage1_summary,
        filtered_df,
        disturbances_df,
        excluded_df,
        stage2_summary,
        data_complete_storms_df,
        metrics_df,
        stage3_summary,
        corr_all_df,
        filtered_display_df,
        disturbances_display_df,
        excluded_display_df,
        stage3_all_data_complete_storms_df,
        stage3_peak_delays_df,
        stage3_full_peak_data_df,
        stage3_replacements_df,
        stage3_rejected_quality_df,
    ) = unpack_render_context(ctx)

    render_section_heading("Main phase filtering", ["Dst: nT", "Mp_duration: hours"])
    render_table_title("Filtered storms")

    stage2_filtered_order = render_sort_radio(
        ["Date", "Dst_min", "Mp_duration"],
        persistent_key="stage2_filtered_order",
        default="Date",
    )

    filtered_sorted_df = sort_stage2_table(filtered_display_df, stage2_filtered_order)
    render_table("Filtered storms", filtered_sorted_df)
    if int(stage2_summary.get("first_storm_no_tstart_boundary", 0)) > 0:
        st.warning("! No reliable t_start could be calculated for the first storm due to the selected start date.")

    render_table_title("Excluded as disturbances")

    stage2_disturbances_order = render_sort_radio(
        ["Date", "Dst_min", "Mp_duration"],
        persistent_key="stage2_disturbances_order",
        default="Date",
    )

    disturbances_sorted_df = sort_stage2_table(disturbances_display_df, stage2_disturbances_order)
    render_table("Disturbances", disturbances_sorted_df)

    render_table_title("Excluded due to previous-storm overlap")
    render_table("Excluded", sort_by_time_generic(excluded_display_df))


def render_data_complete_section(ctx: Dict):
    (
        analysis_cache_key,
        run_info,
        omni_df,
        storms_df,
        episodes_df,
        multi_df,
        stage1_summary,
        filtered_df,
        disturbances_df,
        excluded_df,
        stage2_summary,
        data_complete_storms_df,
        metrics_df,
        stage3_summary,
        corr_all_df,
        filtered_display_df,
        disturbances_display_df,
        excluded_display_df,
        stage3_all_data_complete_storms_df,
        stage3_peak_delays_df,
        stage3_full_peak_data_df,
        stage3_replacements_df,
        stage3_rejected_quality_df,
    ) = unpack_render_context(ctx)

    render_section_heading("Data-complete storms and main phase metrics", ["Dst, IMF, Bz: nT", "Mp_duration: hours", "V: km/s", "Ey: mV/m"])

    render_table_title("Data-complete storms")

    stage3_all_data_complete_order = render_sort_radio(
        ["Date", "Dst_min", "Mp_duration", "IMFp", "Bzp", "Vp", "Eyp", "Eyi"],
        persistent_key="stage3_all_data_complete_order",
        default="Date",
    )

    stage3_all_data_complete_sorted_df = sort_data_complete_storms(stage3_all_data_complete_storms_df, stage3_all_data_complete_order)
    render_data_complete_storms_table("All Data-complete storms", stage3_all_data_complete_sorted_df)

    render_table_title("Peak delays")
    render_table("Peak delays", sort_by_time_generic(stage3_peak_delays_df))

    render_table_title("Full peak timeseries")
    render_table("Full peak data", sort_by_time_generic(stage3_full_peak_data_df))

    render_table_title("Replaced single missing values")
    render_table("Replaced single missing values", sort_by_time_generic(stage3_replacements_df))

    render_table_title("Rejected storms (multiple missing values)")
    rejected_storms_show_df = sort_by_time_generic(stage3_rejected_quality_df).copy()
    if not rejected_storms_show_df.empty:
        for col in rejected_storms_show_df.columns:
            col_l = str(col).lower()
            if pd.api.types.is_datetime64_any_dtype(rejected_storms_show_df[col]):
                rejected_storms_show_df[col] = pd.to_datetime(rejected_storms_show_df[col], utc=True, errors="coerce").map(fmt_dt)
            else:
                rejected_storms_show_df[col] = rejected_storms_show_df[col].map(pretty_value)
    render_static_scroll_table(
        rejected_storms_show_df,
        max_height=260,
        key="rejected_storms_multiple_missing_values_table",
        fit_to_container=True,
        equal_col_widths=False,
    )



def render_correlations_section(ctx: Dict):
    (
        analysis_cache_key,
        run_info,
        omni_df,
        storms_df,
        episodes_df,
        multi_df,
        stage1_summary,
        filtered_df,
        disturbances_df,
        excluded_df,
        stage2_summary,
        data_complete_storms_df,
        metrics_df,
        stage3_summary,
        corr_all_df,
        filtered_display_df,
        disturbances_display_df,
        excluded_display_df,
        stage3_all_data_complete_storms_df,
        stage3_peak_delays_df,
        stage3_full_peak_data_df,
        stage3_replacements_df,
        stage3_rejected_quality_df,
    ) = unpack_render_context(ctx)

    st.subheader("Pearson correlations vs |Dst_min|")
    render_table_title("Correlations")
    render_static_scroll_table(format_corr_df(corr_all_df), key="correlations_table", fit_to_container=True)
    render_table_title("Plots")
    render_correlation_plots(metrics_df)


def render_downloads_section(ctx: Dict):
    (
        analysis_cache_key,
        run_info,
        omni_df,
        storms_df,
        episodes_df,
        multi_df,
        stage1_summary,
        filtered_df,
        disturbances_df,
        excluded_df,
        stage2_summary,
        data_complete_storms_df,
        metrics_df,
        stage3_summary,
        corr_all_df,
        filtered_display_df,
        disturbances_display_df,
        excluded_display_df,
        stage3_all_data_complete_storms_df,
        stage3_peak_delays_df,
        stage3_full_peak_data_df,
        stage3_replacements_df,
        stage3_rejected_quality_df,
    ) = unpack_render_context(ctx)

    st.subheader("Download outputs")




    stage1_storm_minima_download_df = sort_stage1_storm_minima(storms_df, "time")

    time_col = None
    for candidate in ["start_utc", "start", "episode_start", "start_time"]:
        if candidate in episodes_df.columns:
            time_col = candidate
            break
    episodes_download_df = episodes_df.sort_values(by=time_col, ascending=True) if time_col else episodes_df

    multi_download_df = sort_stage1_multi_storm_minima(multi_df, "time")
    filtered_download_df = sort_stage2_table(filtered_display_df, "time")
    disturbances_download_df = sort_stage2_table(disturbances_display_df, "time")
    stage3_all_data_complete_download_df = sort_data_complete_storms(stage3_all_data_complete_storms_df, "time")

    stage1_downloads = [
        ("Total storms", "Total storms.csv", stage1_storm_minima_download_df),
        ("Episodes", "Episodes.csv", episodes_download_df),
        ("Multi-storm episodes", "Multi-storm episodes.csv", multi_download_df),
    ]

    stage2_downloads = [
        ("Filtered storms", "Filtered storms.csv", filtered_download_df),
        ("Excluded as disturbances", "Excluded as disturbances.csv", disturbances_download_df),
        ("Excluded due to previous-storm overlap", "Excluded due to previous-storm overlap.csv", sort_by_time_generic(excluded_display_df)),
    ]

    stage3_downloads = [
        ("Data-complete storms", "Data-complete storms.csv", stage3_all_data_complete_download_df),
        ("Peak delays", "Peak delays.csv", sort_by_time_generic(stage3_peak_delays_df)),
        ("Full peak timeseries", "Full peak timeseries.csv", sort_by_time_generic(stage3_full_peak_data_df)),
        ("Replaced single missing values", "Replaced single missing values.csv", sort_by_time_generic(stage3_replacements_df)),
        ("Rejected storms (multiple missing values)", "Rejected storms (multiple missing values).csv", sort_by_time_generic(stage3_rejected_quality_df)),
    ]

    correlation_downloads = [
        ("Correlations", "Correlations.csv", corr_all_df),
    ]

    ordered_output_files = stage1_downloads + stage2_downloads + stage3_downloads + correlation_downloads

    output_files = {filename: df for _title, filename, df in ordered_output_files}

    zip_bytes = zip_outputs(output_files)
    st.download_button(
        "Download all outputs as ZIP",
        data=zip_bytes,
        file_name="magnetic_storm_outputs.zip",
        mime="application/zip",
        use_container_width=True,
        on_click="ignore",
    )

    for title, filename, df in ordered_output_files:
        st.download_button(
            f"Download {title}",
            data=to_csv_bytes(df),
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}",
            on_click="ignore",
        )



def render_extras_section(ctx: Dict):
    (
        analysis_cache_key,
        run_info,
        omni_df,
        storms_df,
        episodes_df,
        multi_df,
        stage1_summary,
        filtered_df,
        disturbances_df,
        excluded_df,
        stage2_summary,
        data_complete_storms_df,
        metrics_df,
        stage3_summary,
        corr_all_df,
        filtered_display_df,
        disturbances_display_df,
        excluded_display_df,
        stage3_all_data_complete_storms_df,
        stage3_peak_delays_df,
        stage3_full_peak_data_df,
        stage3_replacements_df,
        stage3_rejected_quality_df,
    ) = unpack_render_context(ctx)

    st.subheader("Extras")
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin:0.45rem 0 0.20rem 0; line-height:1.25;">
            <div style="font-size:1.25rem; font-weight:700; margin:0; padding:0;">Parameter-specific data-complete peak tables</div>
            <button
                type="button"
                title="Tables show peak values of solar wind parameters during the main phase separately. A storm may be included in the analysis for one parameter, such as Bz, even if it contains missing values in another parameter, such as V or Ey. As a result, the sample size may differ between parameters and is generally larger than the fully data-complete sample."
                style="
                    width:24px; height:24px; border-radius:0; border:1px solid #000000;
                    background:#F8E463; color:#111827; font-weight:700; cursor:help;
                    line-height:22px; text-align:center; padding:0;
                "
                aria-label="Parameter-Data-complete peak tables hint"
            >?</button>
        </div>
        """,
        unsafe_allow_html=True,
    )

    extras_table_options = {
        "IMF": {
            "key": "imf",
            "title": "**Data-complete for IMF only**",
            "table_name": "Extras IMF values",
            "value_col": "IMFp",
            "correlation_label": "IMFp",
            "sort_key": "extras_IMFp_order",
        },
        "Bz": {
            "key": "bz",
            "title": "**Data-complete for Bz only**",
            "table_name": "Extras Bz values",
            "value_col": "Bzp",
            "correlation_label": "Bzp",
            "sort_key": "extras_Bzp_order",
        },
        "V": {
            "key": "speed",
            "title": "**Data-complete for V only**",
            "table_name": "Extras V values",
            "value_col": "Vp",
            "correlation_label": "Vp",
            "sort_key": "extras_Vp_order",
        },
    }



    extras_parameter_selector_width_px = 430



    if "extras_selected_parameter" not in st.session_state:
        st.session_state.extras_selected_parameter = list(extras_table_options.keys())[0]
    old_parameter_label_map = {
        "IMFp values": "IMF",
        "Bzp values": "Bz",
        "Vp values": "V",
        "Eyp values": "Bz",
        "Eyi values": "Bz",
        "IMFp": "IMF",
        "Bzp": "Bz",
        "Eyp": "Bz",
        "Eyi": "Bz",
        "Vp": "V",
    }
    st.session_state.extras_selected_parameter = old_parameter_label_map.get(
        st.session_state.extras_selected_parameter,
        st.session_state.extras_selected_parameter,
    )
    if st.session_state.extras_selected_parameter not in extras_table_options:
        st.session_state.extras_selected_parameter = list(extras_table_options.keys())[0]

    selected_parameter_label = st.session_state.extras_selected_parameter
    selected_parameter_label_css = selected_parameter_label.replace('\\', '\\\\').replace('"', '\\"')

    st.markdown(
        f"""
        <style>
        div[data-testid="stPopover"] {{
            width: {extras_parameter_selector_width_px}px !important;
            max-width: 100% !important;
            display: block !important;
        }}

        div[data-testid="stPopover"] > button {{
            width: 100% !important;
            max-width: 100% !important;
            justify-content: space-between !important;
            cursor: pointer !important;
            caret-color: transparent !important;
            user-select: none !important;
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111111 !important;
            border: 1px solid rgba(0,0,0,0.22) !important;
            border-width: 1px !important;
            border-radius: 9px !important;
            padding: 0.45rem 0.75rem !important;
            outline: none !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.10) !important;
            font-size: 0 !important;
            line-height: 1.1 !important;
        }}

        div[data-testid="stPopover"] > button::before {{
            content: "{selected_parameter_label_css}" !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            color: #111111 !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            line-height: 1.1 !important;
            opacity: 1 !important;
            filter: none !important;
        }}

        div[data-testid="stPopover"] > button::after {{
            content: "▾" !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            color: #111111 !important;
            font-size: 0.85rem !important;
            font-weight: 700 !important;
            line-height: 1 !important;
            margin-left: 0.5rem !important;
        }}

        div[data-testid="stPopover"] > button:hover,
        div[data-testid="stPopover"] > button:focus,
        div[data-testid="stPopover"] > button:active,
        div[data-testid="stPopover"] > button[aria-expanded="true"] {{
            cursor: pointer !important;
            caret-color: transparent !important;
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111111 !important;
            border-color: rgba(0,0,0,0.22) !important;
            border-width: 1px !important;
            outline: none !important;
            opacity: 1 !important;
            filter: none !important;
        }}

        div[data-testid="stPopover"] > button > * {{
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            opacity: 0 !important;
        }}

        /* Extras parameter menu: keep the opened choices the same width and style as the selected-value button. */
        [data-testid="stPopoverBody"],
        [data-baseweb="popover"] {{
            width: {extras_parameter_selector_width_px}px !important;
            max-width: {extras_parameter_selector_width_px}px !important;
            box-sizing: border-box !important;
        }}

        [data-testid="stPopoverBody"] div[data-testid="stButton"],
        [data-baseweb="popover"] div[data-testid="stButton"] {{
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            margin: 0 !important;
        }}

        [data-testid="stPopoverBody"] div[data-testid="stButton"] > button,
        [data-baseweb="popover"] div[data-testid="stButton"] > button {{
            width: 100% !important;
            max-width: 100% !important;
            min-height: 38px !important;
            justify-content: flex-start !important;
            cursor: pointer !important;
            caret-color: transparent !important;
            user-select: none !important;
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111111 !important;
            border: 1px solid rgba(0,0,0,0.22) !important;
            border-width: 1px !important;
            border-radius: 9px !important;
            padding: 0.45rem 0.75rem !important;
            outline: none !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.10) !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            line-height: 1.1 !important;
            box-sizing: border-box !important;
        }}

        [data-testid="stPopoverBody"] div[data-testid="stButton"] > button p,
        [data-baseweb="popover"] div[data-testid="stButton"] > button p {{
            color: #111111 !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            line-height: 1.1 !important;
            margin: 0 !important;
        }}

        [data-testid="stPopoverBody"] div[data-testid="stButton"] > button:hover,
        [data-testid="stPopoverBody"] div[data-testid="stButton"] > button:focus,
        [data-testid="stPopoverBody"] div[data-testid="stButton"] > button:active,
        [data-baseweb="popover"] div[data-testid="stButton"] > button:hover,
        [data-baseweb="popover"] div[data-testid="stButton"] > button:focus,
        [data-baseweb="popover"] div[data-testid="stButton"] > button:active {{
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111111 !important;
            border-color: rgba(0,0,0,0.22) !important;
            border-width: 1px !important;
            outline: none !important;
            cursor: pointer !important;
            caret-color: transparent !important;
        }}

        /* No permanent selected/focus look on the selected parameter button or option buttons. */
        div[data-testid="stPopover"] > button:focus-visible,
        div[data-testid="stPopover"] > button:focus:not(:focus-visible),
        [data-testid="stPopoverBody"] div[data-testid="stButton"] > button:focus-visible,
        [data-testid="stPopoverBody"] div[data-testid="stButton"] > button:focus:not(:focus-visible),
        [data-baseweb="popover"] div[data-testid="stButton"] > button:focus-visible,
        [data-baseweb="popover"] div[data-testid="stButton"] > button:focus:not(:focus-visible) {{
            border: 1px solid rgba(0,0,0,0.22) !important;
            outline: none !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.10) !important;
        }}
        div[data-testid="stPopover"],
        div[data-testid="stPopover"] *,
        [data-testid="stPopoverBody"],
        [data-testid="stPopoverBody"] *,
        [data-baseweb="popover"],
        [data-baseweb="popover"] * {{
            cursor: pointer !important;
            caret-color: transparent !important;
            user-select: none !important;
        }}

        .extras-parameter-summary {{
            color: #111827 !important;
            font-size: 1rem !important;
            font-weight: 400 !important;
            line-height: 1.45 !important;
            margin-top: 0.35rem !important;
            margin-bottom: 0.35rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Choose parameter for parameter-specific Data-completeness**")
    with st.popover(selected_parameter_label):
        for option_label in extras_table_options.keys():
            if st.button(option_label, key=f"extras_parameter_choice_{extras_table_options[option_label]['key']}", use_container_width=True):
                st.session_state.extras_selected_parameter = option_label
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass

    selected_parameter_label = st.session_state.extras_selected_parameter
    selected_option = extras_table_options[selected_parameter_label]

    if "extras_peak_table_cache" not in st.session_state:
        st.session_state.extras_peak_table_cache = {}

    def get_extras_peak_table_once(parameter_key: str, table_name: str) -> pd.DataFrame:
        cache_key = (analysis_cache_key, parameter_key)
        cache = st.session_state.extras_peak_table_cache
        if cache_key not in cache:
            with st.spinner(f"Calculating {table_name}..."):
                cache[cache_key] = build_parameter_peak_table(filtered_df, omni_df, parameter_key)
        return cache[cache_key]

    def render_parameter_peak_table(title: str, table_name: str, parameter_key: str, value_col: str, sort_key: str, correlation_label: str):
        table_df = get_extras_peak_table_once(parameter_key, table_name)
        render_table_title(title)

        order_choice = render_sort_radio(
            ["storm_id", "peak value"],
            persistent_key=sort_key,
            default="storm_id",
        )

        sorted_df = sort_parameter_peak_table(table_df, value_col, order_choice)
        one_decimal_columns = {value_col} if str(value_col).lower() in {"IMFp", "Bzp"} else set()
        render_table(table_name, sorted_df, one_decimal_columns=one_decimal_columns)
        corr_text = format_parameter_peak_correlation(table_df, value_col, correlation_label)
        st.markdown(
            "<div class='extras-parameter-summary'>"
            f"{escape(str(corr_text))}"
            "</div>",
            unsafe_allow_html=True,
        )

    render_parameter_peak_table(
        selected_option["title"],
        selected_option["table_name"],
        selected_option["key"],
        selected_option["value_col"],
        selected_option["sort_key"],
        selected_option["correlation_label"],
    )

# ---------------------------
# 8) Streamlit app layout and display
# ---------------------------
# This section builds the visible Streamlit page and calls the selected renderer.

def set_bg_image_auto():
    background_data_url = load_background_image_data_url()
    if background_data_url is None:
        return

    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("{background_data_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

st.markdown("""
<style>
/* Remove Streamlit header anchor/action links everywhere */
[data-testid="stHeaderActionElements"],
.stHeadingActionElements,
.stHeaderActionElements,
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
    visibility: hidden !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Keep main content above the background normally */
.block-container {
    position: relative;
    z-index: 1;
}

/* White metric cards */
div[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 10px !important;
    padding: 8px !important;
}



/* White expanders */
details {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 10px !important;
    padding: 6px !important;
}


/* White content panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.10) !important;
    border-radius: 12px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: white !important;
    border-radius: 12px !important;
}

/* White chart wrappers */
div[data-testid="stPyplot"] {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 10px !important;
    padding: 6px !important;
}


/* FORCE title white + keep the app title to exactly two lines on smaller screens. */
h1 {
    color: white !important;
    font-weight: 800 !important;
}

.app-main-title {
    color: white !important;
    font-weight: 800 !important;
    line-height: 1.08 !important;
    margin-top: -34px !important;
    margin-bottom: 0.35rem !important;
    font-size: clamp(1.55rem, 2.65vw, 2.75rem) !important;
}

.app-main-title .title-line {
    white-space: nowrap !important;
}


/* START button: thin white border */
div[data-testid="stButton"] > button[kind="primary"] {
    border: 2px solid white !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:focus,
div[data-testid="stButton"] > button[kind="primary"]:active {
    border: 2px solid white !important;
}


/* Compact header controls for smaller laptop screens. */
div[data-testid="stButton"] > button[kind="primary"] {
    white-space: nowrap !important;
    min-width: 86px !important;
    padding-left: 0.85rem !important;
    padding-right: 0.85rem !important;
}

div[data-testid="stButton"] > button[kind="primary"] p {
    white-space: nowrap !important;
}


@media (max-width: 1250px) {
    .app-main-title {
        font-size: clamp(1.25rem, 2.45vw, 2.05rem) !important;
        line-height: 1.05 !important;
    }
}


</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Main section selector */
.st-key-active_section div[data-testid="stRadio"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.st-key-active_section div[data-testid="stRadio"] > label {
    display: none !important;
}

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: transparent !important;
    border: 2px solid transparent !important;
    border-radius: 10px !important;
    padding: 6px 10px !important;
    margin: 0 !important;
    color: white !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    -webkit-text-stroke: 0px black !important;
    text-stroke: 0px black !important;
}

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    border: 2px solid rgba(255,255,255,0.5) !important;
}

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    border: 3px solid white !important;
    color: white !important;
}

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label * {
    color: white !important;
    font-weight: 800 !important;
    font-size: 16px !important;
}

/* Visible text size for the main section options (Overview, Extras, etc.). */
.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
    font-size: 22px !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    margin-bottom: 0 !important;
}

/* Hide the radio dots only for the main section selector. */

.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Active section panel */
.st-key-active_section_panel,
.st-key-active_section_panel > div,
.st-key-active_section_panel [data-testid="stVerticalBlock"],
.st-key-active_section_panel [data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff !important;
    color: #111827 !important;
    border-radius: 12px !important;
}

.st-key-active_section_panel {
    border: 1px solid rgba(0,0,0,0.14) !important;
    box-shadow: none !important;
    padding: 8px 16px 16px 16px !important;
    margin-top: 0px !important;
}

.st-key-active_section + div {
    margin-top: 0px !important;
    padding-top: 0px !important;
}

.st-key-active_section_panel > div {
    padding-top: 0px !important;
    margin-top: 0px !important;
}

/* Broad panel styling for heading containers. */
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h1),
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h2),
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h3) {
    background-color: #ffffff !important;
    border: 1px solid rgba(0,0,0,0.14) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h1) > div,
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h2) > div,
div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMarkdownContainer"] h3) > div {
    background-color: #ffffff !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Keep ONLY table sorting radio options on one line on smaller laptop screens.
   The main section selector (Overview / Storm detection / etc.) keeps its original wrapping behavior. */
div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] div[role="radiogroup"] {
    flex-wrap: nowrap !important;
    gap: 0.45rem !important;
    align-items: center !important;
}

div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] div[role="radiogroup"] label {
    white-space: nowrap !important;
    min-width: max-content !important;
    padding-right: 0.40rem !important;
}

div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"],
div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
    white-space: nowrap !important;
    overflow-wrap: normal !important;
    word-break: normal !important;
}

/* Sorting rows are intentionally compact; this prevents wrapping inside the sorting option column only. */
div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] {
    min-width: max-content !important;
}

/* Restore section selector wrapping explicitly, even when other radio styling is active. */
.st-key-active_section div[data-testid="stRadio"] div[role="radiogroup"] {
    flex-wrap: wrap !important;
}

/* Table sorting radios: make ONLY the unselected circle outline a bit thicker.
   Keeps the circle size and selected red state unchanged. */
div[class*="st-key-_"][class*="_order_widget"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:not(:checked)) > div:first-child div {
    box-shadow: inset 0 0 0 1px currentColor !important;
}
</style>
""", unsafe_allow_html=True)


set_bg_image_auto()

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 10px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if pearsonr is None:
    st.error(
        "SciPy is not available in this Python environment, so Pearson p-values cannot be computed the same way as in your Colab file. Install SciPy in this exact environment, then rerun the app."
    )
    st.code('python -m pip install scipy', language='bash')
    if SCIPY_IMPORT_ERROR:
        st.caption(f"SciPy import error: {SCIPY_IMPORT_ERROR}")
    st.stop()


ANALYSIS_SETTINGS_LABEL_SIZE_PX = 20
ANALYSIS_SETTINGS_LABEL_FONT_WEIGHT = 800


ALGORITHM_PARAMETERS_ANALYSIS_TOP_SPACER = "4rem"


ANALYSIS_OPTIONS_TITLE_SIZE_PX = 19


ANALYSIS_OPTIONS_TITLE_BULLET = "•"


ANALYSIS_OPTIONS_BACKGROUND_COLOR = "#D9FFB0"


ANALYSIS_OPTIONS_WIDTH_RATIO = 1.05


ANALYSIS_TO_CONTENT_GAP_RATIO = 0.08


RIGHT_SCREEN_GAP_RATIO = 0.25


MAIN_CONTENT_WIDTH_RATIO = 5.35


def render_section_heading(title: str, unit_items: List[str]):
    """Render a section heading with an inline unit legend beside the title."""
    units_html = escape(" | ".join(str(item) for item in unit_items))
    title_html = escape(title)
    st.markdown(
        '<div class="section-heading-row">'
        f'<h3 class="section-heading-title">{title_html}</h3>'
        f'<span class="section-heading-units">{units_html}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def analysis_options_title(title: str):
    st.markdown(
        f"""
        <div class='analysis-options-title'>
            <span class='analysis-options-title-bullet'>{escape(ANALYSIS_OPTIONS_TITLE_BULLET)}</span>
            <span>{escape(title)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_algorithm_parameters_pdf_button():
    pdf_path, pdf_bytes = load_algorithm_parameters_pdf()

    if pdf_path is None or pdf_bytes is None:
        st.caption("Algorithm parameters PDF not found. Put parameters.pdf in the same folder as this app file.")
        return

    st.markdown(
        """
        <style>
        .st-key-algorithm_parameters_download {
            margin: 0 !important;
            padding: 0 !important;
        }

        .st-key-algorithm_parameters_download div[data-testid="stDownloadButton"] {
            margin: 0 !important;
            padding: 0 !important;
        }

        .st-key-algorithm_parameters_download div[data-testid="stDownloadButton"] > button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.35rem !important;
            width: auto !important;
            min-width: 0 !important;
            padding: 0.45rem 0.75rem !important;
            margin: 0 !important;
            border-radius: 9px !important;
            border: 1px solid rgba(0,0,0,0.22) !important;
            background: rgba(255,255,255,0.92) !important;
            color: #111111 !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.10) !important;
        }

        .st-key-algorithm_parameters_download div[data-testid="stDownloadButton"] > button:hover,
        .st-key-algorithm_parameters_download div[data-testid="stDownloadButton"] > button:focus,
        .st-key-algorithm_parameters_download div[data-testid="stDownloadButton"] > button:active {
            background: #ffffff !important;
            border-color: rgba(0,0,0,0.35) !important;
            color: #000000 !important;
        }

        .st-key-algorithm_parameters_download div[data-testid="stDownloadButton"] > button p {
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            line-height: 1.1 !important;
            color: #111111 !important;
            margin: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "Algorithm parameters",
        data=pdf_bytes,
        file_name=pdf_path.name,
        mime="application/pdf",
        key="algorithm_parameters_download",
        use_container_width=False,
    )


st.markdown(f"""
<style>
/* Keep the options panel flush to the left and let the right column hold all output. */
.block-container {{
    max-width: 100% !important;
    padding-top: 0rem !important;
    padding-left: 0rem !important;
    padding-right: 1rem !important;
    margin-top: -1.8rem !important;
}}

/* Remove the tiny inherited offset before the first layout column. */
div[data-testid="stHorizontalBlock"] > div:first-child {{
    padding-left: 0rem !important;
}}

/* Analysis options dropdown background. */
div[data-testid="stHorizontalBlock"] > div:first-child details {{
    background: {ANALYSIS_OPTIONS_BACKGROUND_COLOR} !important;
    border: 1px solid rgba(0,0,0,0.16) !important;
    border-radius: 10px !important;
}}

div[data-testid="stHorizontalBlock"] > div:first-child details summary {{
    background: {ANALYSIS_OPTIONS_BACKGROUND_COLOR} !important;
    border-radius: 8px !important;
}}

/* Only the main "Analysis settings" expander label. */
div[data-testid="stHorizontalBlock"] > div:first-child details summary p {{
    font-size: {ANALYSIS_SETTINGS_LABEL_SIZE_PX}px !important;
    font-weight: {ANALYSIS_SETTINGS_LABEL_FONT_WEIGHT} !important;
    line-height: 1.2 !important;
}}

div[data-testid="stHorizontalBlock"] > div:first-child details [data-testid="stExpanderDetails"] {{
    background: {ANALYSIS_OPTIONS_BACKGROUND_COLOR} !important;
}}

.analysis-options-title {{
    display: flex !important;
    align-items: center !important;
    gap: 0.45rem !important;
    font-size: {ANALYSIS_OPTIONS_TITLE_SIZE_PX}px !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    margin: 1.0rem 0 0.45rem 0 !important;
}}

.analysis-options-title-bullet {{
    font-size: calc({ANALYSIS_OPTIONS_TITLE_SIZE_PX}px + 1px) !important;
    line-height: 1 !important;
    display: inline-block !important;
}}

/* Section headings: title on the left, unit badge on the far right, without adding extra top-page layout space. */
.section-heading-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    flex-wrap: nowrap;
    width: 100%;
    margin: 0 0 0.8rem 0;
    padding: 0;
}}

.section-heading-title {{
    flex: 1 1 auto;
    min-width: 0;
    margin: 0 !important;
    padding: 0 !important;
    color: #111827 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
}}

.section-heading-units {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 auto;
    margin: 0 0 0 auto !important;
    padding: 0.28rem 0.62rem;
    border: 1px solid #111111;
    border-radius: 8px;
    background: #ffffff;
    color: #111111;
    font-size: 0.92rem;
    font-weight: 600;
    line-height: 1.2;
    white-space: nowrap;
}}

</style>
""", unsafe_allow_html=True)


main_options_col, _analysis_gap_col, main_content_col, _right_gap_col = st.columns(
    [ANALYSIS_OPTIONS_WIDTH_RATIO, ANALYSIS_TO_CONTENT_GAP_RATIO, MAIN_CONTENT_WIDTH_RATIO, RIGHT_SCREEN_GAP_RATIO],
    gap="small",
)

with main_options_col:
    st.markdown(f"<div style='height:{ALGORITHM_PARAMETERS_ANALYSIS_TOP_SPACER}'></div>", unsafe_allow_html=True)
    render_algorithm_parameters_pdf_button()
    with st.expander("Analysis settings", expanded=False):
        shift_plus_1_hour = st.checkbox("Shift timestamps by +1 hour", value=True)

        analysis_options_title("Storm detection")
        episode_level = st.number_input("EPISODE_LEVEL", value=-20, step=1)
        storm_level = st.number_input("STORM_LEVEL", value=-50, step=1)
        local_min_radius_hours = st.number_input("LOCAL_MIN_RADIUS_HOURS", value=3, step=1, min_value=0)
        minima_split_factor = st.number_input("MINIMA_SPLIT_FACTOR", value=0.4, step=0.05, format="%.2f")
        storm_limit = st.number_input("STORM_LIMIT", value=-1000, step=10)

        analysis_options_title("Filtering (for main phase)")
        strong_storm_threshold = st.number_input("STRONG_STORM_THRESHOLD", value=-100, step=1)
        strong_mark = st.number_input("STRONG_MARK", value=-50, step=1)
        weak_mark_factor = calculate_weak_mark_factor(strong_storm_threshold, strong_mark)
        if pd.isna(weak_mark_factor):
            st.error("STRONG_STORM_THRESHOLD cannot be 0 because WEAK_MARK is calculated from STRONG_MARK / STRONG_STORM_THRESHOLD.")
        else:
            st.caption(f"WEAK_MARK = Dst_min × {weak_mark_factor:.3f}")
        local_max_radius_hours = st.number_input("LOCAL_MAX_RADIUS_HOURS", value=3, step=1, min_value=0)
        disturbance_filter = st.checkbox("DISTURBANCE_FILTER", value=True)
        disturbance_level = st.number_input(
            "DISTURB_LEVEL",
            value=-100,
            step=1,
            disabled=not disturbance_filter,
        )
        disturb_dip_count = st.number_input(
            "DISTURB_DIP_COUNT",
            value=2,
            step=1,
            min_value=1,
            disabled=not disturbance_filter,
        )
        disturb_dip_radius_hours = st.number_input(
            "DISTURB_DIP_RADIUS_HOURS",
            value=3,
            step=1,
            min_value=0,
            disabled=not disturbance_filter,
        )

with main_content_col:
    header_left, header_mid, header_right = st.columns([4.35, 0.18, 1.47], gap="small")

    with header_left:
        st.markdown(
            "<h1 class='app-main-title'><span class='title-line'>Magnetic Storm Detection</span><br><span class='title-line'>and Main Phase Analysis</span></h1>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
        
        try:
            date_start_col, date_end_col, _spacer_col, start_col, loading_col = st.columns(
                [1.38, 1.38, 0.04, 0.95, 1.25],
                vertical_alignment="bottom",
            )
        except TypeError:
            
            date_start_col, date_end_col, _spacer_col, start_col, loading_col = st.columns([1.38, 1.38, 0.04, 0.95, 1.25])
        with date_start_col:
            st.markdown(
                "<div style='color:white; font-weight:800; font-size:0.95rem; margin-bottom:0.25rem;'>DATE_START</div>",
                unsafe_allow_html=True
            )
            date_start = st.text_input(
                "DATE_START",
                value="1964-01-01 00:00",
                label_visibility="collapsed",
                key="date_start_input"
            )
        with date_end_col:
            st.markdown(
                "<div style='color:white; font-weight:800; font-size:0.95rem; margin-bottom:0.25rem;'>DATE_END</div>",
                unsafe_allow_html=True
            )
            date_end = st.text_input(
                "DATE_END",
                value="2026-06-01 00:00",
                label_visibility="collapsed",
                key="date_end_input"
            )
        with start_col:
            run = st.button("START", type="primary")

        with loading_col:
            
            
            
            loading_indicator = st.empty()
            loading_indicator.markdown(
                "<div class='working-indicator-slot working-indicator-placeholder'></div>",
                unsafe_allow_html=True,
            )

    
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False

    with header_mid:
        st.markdown("<div style='height:3.2rem;'></div>", unsafe_allow_html=True)
    
    with header_right:
        
        
        st.empty()

    if run:
        st.session_state.analysis_started = True

    if st.session_state.analysis_started:
        
        
        loading_indicator.markdown(
            """
            <style>
            @keyframes stormLoaderSpin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .working-indicator-slot {
                height: 46px;
                min-height: 46px;
                display: flex;
                align-items: center;
                justify-content: flex-start;
                padding: 0;
                margin-top: -10px !important;
                transform: translateY(-15px) !important;
                box-sizing: border-box;
                line-height: 1;
            }
            .working-indicator-placeholder {
                visibility: hidden;
            }
            .storm-working-indicator {
                gap: 0.55rem;
                color: black !important;
                font-weight: 900;
                font-size: 20px;
                white-space: nowrap;
                border: none !important;
                border-radius: 12px;
                background: #ffffff !important;
                padding: 7px 13px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.35);
            }
            .storm-loader-dot {
                width: 22px;
                height: 22px;
                border: 4px solid rgba(0,0,0,0.22);
                border-top-color: #000000;
                border-radius: 50%;
                animation: stormLoaderSpin 0.8s linear infinite;
                display: inline-block;
                flex: 0 0 auto;
            }
            </style>
            <div class="working-indicator-slot storm-working-indicator">
                <span class="storm-loader-dot"></span>
                <span style="font-size:20px; font-weight:900; letter-spacing:0.02em; color:black;">Working</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not st.session_state.analysis_started:
        st.stop()

    try:
        raw_text = load_local_omni_text(OMNI_DATA_FILENAME)
        analysis_cache_key = (
            hashlib.sha256(raw_text.encode("utf-8", errors="ignore")).hexdigest(),
            bool(shift_plus_1_hour),
            str(date_start),
            str(date_end),
            float(episode_level),
            float(storm_level),
            int(local_min_radius_hours),
            float(minima_split_factor),
            float(storm_limit),
            float(strong_storm_threshold),
            float(strong_mark),
            int(local_max_radius_hours),
            bool(disturbance_filter),
            float(disturbance_level),
            int(disturb_dip_count),
            int(disturb_dip_radius_hours),
        )
        analysis_results = run_storm_analysis_cached(
            raw_text,
            bool(shift_plus_1_hour),
            date_start,
            date_end,
            float(episode_level),
            float(storm_level),
            int(local_min_radius_hours),
            float(minima_split_factor),
            float(storm_limit),
            float(strong_storm_threshold),
            float(strong_mark),
            int(local_max_radius_hours),
            bool(disturbance_filter),
            float(disturbance_level),
            int(disturb_dip_count),
            int(disturb_dip_radius_hours),
        )

        
        

        
        
        
        
        
        run_info = analysis_results["run_info"]
        omni_df = analysis_results["omni_df"]
        storms_df = analysis_results["storms_df"]
        episodes_df = analysis_results["episodes_df"]
        multi_df = analysis_results["multi_df"]
        stage1_summary = analysis_results["stage1_summary"]
        filtered_df = analysis_results["filtered_df"]
        disturbances_df = analysis_results["disturbances_df"]
        excluded_df = analysis_results["excluded_df"]
        stage2_summary = analysis_results["stage2_summary"]
        data_complete_storms_df = analysis_results["data_complete_storms_df"]
        metrics_df = analysis_results["metrics_df"]
        stage3_summary = analysis_results["stage3_summary"]
        corr_all_df = analysis_results["corr_all_df"]
        filtered_display_df = analysis_results["filtered_display_df"]
        disturbances_display_df = analysis_results["disturbances_display_df"]
        excluded_display_df = analysis_results["excluded_display_df"]
        stage3_all_data_complete_storms_df = analysis_results["stage3_all_data_complete_storms_df"]
        stage3_peak_delays_df = analysis_results["stage3_peak_delays_df"]
        stage3_full_peak_data_df = analysis_results["stage3_full_peak_data_df"]
        stage3_replacements_df = analysis_results["stage3_replacements_df"]
        stage3_rejected_quality_df = analysis_results["stage3_rejected_quality_df"]

        render_context = {
            "analysis_cache_key": analysis_cache_key,
            "run_info": run_info,
            "omni_df": omni_df,
            "storms_df": storms_df,
            "episodes_df": episodes_df,
            "multi_df": multi_df,
            "stage1_summary": stage1_summary,
            "filtered_df": filtered_df,
            "disturbances_df": disturbances_df,
            "excluded_df": excluded_df,
            "stage2_summary": stage2_summary,
            "data_complete_storms_df": data_complete_storms_df,
            "metrics_df": metrics_df,
            "stage3_summary": stage3_summary,
            "corr_all_df": corr_all_df,
            "filtered_display_df": filtered_display_df,
            "disturbances_display_df": disturbances_display_df,
            "excluded_display_df": excluded_display_df,
            "stage3_all_data_complete_storms_df": stage3_all_data_complete_storms_df,
            "stage3_peak_delays_df": stage3_peak_delays_df,
            "stage3_full_peak_data_df": stage3_full_peak_data_df,
            "stage3_replacements_df": stage3_replacements_df,
            "stage3_rejected_quality_df": stage3_rejected_quality_df,
        }

    except FileNotFoundError as e:
        loading_indicator.empty()
        st.error(str(e))
        st.info('Upload the OMNI data file named "1964-may 2026.txt" to the same folder as the running Python file.')
        st.stop()
    except ValueError as e:
        loading_indicator.empty()
        st.error(str(e))
        st.stop()
    except Exception as e:
        loading_indicator.empty()
        st.exception(e)
        st.stop()


    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total storms", f"{len(storms_df):,}")
    col2.metric("Filtered storms", f"{len(filtered_df):,}")
    col3.metric("Data-complete storms", f"{len(data_complete_storms_df):,}")

    with st.expander("Run summary", expanded=False):
        s1a, s1b = st.columns(2)
        s1a.write(f"**Rows read:** {int(run_info.get('rows_read', 0)):,}")
        s1a.write(f"**Rows analyzed:** {int(run_info.get('rows_analyzed', 0)):,}")
        s1a.write(f"**Bad/non-data lines skipped:** {int(run_info.get('bad_rows', 0)):,}")
        s1b.write(f"**Full file span:** {fmt_dt(run_info.get('full_span_start'))} → {fmt_dt(run_info.get('full_span_end'))}")
        s1b.write(f"**Analyzed span:** {fmt_dt(run_info.get('analyzed_span_start'))} → {fmt_dt(run_info.get('analyzed_span_end'))}")


    
    
    
    
    
    section_options = [
        "Overview",
        "Storm detection",
        "Main phase filtering",
        "Data-complete",
        "Correlations",
        "Downloads",
        "Extras",
    ]
    selected_tab = st.radio(
        "Section",
        section_options,
        index=section_options.index(st.session_state.get("active_section", "Overview"))
        if st.session_state.get("active_section", "Overview") in section_options else 0,
        horizontal=True,
        key="active_section",
        label_visibility="collapsed",
    )

    try:
        section_panel = st.container(border=True, key="active_section_panel")
    except TypeError:
        section_panel = st.container(border=True)
    section_renderers = {
        "Overview": render_overview_section,
        "Storm detection": render_storm_detection_section,
        "Main phase filtering": render_main_phase_filtering_section,
        "Data-complete": render_data_complete_section,
        "Correlations": render_correlations_section,
        "Downloads": render_downloads_section,
        "Extras": render_extras_section,
    }

    with section_panel:
        section_renderers[selected_tab](render_context)

    try:
        loading_indicator.empty()
    except Exception:
        pass
