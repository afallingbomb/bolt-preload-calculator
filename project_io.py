"""Project save/load (JSON), unit auto-conversion, and the custom-material data
helpers that back both.

Everything here operates on Streamlit ``session_state``: it (de)serialises the full
input set, coerces loaded values to the types the widgets expect, and converts the
already-entered values when the user flips the unit system. Kept separate from the
page/rendering code in app.py so the persistence layer can evolve independently.
"""
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from mechanics import BoltMaterial, JointMaterial


def _num(v: Any) -> float:
    """Best-effort float conversion (blank/invalid -> 0.0)."""
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


_JOINT_MAT_COLS = ["Name", "Syc (MPa)", "E (GPa)", "CTE (µm/m·°C)"]
_BOLT_MAT_COLS = ["Name", "Sp (MPa)", "Sy (MPa)", "Sut (MPa)", "Se (MPa)", "E (GPa)", "CTE (µm/m·°C)"]


def parse_custom_materials(
        joint_rows: list, bolt_rows: list) -> Tuple[Dict[str, JointMaterial], Dict[str, BoltMaterial]]:
    """Convert custom-material row lists into property dicts in SI units (MPa,
    MPa-modulus, 1/°C). Rows lacking a name or a positive modulus/strength are
    ignored. Pure (no Streamlit calls) so it can run before the dropdowns render."""
    joint: Dict[str, JointMaterial] = {}
    for r in joint_rows:
        name = str(r.get("Name") or "").strip()
        e_gpa, syc = _num(r.get("E (GPa)")), _num(r.get("Syc (MPa)"))
        if name and e_gpa > 0 and syc > 0:
            joint[name] = {"Syc": syc, "E": e_gpa * 1000.0,
                           "CTE": _num(r.get("CTE (µm/m·°C)")) * 1e-6}
    bolt: Dict[str, BoltMaterial] = {}
    for r in bolt_rows:
        name = str(r.get("Name") or "").strip()
        sp, sut, se, e_gpa = (_num(r.get("Sp (MPa)")), _num(r.get("Sut (MPa)")),
                              _num(r.get("Se (MPa)")), _num(r.get("E (GPa)")))
        if name and sp > 0 and sut > 0 and se > 0 and e_gpa > 0:
            sy, cte = _num(r.get("Sy (MPa)")), _num(r.get("CTE (µm/m·°C)"))
            bolt[name] = {"Sp": sp, "Sy": sy if sy > 0 else sp / 0.9, "Sut": sut, "Se": se,
                          "E": e_gpa * 1000.0, "CTE": cte * 1e-6 if cte > 0 else 11.5e-6}
    return joint, bolt


def duplicate_material_names(rows: list) -> List[str]:
    """Names (stripped, case-sensitive) that appear on more than one row.

    Custom materials are keyed by name, so duplicates silently overwrite each other
    in ``parse_custom_materials``; the editor surfaces this list as a warning."""
    seen: dict = {}
    for r in rows:
        name = str(r.get("Name") or "").strip()
        if name:
            seen[name] = seen.get(name, 0) + 1
    return [name for name, count in seen.items() if count > 1]


def _editor_df(rows: list, columns: list) -> Any:
    """Typed (possibly empty) DataFrame so the editor always shows its columns."""
    df: Any = pd.DataFrame(rows, columns=columns)
    for c in columns:
        if c != "Name":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def _editor_rows(df: Any) -> list:
    """Edited DataFrame -> JSON-friendly row list (NaN -> None)."""
    return list(df.where(pd.notnull(df), None).to_dict("records"))


# --- Project save/load and unit auto-conversion --------------------------------
INPUT_KEYS = [
    "units", "bolt_size", "thread_series", "bolt_type", "bolt_material", "use_washer", "is_permanent",
    "friction_condition", "tightening_method", "embedment_um", "load_intro_factor",
    "required_fos", "use_group",
    "use_thermal", "use_fatigue", "use_thread",
    "temp_assembly", "temp_operating", "ext_max", "ext_min", "num_bolts", "fatigue_criterion",
    "thread_engagement", "internal_thread_mat", "pattern_type",
    "g_rows", "g_cols", "g_px", "g_py", "g_n", "g_bcd", "g_start",
    "g_axial_max", "g_axial_min", "g_moment_max", "g_moment_min", "moment_axis_label",
    "g_shear", "g_ecc", "slip_mu", "slip_ns",
]
_TABLE_KEYS = ["layers", "group_table", "custom_joint_rows", "custom_bolt_rows"]
_LEN_KEYS = ("thread_engagement", "g_px", "g_py", "g_bcd", "g_ecc")
_FORCE_KEYS = ("ext_max", "ext_min", "g_axial_max", "g_axial_min", "g_shear")
_TEMP_KEYS = ("temp_assembly", "temp_operating")
_MOMENT_KEYS = ("g_moment_max", "g_moment_min")
# Type buckets used to coerce loaded project values back to what each widget expects.
_INT_KEYS = ("num_bolts", "g_rows", "g_cols", "g_n", "slip_ns")
_BOOL_KEYS = ("use_washer", "is_permanent", "use_group", "use_thermal", "use_fatigue", "use_thread")
_FLOAT_KEYS = (_LEN_KEYS + _FORCE_KEYS + _TEMP_KEYS + _MOMENT_KEYS
               + ("embedment_um", "load_intro_factor", "required_fos", "g_start", "slip_mu"))
PROJECT_SCHEMA = "bolt-preload-project-v1"


def _validate_choice(key: str, options: list) -> None:
    """Drop a stored selectbox value no longer in ``options`` (e.g. after a unit
    switch or removing a custom material) so the widget reverts to its default."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def convert_inputs(to_metric: bool) -> None:
    """Convert already-entered values when the user flips the unit system."""
    f_len = 25.4 if to_metric else 1.0 / 25.4
    f_force = 4.44822 if to_metric else 1.0 / 4.44822
    f_mom = 1.355818 if to_metric else 1.0 / 1.355818
    for k in _LEN_KEYS:
        if k in st.session_state:
            st.session_state[k] = round(float(st.session_state[k]) * f_len, 6)
    for k in _FORCE_KEYS:
        if k in st.session_state:
            st.session_state[k] = round(float(st.session_state[k]) * f_force, 6)
    for k in _MOMENT_KEYS:
        if k in st.session_state:
            st.session_state[k] = round(float(st.session_state[k]) * f_mom, 6)
    for k in _TEMP_KEYS:
        if k in st.session_state:
            v = float(st.session_state[k])
            st.session_state[k] = round((v - 32.0) * 5.0 / 9.0 if to_metric else v * 9.0 / 5.0 + 32.0, 4)
    for row in st.session_state.get("layers", []):
        if row.get("Thickness") is not None:
            row["Thickness"] = round(float(row["Thickness"]) * f_len, 6)
    for row in st.session_state.get("group_table", []):
        for c in ("X", "Y"):
            if row.get(c) is not None:
                row[c] = round(float(row[c]) * f_len, 6)


def build_project() -> dict:
    """Gather every input from session_state into a serialisable project dict."""
    proj: dict = {"schema": PROJECT_SCHEMA,
                  "scalars": {k: st.session_state[k] for k in INPUT_KEYS if k in st.session_state}}
    for tbl in _TABLE_KEYS:
        proj[tbl] = st.session_state.get(tbl, [])
    return proj


def _coerce_scalar(key: str, value: Any) -> Any:
    """Coerce a loaded scalar to the type its widget expects (raises on failure).

    String-valued selections are left as-is; they are validated against the live
    options later by _validate_choice."""
    if key in _BOOL_KEYS:
        return bool(value)
    if key in _INT_KEYS:
        return int(round(float(value)))
    if key in _FLOAT_KEYS:
        return float(value)
    return value


def apply_project(proj: dict) -> None:
    """Load a project into session_state. Call before widgets render, then rerun.

    Validates the schema and coerces each scalar to the type its widget expects, so
    a hand-edited, stale or foreign file cannot drop a wrong-typed value into a keyed
    widget (which would otherwise crash on the next rerun, outside the load guard).
    Uncoercible scalars and malformed tables are skipped, falling back to defaults."""
    if not isinstance(proj, dict) or proj.get("schema") != PROJECT_SCHEMA:
        raise ValueError(f"Unrecognised project file (expected schema '{PROJECT_SCHEMA}').")
    scalars = proj.get("scalars", {})
    if not isinstance(scalars, dict):
        raise ValueError("Project 'scalars' section is malformed.")
    for k, v in scalars.items():
        if k not in INPUT_KEYS:
            continue
        try:
            st.session_state[k] = _coerce_scalar(k, v)
        except (TypeError, ValueError):
            continue  # leave the widget at its default
    for tbl in _TABLE_KEYS:
        if isinstance(proj.get(tbl), list):
            st.session_state[tbl] = proj[tbl]

    # custom_joint_rows / custom_bolt_rows are restored by the loop above; the
    # material editors rebuild their DataFrame from those rows each run, so there
    # is no separate ``*_materials_df`` to seed here.
    st.session_state["_prev_units"] = scalars.get("units")
