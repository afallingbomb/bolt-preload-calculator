import html
import json
import logging
import math
from typing import Any, List, Optional, Tuple

import streamlit as st

from mechanics import (
    BOLT_SIZES_METRIC, BOLT_SIZES_IMPERIAL,
    BOLT_THREAD_SERIES_METRIC, BOLT_THREAD_SERIES_IMPERIAL,
    BOLT_MATERIALS_METRIC, BOLT_MATERIALS_IMPERIAL,
    JOINT_MATERIALS,
    BOLT_TYPES,
    FRICTION_COEFFICIENTS,
    TIGHTENING_METHODS,
    FATIGUE_CRITERIA,
    STANDARD_BOLT_LENGTHS_METRIC_MM, STANDARD_BOLT_LENGTHS_IMPERIAL_MM,
    PreloadResult, Layer,
    calculate_preload, calculate_bearing_diameter,
    rectangular_pattern, circular_pattern, analyze_bolt_group,
    preload_from_torque, preload_from_yield_percent, tightening_angle, recommend_bolt_length,
    grip_thread_engagement, bolt_hardware_reference, recommend_bolt,
    clamp_load_budget, combined_tension_shear_fos,
    thread_series_options, thread_designation,
    exact_tightening_torque
)
from report import build_pdf_report, build_theory_pdf
from theory import THEORY_BLOCKS, THEORY_TITLE
from charts import (
    make_joint_diagram, make_bolt_group_figure, make_haigh_diagram, make_cross_section_diagram,
    fig_to_png,
    alt_joint_diagram, alt_bolt_force_chart, alt_clamp_waterfall, alt_load_sharing,
    alt_bolt_group_chart,
)
from project_io import (
    parse_custom_materials, duplicate_material_names, _editor_df,
    _JOINT_MAT_COLS, _BOLT_MAT_COLS,
    convert_inputs, build_project, apply_project, _validate_choice,
)
from fe_import import render_fe_import_tab
from analysis import AnalysisContext, DisplayUnits, collect_findings, fos_str
from version import __version__

logger = logging.getLogger("bolt_calculator")


@st.cache_data(show_spinner=False)
def generate_theory_pdf() -> bytes:
    """The theory manual (theory.THEORY_BLOCKS) as a standalone PDF; static, so cached."""
    return build_theory_pdf(THEORY_TITLE, THEORY_BLOCKS)


# Page Configuration
st.set_page_config(
    page_title="Bolt Preload & Joint Analysis",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Design verdict banner + colour-coded safety cards (HTML, styled by style.css)
# =============================================================================
def _fos_status(value: float, required: float) -> str:
    """Traffic-light bucket for a factor of safety relative to the required FoS:
    green at/above the requirement, amber with positive margin but below it, red below 1."""
    if value == float("inf"):
        return "ok"
    if value < 1.0:
        return "bad"
    if value < required:
        return "warn"
    return "ok"


def _stat_card(label: str, value: str, status: str, sub: str = "") -> str:
    """One colour-coded safety card. ``status`` in {ok, warn, bad, na}."""
    # Escape any caller-supplied text before it reaches unsafe_allow_html. ``status``
    # is a fixed token used in a CSS class, so it is intentionally left unescaped.
    label, value, sub = html.escape(label), html.escape(value), html.escape(sub)
    sub_html = f'<div class="sc-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="stat-card sc-{status}">'
        f'<div class="sc-lbl">{label}</div>'
        f'<div class="sc-val">{value}</div>'
        f'{sub_html}</div>'
    )


def _verdict(applicable: List[Tuple[str, float]], required: float,
             has_error: bool) -> Tuple[str, float, str]:
    """Overall pass/marginal/fail from the governing (minimum) FoS vs the required FoS.

    Hard errors (a FoS below 1, or an exceeded assembly limit) fail the joint; a governing
    FoS below the required value is marginal. Soft advisories (e.g. high tightening
    utilisation) do NOT downgrade a joint whose factors of safety all meet the requirement.
    """
    if applicable:
        gov_label, gov_val = min(applicable, key=lambda kv: kv[1])
    else:
        gov_label, gov_val = "", float("inf")
    if has_error or gov_val < 1.0:
        status = "fail"
    elif gov_val < required:
        status = "marginal"
    else:
        status = "pass"
    return status, gov_val, gov_label


def render_verdict_banner(status: str, gov_val: float, gov_label: str,
                          required: float, n_advisories: int = 0) -> None:
    """Big PASS / MARGINAL / FAIL banner with the governing factor of safety."""
    icon, title = {"pass": ("✓", "PASS"),
                   "marginal": ("!", "MARGINAL"),
                   "fail": ("✕", "FAIL")}[status]
    fos_txt = "∞" if gov_val == float("inf") else f"{gov_val:.2f}"
    req_txt = f"{required:.2f}"
    if gov_label:
        tail = {"pass": f"Meets the required FoS of {req_txt}.",
                "marginal": f"Below the required FoS of {req_txt} — review.",
                "fail": "A check fails (FoS < 1) or an assembly limit is exceeded."}[status]
        sub = f"Governing factor of safety {fos_txt} ({html.escape(gov_label)}). {tail}"
    else:
        sub = "No applicable strength checks for the current inputs."
    if n_advisories:
        sub += f" · {n_advisories} advisor{'y' if n_advisories == 1 else 'ies'} below"
    st.markdown(
        f'<div class="verdict verdict-{status}">'
        f'<div class="v-icon">{icon}</div>'
        f'<div class="v-body"><div class="v-title">{title}</div>'
        f'<div class="v-sub">{sub}</div></div>'
        f'<div class="v-fos"><span class="v-num">{fos_txt}</span>'
        f'<span class="v-cap">min FoS · req {req_txt}</span></div></div>',
        unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=8)
def generate_pdf_report(
    title: str, subtitle: str,
    result_rows: Tuple[Tuple[str, str], ...],
    warnings: Tuple[str, ...],
    bolt_group_rows: Optional[Tuple[Tuple[str, str], ...]],
    assumptions: Tuple[str, ...],
    results: PreloadResult, ext_max_N: float,
    units: DisplayUnits,
    group_coords: Tuple[Tuple[float, float], ...],
    group_tensions: Tuple[float, ...], group_gov_idx: int,
    cs_layers: Tuple[Tuple[str, float], ...] = (),
    d_for_cs: float = 0.0, dw_for_cs: float = 0.0, use_washer_cs: bool = False,
    group_shear_vectors: Tuple[Tuple[float, float], ...] = (),
) -> bytes:
    """Build the (light-themed) report figures and the PDF, cached on its inputs.

    Streamlit reruns the whole script on every interaction; without this cache the
    figures would be rasterised and the PDF rebuilt on each rerun even when nothing
    relevant changed and the user never downloads it. Caching keyed on these
    (hashable) arguments rebuilds only when an input actually changes. Arguments
    are passed as tuples so they hash deterministically."""
    figs: List[bytes] = []
    if cs_layers and d_for_cs > 0:
        fig_cs = make_cross_section_diagram(
            [{"Material": m, "thickness": t} for m, t in cs_layers],
            d_for_cs, dw_for_cs, use_washer_cs, units.length_factor, units.len_unit, dark=False)
        if fig_cs is not None:
            figs.append(fig_to_png(fig_cs))
    fig_jd = make_joint_diagram(
        results['kb_N_mm'], results['km_N_mm'], results['operating_preload_N'],
        ext_max_N, results['joint_constant_C'],
        units.force_factor, units.length_factor, units.force_unit, units.len_unit, dark=False)
    if fig_jd is not None:
        figs.append(fig_to_png(fig_jd))
    if group_coords:
        fig_bg = make_bolt_group_figure(
            list(group_coords), list(group_tensions), group_gov_idx,
            units.length_factor, units.force_factor, units.len_unit, units.force_unit,
            shear_vectors=list(group_shear_vectors) if group_shear_vectors else None, dark=False)
        figs.append(fig_to_png(fig_bg))
    fig_h = make_haigh_diagram(results, units.stress_factor, units.stress_unit, dark=False)
    if fig_h is not None:
        figs.append(fig_to_png(fig_h))

    return build_pdf_report(
        title=title, subtitle=subtitle,
        result_rows=list(result_rows),
        warnings=list(warnings) or None,
        bolt_group_rows=list(bolt_group_rows) if bolt_group_rows else None,
        figures=figs,
        assumptions=list(assumptions),
    )


# =============================================================================
# Custom-materials tab (the data helpers live in project_io)
# =============================================================================
def _builtin_material_table() -> None:
    """Read-only reference of the built-in materials (SI), with data sources."""
    st.caption(f"{len(JOINT_MATERIALS)} joint materials and {len(BOLT_MATERIALS_METRIC)} metric bolt "
               "grades. Values are nominal, room-temperature and condition-specific — verify against "
               "certified data before design use.")
    st.markdown("**Joint / clamped materials**")
    st.dataframe([{"Material": k, "Syc (MPa)": v["Syc"], "E (GPa)": round(v["E"] / 1000.0, 1),
                   "CTE (µm/m·°C)": round(v["CTE"] * 1e6, 1), "Source": v.get("source", "—")}
                  for k, v in JOINT_MATERIALS.items()], hide_index=True, width="stretch")
    st.markdown("**Bolt materials (metric grades)**")
    st.dataframe([{"Grade": k, "Sp": v["Sp"], "Sy": v.get("Sy"), "Sut": v["Sut"], "Se": v["Se"],
                   "E (GPa)": round(v["E"] / 1000.0, 1), "CTE (µm/m·°C)": round(v["CTE"] * 1e6, 1),
                   "Source": v.get("source", "—")}
                  for k, v in BOLT_MATERIALS_METRIC.items()], hide_index=True, width="stretch")


@st.dialog("Add Clamped-joint Material")
def add_joint_material_dialog() -> None:
    st.write("Enter properties for the new joint/layer material.")
    with st.form("new_joint_form"):
        name = st.text_input("Name")
        syc = st.number_input("Syc (MPa) [Compressive yield]", min_value=0.0, format="%.0f", step=10.0)
        e = st.number_input("E (GPa) [Young's modulus]", min_value=0.0, format="%.1f", step=1.0)
        cte = st.number_input("CTE (µm/m·°C) [Thermal expansion]", min_value=0.0, format="%.1f", step=1.0)

        if st.form_submit_button("Add Material"):
            name_clean = name.strip()
            existing_names = [r["Name"].lower() for r in st.session_state.custom_joint_rows]
            
            if not name_clean:
                st.error("Please provide a name.")
            elif name_clean.lower() in existing_names:
                st.error(f"A joint material named '{name_clean}' already exists.")
            elif syc <= 0 or e <= 0:
                st.error("Syc and E must be greater than 0.")
            else:
                st.session_state.custom_joint_rows.append({
                    "Name": name_clean, "Syc (MPa)": syc, "E (GPa)": e, "CTE (µm/m·°C)": cte
                })
                st.rerun()


@st.dialog("Add Bolt Material")
def add_bolt_material_dialog() -> None:
    st.write("Enter properties for the new bolt material.")
    with st.form("new_bolt_form"):
        name = st.text_input("Name")
        sp = st.number_input("Sp (MPa) [Proof strength]", min_value=0.0, format="%.0f", step=10.0)
        sy = st.number_input("Sy (MPa) [Yield strength]", min_value=0.0, format="%.0f", step=10.0)
        sut = st.number_input("Sut (MPa) [Ultimate strength]", min_value=0.0, format="%.0f", step=10.0)
        se = st.number_input("Se (MPa) [Endurance limit]", min_value=0.0, format="%.0f", step=10.0)
        e = st.number_input("E (GPa) [Young's modulus]", min_value=0.0, format="%.1f", step=1.0)
        cte = st.number_input("CTE (µm/m·°C) [Thermal expansion]", min_value=0.0, format="%.1f", step=1.0)

        if st.form_submit_button("Add Material"):
            name_clean = name.strip()
            existing_names = [r["Name"].lower() for r in st.session_state.custom_bolt_rows]
            
            if not name_clean:
                st.error("Please provide a name.")
            elif name_clean.lower() in existing_names:
                st.error(f"A bolt material named '{name_clean}' already exists.")
            elif sp <= 0 or sut <= 0 or e <= 0:
                st.error("Sp, Sut, and E must be greater than 0.")
            else:
                st.session_state.custom_bolt_rows.append({
                    "Name": name_clean, "Sp (MPa)": sp, "Sy (MPa)": sy, "Sut (MPa)": sut,
                    "Se (MPa)": se, "E (GPa)": e, "CTE (µm/m·°C)": cte
                })
                st.rerun()


def render_materials_tab() -> None:
    """Full editor for user-defined materials (its own tab)."""
    st.subheader("Custom Materials")
    st.markdown(
        "Define your own materials in **SI units**. They become available in the bolt-material and "
        "joint-layer dropdowns as soon as they are added. "
        "Use the **Add** buttons to create a new material, and click the **Delete** checkbox to remove them.")

    st.markdown("##### Clamped-joint / layer materials")
    st.caption("Syc = compressive (bearing) yield · E = Young's modulus · CTE = thermal expansion.")

    if st.button("➕ Add Joint Material", use_container_width=True):
        add_joint_material_dialog()

    df_joint = _editor_df(st.session_state.custom_joint_rows, _JOINT_MAT_COLS)
    if not df_joint.empty:
        df_joint.insert(0, "Delete", False)
        edited_joint = st.data_editor(
            df_joint, hide_index=True, width="stretch",
            disabled=_JOINT_MAT_COLS,  # only the Delete column is editable
            column_config={
                "Delete": st.column_config.CheckboxColumn("🗑️", default=False, width="small"),
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "Syc (MPa)": st.column_config.NumberColumn("Syc (MPa)", format="%.0f"),
                "E (GPa)": st.column_config.NumberColumn("E (GPa)", format="%.1f"),
                "CTE (µm/m·°C)": st.column_config.NumberColumn("CTE (µm/m·°C)", format="%.1f"),
            })
        if edited_joint["Delete"].any():
            to_delete = edited_joint[edited_joint["Delete"]]["Name"].tolist()
            st.session_state.custom_joint_rows = [
                r for r in st.session_state.custom_joint_rows if r["Name"] not in to_delete]
            st.rerun()
    else:
        st.info("No custom joint materials defined.")

    st.divider()

    st.markdown("##### Bolt materials")
    st.caption("Sp = proof · Sy = yield · Sut = ultimate · Se = endurance (rolled threads, incl. Kf).")

    if st.button("➕ Add Bolt Material", use_container_width=True):
        add_bolt_material_dialog()

    df_bolt = _editor_df(st.session_state.custom_bolt_rows, _BOLT_MAT_COLS)
    if not df_bolt.empty:
        df_bolt.insert(0, "Delete", False)
        edited_bolt = st.data_editor(
            df_bolt, hide_index=True, width="stretch",
            disabled=_BOLT_MAT_COLS,  # only the Delete column is editable
            column_config={
                "Delete": st.column_config.CheckboxColumn("🗑️", default=False, width="small"),
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "Sp (MPa)": st.column_config.NumberColumn("Sp (MPa)", format="%.0f"),
                "Sy (MPa)": st.column_config.NumberColumn("Sy (MPa)", format="%.0f"),
                "Sut (MPa)": st.column_config.NumberColumn("Sut (MPa)", format="%.0f"),
                "Se (MPa)": st.column_config.NumberColumn("Se (MPa)", format="%.0f"),
                "E (GPa)": st.column_config.NumberColumn("E (GPa)", format="%.1f"),
                "CTE (µm/m·°C)": st.column_config.NumberColumn("CTE (µm/m·°C)", format="%.1f"),
            })
        if edited_bolt["Delete"].any():
            to_delete_b = edited_bolt[edited_bolt["Delete"]]["Name"].tolist()
            st.session_state.custom_bolt_rows = [
                r for r in st.session_state.custom_bolt_rows if r["Name"] not in to_delete_b]
            st.rerun()
    else:
        st.info("No custom bolt materials defined.")

    cj, cb = parse_custom_materials(st.session_state.custom_joint_rows, st.session_state.custom_bolt_rows)
    a_col, b_col = st.columns(2)
    a_col.success(f"Active joint materials: {', '.join(cj) if cj else '—'}")
    b_col.success(f"Active bolt materials: {', '.join(cb) if cb else '—'}")
    _dups = sorted(set(duplicate_material_names(st.session_state.custom_joint_rows)
                       + duplicate_material_names(st.session_state.custom_bolt_rows)))
    if _dups:
        st.warning("⚠️ Duplicate material name(s): **" + "**, **".join(_dups) + "**. Materials are "
                   "keyed by name, so only the last row with a given name is used — rename or remove "
                   "the duplicates.")
    st.caption("Custom materials are saved inside the project file (sidebar → Save / load project). "
               "Properties are intrinsic, so they are entered in SI regardless of the display units.")

    with st.expander("📖 Built-in materials (reference)"):
        _builtin_material_table()


# Load Custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.markdown(
    '<div class="app-hero">'
    '<div class="hero-icon">⚙️</div>'
    '<div class="hero-text">'
    '<h1 class="hero-title">Bolt Preload &amp; Joint Analysis</h1>'
    '</div></div>',
    unsafe_allow_html=True)
st.caption(
    "⚠️ Results use **nominal** material and reference data with standard first-pass models — "
    "verify against certified data and the governing design code before design use.")

# --- Sidebar Inputs ---
with st.sidebar:
    # Project load runs FIRST so restored values exist before any keyed widget.
    with st.expander("💾 Save / load project"):
        _uploaded = st.file_uploader("Load project (JSON)", type="json", key="project_uploader")
        if _uploaded is not None and st.session_state.get("_loaded_file_id") != _uploaded.file_id:
            try:
                apply_project(json.load(_uploaded))
                st.session_state["_loaded_file_id"] = _uploaded.file_id
                st.rerun()
            except Exception as _exc:
                st.error(f"Could not load project: {_exc}")
        st.caption("The **Save project** button is at the bottom, with the other exports.")

    st.header("Unit System")
    unit_system = st.radio("Select Units", ["Metric (mm, N, MPa, °C)", "Imperial (in, lbf, psi, °F)"],
                           key="units")
    is_metric = "Metric" in unit_system

    # Auto-convert entered values when the unit system changes.
    _prev_units = st.session_state.get("_prev_units")
    if _prev_units is not None and _prev_units != unit_system:
        convert_inputs(to_metric=is_metric)
    st.session_state["_prev_units"] = unit_system

    # Unit-dependent reference data, merged with any user-defined materials. The
    # custom-material editors live in the Custom Materials tab; here we only read
    # the stored rows so the merged lists are ready for every dropdown below.
    sizes_dict = BOLT_SIZES_METRIC if is_metric else BOLT_SIZES_IMPERIAL
    bolt_materials_dict = BOLT_MATERIALS_METRIC if is_metric else BOLT_MATERIALS_IMPERIAL
    if "custom_joint_rows" not in st.session_state:
        st.session_state.custom_joint_rows = []
    if "custom_bolt_rows" not in st.session_state:
        st.session_state.custom_bolt_rows = []
    custom_joint_materials, custom_bolt_materials = parse_custom_materials(
        st.session_state.custom_joint_rows, st.session_state.custom_bolt_rows)
    joint_materials_all = {**JOINT_MATERIALS, **custom_joint_materials}
    bolt_materials_all = {**bolt_materials_dict, **custom_bolt_materials}

    st.markdown("---")
    st.header("Bolt Configuration")

    _validate_choice("bolt_size", list(sizes_dict.keys()))
    bolt_size = st.selectbox("Bolt Size", options=list(sizes_dict.keys()), index=3, key="bolt_size")

    # Thread series / pitch (same 60° profile; fine/UNF -> larger At, finer control).
    _series_opts = thread_series_options(bolt_size, is_metric)
    _series_labels = [lbl for lbl, _ in _series_opts]
    _validate_choice("thread_series", _series_labels)
    thread_series = st.selectbox(
        "Thread series / pitch", options=_series_labels, key="thread_series",
        help="Same 60° thread profile. A finer pitch (metric fine / UNF / UNEF) gives a larger "
             "tensile-stress area — so a higher proof load — and finer preload-per-turn control "
             "than coarse/UNC, but strips more easily in soft materials.")
    _pitch_lookup = dict(_series_opts)
    selected_pitch_mm = _pitch_lookup.get(thread_series, sizes_dict[bolt_size][1])
    st.caption(f"Selected thread: **{thread_designation(bolt_size, thread_series, is_metric, selected_pitch_mm)}** "
               f"(pitch {selected_pitch_mm:g} mm)")

    bolt_type = st.selectbox("Bolt Type", options=list(BOLT_TYPES.keys()), key="bolt_type")
    _validate_choice("bolt_material", list(bolt_materials_all.keys()))
    bolt_material = st.selectbox("Bolt Material", options=list(bolt_materials_all.keys()),
                                 index=1, key="bolt_material")

    Sp_disp = bolt_materials_all[bolt_material]["Sp"]
    Sut_disp = bolt_materials_all[bolt_material]["Sut"]
    if not is_metric:
        st.caption(f"**Properties:** $S_p$ = {Sp_disp/6.89476:,.0f} ksi, $S_{{ut}}$ = {Sut_disp/6.89476:,.0f} ksi")
    else:
        st.caption(f"**Properties:** $S_p$ = {Sp_disp} MPa, $S_{{ut}}$ = {Sut_disp} MPa")

    st.markdown("---")
    st.header("Additional Factors")
    use_washer = st.checkbox("Washers Included?", value=True, key="use_washer",
                             help="Hardened washers spread the bearing load and are usually "
                                  "required under high-grade bolts to avoid crushing.")
    is_permanent = st.checkbox("Permanent Joint?", value=False, key="is_permanent",
                               help="Uses 90% proof load instead of 75%.")
    _validate_choice("friction_condition", list(FRICTION_COEFFICIENTS.keys()))
    friction_condition = st.selectbox(
        "Nut Factor $K$ ($T = K F d$)",
        options=list(FRICTION_COEFFICIENTS.keys()),
        index=0, key="friction_condition",
        help="Torque coefficient bundling thread + head friction and geometry."
    )
    _validate_choice("tightening_method", list(TIGHTENING_METHODS.keys()))
    tightening_method = st.selectbox(
        "Tightening Method (preload scatter)",
        options=list(TIGHTENING_METHODS.keys()),
        index=0, key="tightening_method",
        help="Sets the expected ± scatter of the achieved preload about the target."
    )
    scatter = TIGHTENING_METHODS[tightening_method]
    embedment_um = st.number_input(
        "Embedment / relaxation fz (µm)", min_value=0.0, value=0.0, step=1.0, key="embedment_um",
        help="VDI 2230 short-term preload loss from surface flattening at the contact "
             "interfaces. Typical guideline: ~2–3 µm per loaded interface (head, thread, "
             "each joint face). Deducted from the operating preload.")
    load_intro_factor = st.number_input(
        "Load-introduction factor n (VDI 2230)", min_value=0.0, max_value=1.0, value=1.0, step=0.1,
        key="load_intro_factor",
        help="Fraction of the grip through which the external load is introduced (VDI 2230). The bolt "
             "sees n·C of the external load; n = 1 applies the load at the joint faces (Shigley). "
             "A lower n (~0.5, load introduced within the members) reduces the bolt's share — better "
             "for fatigue — but relieves the members more, lowering the separation load.")

    st.markdown("---")
    st.header("Acceptance Criteria")
    required_fos = st.number_input(
        "Required factor of safety", min_value=1.0, value=1.5, step=0.05, key="required_fos",
        help="Acceptance threshold for every check. A factor of safety is green at or above this "
             "value, amber with positive margin but below it, and red below 1.0 (actual failure). "
             "It drives the colour of the safety cards and the PASS / MARGINAL / FAIL verdict.")

    st.markdown("---")
    st.header("Bolt Group")
    use_group = st.checkbox(
        "Eccentric bolt-group analysis", value=False, key="use_group",
        help="Distribute joint loads over a bolt pattern (defined in the Bolt Group "
             "tab) and analyse the governing bolt. Overrides the simple external "
             "load + bolt count in the Analysis tab.")

# --- Display-unit factors (internal metric -> display) and labels ---
out_force_factor = 1.0 if is_metric else 1.0 / 4.44822
out_torque_factor = 1.0 if is_metric else 0.73756        # N·m -> ft·lbf
out_length_factor = 1.0 if is_metric else 1.0 / 25.4
out_stiffness_factor = 1.0 if is_metric else (1.0 / 4.44822) / (1.0 / 25.4)   # N/mm -> lbf/in
out_area_factor = 1.0 if is_metric else 1.0 / 645.16     # mm² -> in²
out_stress_factor = 1.0 if is_metric else 1.0 / 6.89476  # MPa -> ksi
force_unit = "N" if is_metric else "lbf"
torque_unit = "N·m" if is_metric else "ft-lbf"
len_unit = "mm" if is_metric else "in"
stiff_unit = "N/mm" if is_metric else "lbf/in"
area_unit = "mm²" if is_metric else "in²"
moment_unit = "N·m" if is_metric else "lbf·ft"
stress_unit = "MPa" if is_metric else "ksi"
units = DisplayUnits(out_force_factor, out_length_factor, out_stress_factor,
                     force_unit, len_unit, stress_unit)

# --- Main Layout ---
# The main workflow (joint composition, thermal, fatigue, thread stripping and the
# fastener tools) is consolidated into one "Analysis" tab; Bolt Group, FE Import
# and Custom Materials keep their own tabs. tab1 = Analysis holds the merged inputs;
# the main results render below the tab bar (top level) as before.
tab1, tab4, tab7, tab5 = st.tabs(
    ["⚙️ Analysis", "🧩 Bolt Group", "📥 FE Import", "🧪 Custom Materials"])

with tab1:
    # ---- Analysis scope: switch on only the checks this joint needs -----------
    st.markdown("##### Analysis scope")
    _sc1, _sc2, _sc3 = st.columns(3)
    use_thermal = _sc1.toggle(
        "🌡️ Thermal", value=False, key="use_thermal",
        help="Include differential thermal expansion between the bolt and the clamped members.")
    use_fatigue = _sc2.toggle(
        "🔄 Fatigue", value=False, key="use_fatigue",
        help="Include the cyclic-load fatigue factor of safety (and the Haigh / criteria comparison).")
    use_thread = _sc3.toggle(
        "🔩 Thread stripping", value=False, key="use_thread",
        help="Include the internal-thread (tapped-hole) stripping check.")
    st.caption("Eccentric **bolt-group** analysis has its own tab (toggle it from the sidebar).")
    st.markdown("---")

    st.subheader("Clamped Joint Layers")
    st.markdown(f"Specify the materials and thicknesses of the clamped joint. Units are in "
                f"**{'mm' if is_metric else 'inches'}**.")

    if "layers" not in st.session_state:
        st.session_state.layers = [
            {"Material": "Steel (Mild)", "Thickness": 10.0 if is_metric else 0.5},
            {"Material": "Steel (Mild)", "Thickness": 10.0 if is_metric else 0.5}
        ]

    edited_layers = st.data_editor(
        st.session_state.layers,
        num_rows="dynamic",
        column_config={
            "Material": st.column_config.SelectboxColumn(
                "Material", options=list(joint_materials_all.keys()), required=True),
            "Thickness": st.column_config.NumberColumn(
                f"Thickness ({len_unit})", min_value=0.001, required=True)
        },
    )
    st.session_state.layers = edited_layers
    st.caption(f"ℹ️ Thicknesses are in **{len_unit}**; switching units converts the values "
               "automatically.")

    # Properties (and sources) of the materials currently used in the stack.
    used_materials: List[str] = []
    for layer_row in edited_layers:
        mk = layer_row.get("Material")
        if mk in joint_materials_all and mk not in used_materials:
            used_materials.append(mk)
    if used_materials:
        syc_f = 1.0 if is_metric else 1.0 / 6.89476
        e_f = 1.0 / 1000.0 if is_metric else 1.0 / 6894.76
        cte_f = 1e6 if is_metric else 1e6 * 5.0 / 9.0
        syc_u = "MPa" if is_metric else "ksi"
        e_u = "GPa" if is_metric else "Msi"
        cte_u = "µm/m·°C" if is_metric else "µin/in·°F"
        st.markdown("**Material properties of the clamped layers**")
        st.dataframe([{
            "Material": mk,
            f"Comp. yield Syc ({syc_u})": round(joint_materials_all[mk]["Syc"] * syc_f, 1),
            f"Modulus E ({e_u})": round(joint_materials_all[mk]["E"] * e_f, 1),
            f"CTE ({cte_u})": round(joint_materials_all[mk]["CTE"] * cte_f, 2),
            "Source": joint_materials_all[mk].get("source", "user-defined"),
        } for mk in used_materials], hide_index=True)
        b_E = bolt_materials_all[bolt_material]["E"] * e_f
        b_cte = bolt_materials_all[bolt_material]["CTE"] * cte_f
        st.caption(f"For thermal comparison, the bolt (**{bolt_material}**) has $E$ = {b_E:,.1f} {e_u} "
                   f"and CTE = {b_cte:,.2f} {cte_u}. Temperature changes preload only when a layer's CTE "
                   f"differs from the bolt's (see §10 of the equations panel).")
        with st.expander("📚 Material property sources & notes"):
            st.markdown(r"""
                Values are **nominal, room-temperature** properties for the listed alloy/temper,
                consistent with standard references. Treat them as representative — always confirm
                against your material certificate, especially yield strength and properties at temperature.

                - **Young's modulus $E$ and CTE $\alpha$:** *Shigley's Mechanical Engineering Design*,
                  Table A-5 (physical constants of materials); *The Engineering ToolBox*; ASM data.
                - **Compressive yield $S_{yc}$:** the typical tensile yield of the listed temper for
                  ductile metals ($S_{yc} \approx S_y$), per *MatWeb* / *ASM Metals Reference Book*.
                  For grey **cast iron** it is the (higher) compressive strength; for **nylon**, the
                  compressive yield of the unfilled polymer.
                - **Bolt grades:** proof $S_p$ and ultimate $S_{ut}$ per **ISO 898-1** (metric) /
                  **SAE J429** (imperial); endurance $S_e$ for rolled threads per *Shigley* Table 8-17.
                - Usage: $S_{yc}$ → bearing/crushing (§6); $E$ → member stiffness (§8); $\alpha$ →
                  thermal effects (§10).
                """)

    # ---- External load (drives separation, proof, and -- if on -- fatigue) ---
    with st.expander("⬇️ External load", expanded=True):
        if use_group:
            st.info("🧩 Bolt-group mode is **ON** — these inputs are ignored. The governing "
                    "per-bolt load is taken from the **Bolt Group** tab.")
        st.markdown("Cyclic/static external load on the joint. If several identical bolts share the "
                    "load, set the bolt count for the per-bolt result.")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            ext_max = st.number_input(f"Max External Load ({force_unit})", value=0.0, key="ext_max")
        with f_col2:
            ext_min = st.number_input(f"Min External Load ({force_unit})", value=0.0, key="ext_min")
        with f_col3:
            num_bolts = st.number_input("Number of Bolts", min_value=1, value=1, step=1,
                                        key="num_bolts",
                                        help="External load is divided equally among this many bolts.")

    # ---- Thermal (optional) --------------------------------------------------
    if use_thermal:
        with st.expander("🌡️ Thermal expansion", expanded=True):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                temp_assembly = st.number_input(
                    f"Assembly Temperature ({'°C' if is_metric else '°F'})",
                    value=20.0 if is_metric else 68.0, key="temp_assembly")
            with t_col2:
                temp_operating = st.number_input(
                    f"Operating Temperature ({'°C' if is_metric else '°F'})",
                    value=20.0 if is_metric else 68.0, key="temp_operating")
    else:
        # Disabled -> equal temperatures, so there is no thermal preload change.
        temp_assembly = 20.0 if is_metric else 68.0
        temp_operating = temp_assembly

    # ---- Fatigue (optional) --------------------------------------------------
    if use_fatigue:
        with st.expander("🔄 Fatigue", expanded=True):
            fatigue_criterion = st.selectbox(
                "Fatigue criterion", options=list(FATIGUE_CRITERIA), index=0, key="fatigue_criterion",
                help="Goodman/Gerber/ASME-elliptic/Soderberg/SWT/Morrow are mean-stress diagrams "
                     "evaluated along the preloaded-bolt load line (Shigley §8-12); conservatism "
                     "roughly Soderberg > Goodman > ASME-elliptic ≳ Gerber, with Morrow least "
                     "conservative. VDI 2230 instead checks the amplitude against a diameter-based "
                     "endurance limit (higher when threads are rolled after heat treatment).")
    else:
        fatigue_criterion = "Goodman"   # default for the engine; fatigue results are hidden

    # ---- Thread stripping (optional) -----------------------------------------
    if use_thread:
        with st.expander("🔩 Thread stripping", expanded=True):
            st.markdown("Check whether internal (tapped) threads will strip before the bolt yields.")
            eng_col1, eng_col2 = st.columns(2)
            with eng_col1:
                thread_engagement = st.number_input(
                    f"Thread Engagement Length ({len_unit})", min_value=0.0, value=0.0,
                    key="thread_engagement")
            with eng_col2:
                _internal_opts = ["(None)"] + list(joint_materials_all.keys())
                _validate_choice("internal_thread_mat", _internal_opts)
                internal_thread_mat = st.selectbox(
                    "Internal Thread Material", options=_internal_opts, key="internal_thread_mat")
    else:
        thread_engagement = 0.0
        internal_thread_mat = "(None)"

with tab4:
    st.subheader("Bolt Group / Pattern (eccentric loading)")
    st.markdown("Distribute joint loads over a fastener pattern using the elastic method and "
                "feed the **governing** (most-loaded) bolt into the analysis. Enable it from the "
                "sidebar (**Bolt Group**).")

    pattern_type = st.selectbox("Pattern", ["Rectangular grid", "Bolt circle", "Custom (X, Y table)"],
                                key="pattern_type")
    conv_len = 1.0 if is_metric else 25.4
    group_coords: List[Tuple[float, float]] = []

    if pattern_type == "Rectangular grid":
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            g_rows = st.number_input("Rows", min_value=1, value=2, step=1, key="g_rows")
        with p_c2:
            g_cols = st.number_input("Columns", min_value=1, value=2, step=1, key="g_cols")
        with p_c3:
            g_px = st.number_input(f"Pitch X ({len_unit})", min_value=0.0,
                                   value=50.0 if is_metric else 2.0, key="g_px")
        with p_c4:
            g_py = st.number_input(f"Pitch Y ({len_unit})", min_value=0.0,
                                   value=50.0 if is_metric else 2.0, key="g_py")
        group_coords = rectangular_pattern(int(g_rows), int(g_cols), g_px * conv_len, g_py * conv_len)
    elif pattern_type == "Bolt circle":
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            g_n = st.number_input("Number of bolts", min_value=1, value=6, step=1, key="g_n")
        with p_c2:
            g_bcd = st.number_input(f"Bolt circle diameter ({len_unit})",
                                    min_value=0.0, value=120.0 if is_metric else 5.0, key="g_bcd")
        with p_c3:
            g_start = st.number_input("Start angle (deg)", value=0.0, key="g_start")
        group_coords = circular_pattern(int(g_n), g_bcd * conv_len, g_start)
    else:
        if "group_table" not in st.session_state:
            st.session_state.group_table = [
                {"X": 0.0, "Y": 50.0}, {"X": 0.0, "Y": -50.0},
                {"X": 60.0, "Y": 0.0}, {"X": -60.0, "Y": 0.0},
            ]
        edited_coords = st.data_editor(
            st.session_state.group_table, num_rows="dynamic",
            column_config={
                "X": st.column_config.NumberColumn(f"X ({len_unit})", required=True),
                "Y": st.column_config.NumberColumn(f"Y ({len_unit})", required=True),
            })
        st.session_state.group_table = edited_coords
        for row in edited_coords:
            gx = row.get("X", 0.0)
            gy = row.get("Y", 0.0)
            gx = 0.0 if gx is None else gx
            gy = 0.0 if gy is None else gy
            group_coords.append((gx * conv_len, gy * conv_len))

    st.markdown("##### Applied joint loads (whole pattern)")
    l_c1, l_c2, l_c3 = st.columns(3)
    with l_c1:
        g_axial_max = st.number_input(f"Max axial tension ({force_unit})", value=0.0, key="g_axial_max")
        g_axial_min = st.number_input(f"Min axial tension ({force_unit})", value=0.0, key="g_axial_min")
    with l_c2:
        g_moment_max = st.number_input(f"Max moment ({moment_unit})", value=0.0, key="g_moment_max")
        g_moment_min = st.number_input(f"Min moment ({moment_unit})", value=0.0, key="g_moment_min")
    with l_c3:
        moment_axis_label = st.radio(
            "Bending about", ["X-axis (tension varies with Y)", "Y-axis (tension varies with X)"],
            key="moment_axis_label")
        moment_axis = "x" if moment_axis_label.startswith("X") else "y"

    st.markdown("##### In-plane shear (slip / torsion)")
    s_c1, s_c2, s_c3, s_c4 = st.columns(4)
    with s_c1:
        g_shear = st.number_input(f"Shear force ({force_unit})", min_value=0.0, value=0.0, key="g_shear")
    with s_c2:
        g_ecc = st.number_input(f"Shear eccentricity ({len_unit})", min_value=0.0, value=0.0, key="g_ecc",
                                help="Distance from the pattern centroid to the shear line of action.")
    with s_c3:
        slip_mu = st.number_input("Slip coefficient μ", min_value=0.0, value=0.30, step=0.05, key="slip_mu",
                                  help="Faying-surface slip factor (e.g. ~0.33 clean mill scale).")
    with s_c4:
        slip_ns = st.number_input("Faying surfaces", min_value=1, value=1, step=1, key="slip_ns")

    # Convert pattern loads to metric (N, N·mm)
    moment_to_Nmm = 1000.0 if is_metric else 1.355818 * 1000.0   # N·m or lbf·ft -> N·mm
    force_to_N = 1.0 if is_metric else 4.44822
    g_axial_max_N = g_axial_max * force_to_N
    g_axial_min_N = g_axial_min * force_to_N
    g_moment_max_Nmm = g_moment_max * moment_to_Nmm
    g_moment_min_Nmm = g_moment_min * moment_to_Nmm
    g_shear_N = g_shear * force_to_N
    g_ecc_mm = g_ecc * conv_len

    g_res_max = analyze_bolt_group(group_coords, g_axial_max_N, g_moment_max_Nmm,
                                   moment_axis, g_shear_N, g_ecc_mm) if group_coords else None
    g_res_min = analyze_bolt_group(group_coords, g_axial_min_N, g_moment_min_Nmm,
                                   moment_axis, g_shear_N, g_ecc_mm) if group_coords else None

    if group_coords and g_res_max is not None:
        gv_c1, gv_c2, gv_c3 = st.columns(3)
        with gv_c1:
            st.metric("Bolts in pattern", f"{len(group_coords)}")
        with gv_c2:
            st.metric("Governing bolt tension",
                      f"{g_res_max['governing_tension_N'] * out_force_factor:,.0f} {force_unit}")
        with gv_c3:
            st.metric("Max bolt shear",
                      f"{g_res_max['governing_shear_N'] * out_force_factor:,.0f} {force_unit}")

        if not g_res_max["moment_reactable"] and (g_moment_max != 0.0 or g_moment_min != 0.0):
            st.warning("⚠️ The pattern cannot react the applied moment about this axis (all bolts lie on "
                       "the bending axis). Only the axial share is distributed.")

        ch_bg = alt_bolt_group_chart(
            group_coords, g_res_max["tensions_N"], g_res_max["shear_vectors_N"],
            g_res_max["governing_index"], out_length_factor, out_force_factor, len_unit, force_unit)
        if ch_bg is not None:
            # Center the larger square chart with spacer columns; width="content"
            # keeps Streamlit from stretching the square spec to the container width
            # (which would distort the equal x/y scaling).
            _bg_l, _bg_c, _bg_r = st.columns([1, 2, 1])
            with _bg_c:
                st.altair_chart(ch_bg, width="content")
        st.caption("Marker colour is per-bolt tension (elastic centroidal model); green arrows are the "
                   "per-bolt shear resultant (direct $V/N$ + torsional $T r_i / J$), the ring is the "
                   "governing bolt and the cross is the centroid. Hover for per-bolt values; scroll to "
                   "zoom. The centroidal tension model can be unconservative for prying-dominated joints.")
    elif use_group:
        st.warning("⚠️ Add at least one bolt to the pattern to run the bolt-group analysis.")

with tab5:
    render_materials_tab()

with tab7:
    render_fe_import_tab(bolt_materials_all)

# --- Preparation & Conversion to Metric ---
try:
    d, _coarse_p = sizes_dict[bolt_size]
    bolt_props = bolt_materials_all[bolt_material]

    # Bolt dimensions come from the size table, which stores millimetre
    # equivalents for BOTH unit systems (reference data is kept metric, like the
    # material strengths). They are NOT user-entered, so no conversion is applied
    # here -- only values the user types in display units get converted below. The
    # pitch comes from the thread-series selector (coarse/fine/UNC/UNF/UNEF).
    d_mm = d
    p_mm = selected_pitch_mm
    dw_mm = calculate_bearing_diameter(d_mm, bolt_type, use_washer)
    thread_desig = thread_designation(bolt_size, thread_series, is_metric, p_mm)

    layers_metric: List[Layer] = []
    for row in edited_layers:
        mat_key = row["Material"]
        t_input = row.get("Thickness", 0.0)
        if t_input is None:
            t_input = 0.0

        t_mm = t_input if is_metric else t_input * 25.4
        layers_metric.append({
            "Material": mat_key,
            "thickness": t_mm,
            "Syc": joint_materials_all[mat_key]["Syc"],
            "E": joint_materials_all[mat_key]["E"],
            "CTE": joint_materials_all[mat_key]["CTE"]
        })

    if not layers_metric:
        st.info("Add at least one joint layer to run the analysis.")
        st.stop()

    temp_assembly_C = temp_assembly if is_metric else (temp_assembly - 32) * 5.0/9.0
    temp_operating_C = temp_operating if is_metric else (temp_operating - 32) * 5.0/9.0

    # --- Determine the per-bolt external load driving the analysis ---
    group_active = use_group and len(group_coords) > 0 and g_res_max is not None
    governing_shear_N = 0.0
    if group_active and g_res_max is not None and g_res_min is not None:
        gi = g_res_max["governing_index"]
        ext_max_N = g_res_max["tensions_N"][gi]
        ext_min_N = g_res_min["tensions_N"][gi]
        governing_shear_N = g_res_max["governing_shear_N"]
        n_bolts = len(group_coords)
    else:
        n_bolts = max(1, int(num_bolts))
        ext_max_N = (ext_max if is_metric else ext_max * 4.44822) / n_bolts
        ext_min_N = (ext_min if is_metric else ext_min * 4.44822) / n_bolts

    eng_len_mm = thread_engagement if is_metric else thread_engagement * 25.4
    internal_props = joint_materials_all[internal_thread_mat] if internal_thread_mat != "(None)" else None

    # Calculate
    results = calculate_preload(
        d=d_mm,
        p=p_mm,
        bolt_material_props=bolt_props,
        layers=layers_metric,
        bolt_type=bolt_type,
        use_washer=use_washer,
        is_permanent=is_permanent,
        friction_condition=friction_condition,
        temp_assembly=temp_assembly_C,
        temp_operating=temp_operating_C,
        external_load_max=ext_max_N,
        external_load_min=ext_min_N,
        thread_engagement_length=eng_len_mm,
        internal_thread_material_props=internal_props,
        fatigue_criterion=fatigue_criterion,
        embedment_um=embedment_um,
        load_intro_factor=load_intro_factor
    )

    # Slip-critical check (shear carried by friction from the clamp force)
    slip_fos: Optional[float] = None
    combined_fos: Optional[float] = None
    if group_active and governing_shear_N > 0:
        slip_resistance = slip_mu * slip_ns * results["operating_preload_N"]
        slip_fos = slip_resistance / governing_shear_N
        # Elliptic tension-shear interaction on the governing bolt (bolt body in bearing).
        combined_fos = combined_tension_shear_fos(
            results["max_bolt_force_N"], governing_shear_N, results["tensile_stress_area_mm2"],
            results["proof_strength_MPa"], results["yield_Sy_MPa"])

    st.markdown("---")

    # Collect findings (rendered below the verdict banner) + warnings for the PDF.
    preload_disp = results["recommended_preload_N"]
    preload_hi = preload_disp * (1.0 + scatter)
    ctx = AnalysisContext(
        scatter=scatter, preload_disp=preload_disp, preload_hi=preload_hi,
        required_fos=required_fos, fatigue_criterion=fatigue_criterion,
        has_internal=internal_props is not None, ext_max_N=ext_max_N,
        slip_fos=slip_fos, combined_fos=combined_fos, embedment_um=embedment_um
    )
    findings, report_warnings = collect_findings(results, units, ctx)

    # ---- Design verdict + findings (answer first, evidence second) -----------
    _applicable: List[Tuple[str, float]] = [("Proof", results["proof_fos"])]
    if use_fatigue:
        _applicable.insert(0, ("Fatigue", results["fatigue_fos"]))
    if ext_max_N > 0:
        _applicable.append(("Separation", results["separation_fos"]))
    if group_active and slip_fos is not None:
        _applicable.append(("Slip", slip_fos))
    if group_active and combined_fos is not None:
        _applicable.append(("Combined T+S", combined_fos))
    if internal_props is not None:
        _applicable.append(("Thread shear", results["thread_shear_fos"]))
    _has_err = any(sev == "error" for sev, _ in findings)
    _n_advisories = sum(1 for sev, _ in findings if sev == "warn")
    _status, _gov, _gov_label = _verdict(_applicable, required_fos, _has_err)
    render_verdict_banner(_status, _gov, _gov_label, required_fos, _n_advisories)
    if findings:
        st.markdown("##### Findings")
        for _sev, _msg in findings:
            if _sev == "error":
                st.error("❌ " + _msg)
            else:
                st.warning("⚠️ " + _msg)

    # 2. Main Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Installation Preload", f"{preload_disp * out_force_factor:,.0f} {force_unit}")
    with col2:
        st.metric("Installation Torque", f"{results['torque_Nm'] * out_torque_factor:,.1f} {torque_unit}")
    with col3:
        thermal_dF = results['thermal_delta_F_N'] * out_force_factor
        st.metric("Operating Preload",
                  f"{results['operating_preload_N'] * out_force_factor:,.0f} {force_unit}",
                  delta=(f"{thermal_dF:+,.0f} {force_unit} (thermal)"
                         if abs(results['thermal_delta_F_N']) > 1e-6 else None),
                  help="Preload after differential thermal expansion between the bolt and the clamped "
                       "members. Equal CTE → no change.")
    with col4:
        st.metric("Joint Constant ($C$)", f"{results['joint_constant_C']:.3f}")

    # Preload scatter band
    st.caption(
        f"**Expected preload scatter ({tightening_method}):** "
        f"{preload_disp*(1.0-scatter)*out_force_factor:,.0f} – {preload_hi*out_force_factor:,.0f} {force_unit} "
        f"(target {preload_disp*out_force_factor:,.0f} {force_unit}). Analysis below uses the target preload."
    )
    if group_active:
        st.caption(
            f"🧩 **Bolt-group governing load:** tension "
            f"{ext_max_N*out_force_factor:,.0f} → {ext_min_N*out_force_factor:,.0f} {force_unit} per bolt, "
            f"max shear {governing_shear_N*out_force_factor:,.0f} {force_unit} (pattern of {n_bolts} bolts)."
        )

    if temp_operating != temp_assembly and abs(results['thermal_delta_F_N']) <= 1e-6:
        st.caption("🌡️ **No thermal preload change:** the bolt and all clamped layers share the same "
                   "coefficient of thermal expansion (CTE), so they expand together. A thermal effect "
                   "appears only with dissimilar materials (e.g. aluminium members with a steel bolt). "
                   "The model assumes a **uniform joint temperature** — it does not use heat-transfer "
                   "coefficients or model transient warm-up or through-thickness gradients.")

    if embedment_um > 0:
        st.caption(
            f"🔧 **Embedment/relaxation loss** ≈ {results['embedment_loss_N']*out_force_factor:,.0f} "
            f"{force_unit} (fz = {embedment_um:.0f} µm), already deducted from the operating preload.")

    st.caption(
        f"🔩 **Tightening stress (von Mises, axial + thread torsion):** "
        f"{results['tightening_stress_MPa']*out_stress_factor:,.0f} {stress_unit} = "
        f"{results['tightening_utilization']*100:.0f}% of yield (assembly guideline ≤ 90%). "
        f"Torsion largely relaxes after the wrench is removed."
    )

    st.markdown("---")

    # 3. Safety-Factor Metrics
    st.markdown("##### Safety check")
    sec_col1, sec_col2, sec_col3, sec_col4 = st.columns(4)
    with sec_col1:
        if use_fatigue:
            st.markdown(_stat_card(f"Fatigue · {fatigue_criterion}", fos_str(results['fatigue_fos']),
                                   _fos_status(results['fatigue_fos'], required_fos)),
                        unsafe_allow_html=True)
        else:
            st.markdown(_stat_card("Fatigue", "off", "na", "check disabled"), unsafe_allow_html=True)
    with sec_col2:
        st.markdown(_stat_card("Proof (static)", fos_str(results['proof_fos']),
                               _fos_status(results['proof_fos'], required_fos)),
                    unsafe_allow_html=True)
    with sec_col3:
        if ext_max_N > 0:
            st.markdown(_stat_card("Separation", fos_str(results['separation_fos']),
                                   _fos_status(results['separation_fos'], required_fos)),
                        unsafe_allow_html=True)
        else:
            st.markdown(_stat_card("Separation", "N/A", "na", "no external load"),
                        unsafe_allow_html=True)
    with sec_col4:
        if group_active:
            if slip_fos is not None:
                st.markdown(_stat_card("Slip (friction)", fos_str(slip_fos),
                                       _fos_status(slip_fos, required_fos)), unsafe_allow_html=True)
            else:
                st.markdown(_stat_card("Slip (friction)", "N/A", "na", "no shear load"),
                            unsafe_allow_html=True)
        elif internal_props is not None:
            st.markdown(_stat_card("Thread shear", fos_str(results['thread_shear_fos']),
                                   _fos_status(results['thread_shear_fos'], required_fos)),
                        unsafe_allow_html=True)
        else:
            st.markdown(_stat_card("Thread shear", "N/A", "na", "no internal thread"),
                        unsafe_allow_html=True)

    if group_active and internal_props:
        st.caption(f"Thread Shear FOS: {fos_str(results['thread_shear_fos'])} "
                   f"(min recommended engagement ≥ "
                   f"{results['required_engagement_mm']*out_length_factor:,.2f} {len_unit}).")

    st.markdown("---")

    # 4. Stiffness & geometry Metrics
    geo_col1, geo_col2, geo_col3, geo_col4 = st.columns(4)
    with geo_col1:
        st.metric("Bolt Stiffness ($k_b$)", f"{results['kb_N_mm'] * out_stiffness_factor:,.0f} {stiff_unit}")
    with geo_col2:
        st.metric("Joint Stiffness ($k_m$)", f"{results['km_N_mm'] * out_stiffness_factor:,.0f} {stiff_unit}")
    with geo_col3:
        st.metric("Stress Area ($A_t$)",
                  f"{results['tensile_stress_area_mm2'] * out_area_factor:,.3f} {area_unit}")
    with geo_col4:
        st.metric("Grip Length ($L$)", f"{results['total_grip_length_mm'] * out_length_factor:,.2f} {len_unit}")

    st.markdown("---")

    # 5. Joint Visualizations (in display units)
    st.subheader("Joint Visualizations")
    viz_c1, viz_c2 = st.columns([1, 1])
    with viz_c1:
        st.markdown("**Joint cross-section & compression cone**")
        fig_cs = make_cross_section_diagram(layers_metric, d_mm, dw_mm, use_washer,
                                            out_length_factor, len_unit, dark=True)
        if fig_cs is not None:
            st.pyplot(fig_cs)
        else:
            st.info("Add a layer with thickness to draw the cross-section.")
    with viz_c2:
        st.markdown("**Joint diagram (force vs. deflection)**")
        ch_jd = alt_joint_diagram(
            results['kb_N_mm'], results['km_N_mm'], results['operating_preload_N'],
            ext_max_N, results['joint_constant_C'],
            out_force_factor, out_length_factor, force_unit, len_unit)
        if ch_jd is not None:
            st.altair_chart(ch_jd)
            st.caption("Blue = bolt stiffness, red = member stiffness; dashed grey = preload. "
                       "Interactive: hover to read values, scroll to zoom.")
        else:
            st.info("Joint diagram not available for infinite or zero stiffness.")

    bf_c1, bf_c2 = st.columns([1, 1])
    with bf_c1:
        st.markdown("**Bolt & member force vs. external load**")
        ch_bf = alt_bolt_force_chart(
            results['operating_preload_N'], results['joint_constant_C'],
            results['separation_load_N'], ext_max_N, out_force_factor, force_unit)
        if ch_bf is not None:
            st.altair_chart(ch_bf)
            st.caption("Green dashed = separation load (members go slack, bolt then takes the full "
                       "load); grey dashed = current max external load.")
        else:
            st.info("Force-sharing chart needs finite stiffness and preload.")
    with bf_c2:
        st.markdown("**Clamp-load budget**")
        _budget = clamp_load_budget(
            results['recommended_preload_N'], results['embedment_loss_N'],
            results['thermal_delta_F_N'], results['joint_constant_C'], ext_max_N)
        st.altair_chart(alt_clamp_waterfall(_budget, out_force_factor, force_unit))
        st.caption("Installation preload stepped by embedment, thermal change and external-load relief "
                   "down to the residual clamp on the members.")

    st.markdown("**External-load sharing (joint constant $C$)**")
    st.altair_chart(alt_load_sharing(results['joint_constant_C'], ext_max_N,
                                     out_force_factor, force_unit))
    if ext_max_N > 0:
        st.caption(f"Of the {ext_max_N*out_force_factor:,.0f} {force_unit} external load, the bolt sees "
                   f"$C\\,P$ = {results['joint_constant_C']*ext_max_N*out_force_factor:,.0f} {force_unit}; "
                   "the remainder relieves the members.")
    else:
        st.caption("No external load entered — the split shows the fractions $C$ and $1-C$ only.")

    # 5b. Fatigue criteria comparison + Haigh diagram (only when fatigue is enabled)
    if use_fatigue:
        with st.expander("📊 Fatigue criteria comparison & Haigh diagram", expanded=False):
            all_fos = results["fatigue_all_fos"]
            comp_rows = [{"Criterion": c, "Fatigue FOS": fos_str(all_fos[c])} for c in all_fos]
            st.markdown("**Factor of safety by criterion** (same operating point, current load):")
            st.dataframe(comp_rows, hide_index=True)
            st.caption(
                f"Operating point: σ_a = {results['fatigue_sigma_a_MPa']*out_stress_factor:,.1f}, "
                f"σ_m = {results['fatigue_sigma_m_MPa']*out_stress_factor:,.1f}, "
                f"preload σ_i = {results['preload_stress_MPa']*out_stress_factor:,.1f} {stress_unit}. "
                f"The selected criterion (**{fatigue_criterion}**) drives the result above.")
            fig_h = make_haigh_diagram(results, out_stress_factor, stress_unit, dark=True)
            if fig_h is not None:
                st.pyplot(fig_h)
                st.caption("Each curve is a failure locus; the black dot is the operating point and "
                           "the grey line is the load line from the preload point. A point below/left "
                           "of a locus is safe by that criterion.")

    # 5c. Fastener Tools (reuses the computed stiffness/preload of this analysis).
    # Rendered inline in the results flow; it has its own sub-expanders, so it
    # cannot itself be an expander (Streamlit forbids nesting).
    st.markdown("---")
    with st.container():
        st.subheader("🛠️ Fastener Tools")
        st.caption("Quick calculators that reuse the current joint, bolt and stiffness results. "
                   "See §16–§20 of the equations panel for the underlying models.")
        K_nut = FRICTION_COEFFICIENTS.get(friction_condition, 0.20)
        to_mm = 1.0 / out_length_factor if out_length_factor else 1.0

        # 1) Torque -> preload (reverse of T = K F d)
        with st.expander("🔁 Torque → preload (reverse)", expanded=True):
            st.markdown(f"Estimate the achieved preload from an applied torque using the current "
                        f"nut factor **K = {K_nut:.2f}** ({friction_condition}).")
            default_T = results['torque_Nm'] * out_torque_factor
            t_in = st.number_input(f"Applied torque ({torque_unit})", min_value=0.0,
                                   value=float(round(default_T, 1)), key="tool_torque")
            torque_Nm_in = t_in / out_torque_factor if out_torque_factor else 0.0
            F_est = preload_from_torque(torque_Nm_in, K_nut, d_mm)
            pct_proof = F_est / results['proof_load_N'] * 100.0 if results['proof_load_N'] > 0 else 0.0
            c1, c2 = st.columns(2)
            c1.metric("Estimated preload", f"{F_est * out_force_factor:,.0f} {force_unit}")
            c2.metric("As % of proof load", f"{pct_proof:.0f}%")
            st.caption(f"Scatter band ({tightening_method}): "
                       f"{F_est*(1-scatter)*out_force_factor:,.0f} – "
                       f"{F_est*(1+scatter)*out_force_factor:,.0f} {force_unit}. Inverts $T = K F d$ (§16).")

        # 2) Target Yield Preload
        with st.expander("💪 Target Yield Preload"):
            st.markdown("Determine the required preload to reach a desired percentage of the bolt's yield strength.")
            yield_pct = st.number_input("Target Yield (%)", min_value=1.0, max_value=150.0,
                                        value=75.0, key="tool_yield_pct")
            Sy_MPa = bolt_materials_all[bolt_material]["Sy"]
            At = results['tensile_stress_area_mm2']
            target_yield_N = preload_from_yield_percent(yield_pct, Sy_MPa, At)

            c1, c2 = st.columns(2)
            c1.metric("Target Preload", f"{target_yield_N * out_force_factor:,.0f} {force_unit}")

            # Estimate torque using the standard short-form formula T = K F d
            torque_req = target_yield_N * K_nut * (d_mm / 1000.0)
            c2.metric("Required Torque", f"{torque_req * out_torque_factor:,.1f} {torque_unit}")

            st.caption(f"Based on a bolt yield strength of {Sy_MPa:,.0f} MPa and tensile stress area of {At:,.1f} mm².")

        # 3) Exact Torque-Tension Relationship
        with st.expander("⚙️ Exact Torque-Tension Relationship"):
            st.markdown("Calculate the exact tightening torque using Shigley's power-screw formulas (Eq. 8-1/8-2), "
                        "accounting for pitch diameter and independent friction coefficients.")
            
            c1, c2, c3 = st.columns(3)
            ft = c1.number_input("Thread Friction (μt)", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
            fc = c2.number_input("Collar Friction (μc)", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
            
            # Default collar diameter: (clearance + hex_af) / 2
            hw = bolt_hardware_reference(bolt_size, d_mm, p_mm)
            d_clear = hw.get("clearance", d_mm * 1.1)
            d_hex = hw.get("hex_af", d_mm * 1.5)
            dc_default = float(round((d_clear + d_hex) / 2.0, 2))
            
            # Convert to display units
            dc_default_disp = float(round(dc_default * out_length_factor, 2))
            
            dc_in = c3.number_input(f"Collar Dia. dc ({len_unit})", min_value=float(round(d_mm * out_length_factor, 2)), 
                                    value=dc_default_disp, step=0.1, key="tool_exact_dc")
            dc_mm = dc_in / out_length_factor if out_length_factor else dc_in
            
            target_F_disp = st.number_input("Target Preload", min_value=0.0, 
                                            value=float(results['recommended_preload_N']) * out_force_factor, 
                                            step=1000.0, key="tool_exact_targetF")
            target_F_N = target_F_disp / out_force_factor if out_force_factor else target_F_disp
            
            total_Nm, thread_Nm, collar_Nm, equiv_K = exact_tightening_torque(target_F_N, d_mm, p_mm, ft, fc, dc_mm)
            
            st.markdown("#### Results")
            r1, r2, r3 = st.columns(3)
            r1.metric("Total Torque", f"{total_Nm * out_torque_factor:,.1f} {torque_unit}")
            r2.metric("Thread Torque", f"{thread_Nm * out_torque_factor:,.1f} {torque_unit}")
            r3.metric("Collar Torque", f"{collar_Nm * out_torque_factor:,.1f} {torque_unit}")
            
            st.caption(f"Equivalent Nut Factor **K = {equiv_K:.3f}** (compare against the simplified K = {K_nut:.2f})")

        # 4) Angle (turn-of-nut) control
        with st.expander("📐 Angle control (turn-of-nut)"):
            st.markdown("Rotation beyond snug to reach the target preload, from the joint's elastic "
                        "stiffness (§17).")
            snug_pct = st.slider("Snug preload (% of target)", 0, 50, 10, key="tool_snug")
            target_F = results['recommended_preload_N']
            snug_F = target_F * snug_pct / 100.0
            theta = tightening_angle(target_F, results['kb_N_mm'], results['km_N_mm'], p_mm, snug_F)
            c1, c2 = st.columns(2)
            c1.metric("Rotation past snug", f"{theta:,.0f}°")
            c2.metric("≈ turns", f"{theta/360.0:,.2f}")
            st.caption("Elastic rotation only; run-down and embedment add to the practical angle. "
                       "Angle control sidesteps friction (torque) scatter, so it is more repeatable.")

        # 3) Bolt length & thread engagement
        with st.expander("📏 Bolt length & thread-in-grip"):
            grip_mm = results['total_grip_length_mm']
            st.markdown(f"Grip (clamped thickness) = **{grip_mm*out_length_factor:,.2f} {len_unit}**. "
                        "Add the stack consumed beyond the grip to size the bolt (§18).")
            d1, d2, d3 = st.columns(3)
            with d1:
                nut_h = st.number_input(f"Nut height ({len_unit})", min_value=0.0,
                                        value=float(round(0.8 * d_mm * out_length_factor, 2)),
                                        key="tool_nut_h")
            with d2:
                wash_t = st.number_input(f"Washer thickness ({len_unit})", min_value=0.0,
                                         value=float(round(0.15 * d_mm * out_length_factor, 2)),
                                         key="tool_wash_t")
                n_wash = st.number_input("Washers", min_value=0, value=1 if use_washer else 0,
                                         step=1, key="tool_n_wash")
            with d3:
                protr = st.number_input(f"Thread protrusion ({len_unit})", min_value=0.0,
                                        value=float(round(2.0 * p_mm * out_length_factor, 2)),
                                        key="tool_protr")
            extra_stack_mm = (nut_h + n_wash * wash_t + protr) * to_mm
            lengths = STANDARD_BOLT_LENGTHS_METRIC_MM if is_metric else STANDARD_BOLT_LENGTHS_IMPERIAL_MM
            l_min_mm, rec_mm = recommend_bolt_length(grip_mm, extra_stack_mm, lengths)
            c1, c2 = st.columns(2)
            c1.metric("Minimum length", f"{l_min_mm*out_length_factor:,.2f} {len_unit}")
            c2.metric("Recommended standard",
                      f"{rec_mm*out_length_factor:,.2f} {len_unit}" if rec_mm else "—")
            if rec_mm:
                eng = grip_thread_engagement(d_mm, rec_mm, grip_mm, metric=is_metric)
                shank_disp = eng['shank_length_mm'] * out_length_factor
                if eng["threads_in_grip"]:
                    st.warning(f"⚠️ Threads fall within the grip (unthreaded shank {shank_disp:,.1f} "
                               f"{len_unit} < grip). This lowers bolt stiffness; use a longer bolt or a "
                               "reduced thread length if a full shank in the grip is required.")
                else:
                    st.success(f"✓ Full unthreaded shank spans the grip (shank {shank_disp:,.1f} {len_unit}).")
            else:
                st.info("No standard length in the series is long enough — use a longer custom bolt.")

        # 4) Min-size / grade selector
        with st.expander("🔎 Size / grade selector"):
            st.markdown("Find the smallest bolt (by stress area) that meets your factor-of-safety "
                        "targets for the **current joint and per-bolt loads** (§19). Every available "
                        "**pitch** of each size (coarse and fine / UNF) is evaluated.")
            s1, s2, s3 = st.columns(3)
            with s1:
                tgt_proof = st.number_input("Target proof FOS", min_value=1.0, value=1.5,
                                            step=0.1, key="tool_tgt_proof")
            with s2:
                tgt_fat = st.number_input("Target fatigue FOS", min_value=1.0, value=1.5,
                                          step=0.1, key="tool_tgt_fat")
            with s3:
                tgt_sep = st.number_input("Target separation FOS", min_value=1.0, value=1.1,
                                          step=0.1, key="tool_tgt_sep")
            search_all = st.checkbox("Search all grades (else current grade only)", value=True,
                                     key="tool_search_all")
            if st.checkbox("Run selector", value=False, key="tool_run_solver"):
                mats = bolt_materials_all if search_all else {bolt_material: bolt_props}
                rec = recommend_bolt(
                    sizes_dict, mats, layers_metric,
                    bolt_type=bolt_type, use_washer=use_washer, is_permanent=is_permanent,
                    friction_condition=friction_condition, temp_assembly=temp_assembly_C,
                    temp_operating=temp_operating_C, external_load_max=ext_max_N,
                    external_load_min=ext_min_N, fatigue_criterion=fatigue_criterion,
                    target_proof_fos=tgt_proof, target_fatigue_fos=tgt_fat,
                    target_separation_fos=tgt_sep,
                    thread_series=BOLT_THREAD_SERIES_METRIC if is_metric else BOLT_THREAD_SERIES_IMPERIAL)
                if rec["found"] and rec["best"] is not None:
                    best = rec["best"]
                    st.success(
                        f"Smallest adequate: **{best['size']} / {best['material']}** "
                        f"(pitch {best['pitch_mm']:g} mm, "
                        f"At = {best['stress_area_mm2']*out_area_factor:,.3f} {area_unit}, "
                        f"preload {best['preload_N']*out_force_factor:,.0f} {force_unit}, "
                        f"torque {best['torque_Nm']*out_torque_factor:,.1f} {torque_unit}).")
                    st.dataframe([{
                        "Size": c["size"], "Thread": c["thread"], "Grade": c["material"],
                        f"At ({area_unit})": round(c["stress_area_mm2"] * out_area_factor, 3),
                        "Proof FOS": round(c["proof_fos"], 2),
                        "Fatigue FOS": (None if c["fatigue_fos"] == float('inf')
                                        else round(c["fatigue_fos"], 2)),
                        "Sep. FOS": (None if c["separation_fos"] == float('inf')
                                     else round(c["separation_fos"], 2)),
                    } for c in rec["candidates"][:12]], hide_index=True, width="stretch")
                else:
                    st.error("No size/grade combination in the database meets all targets for these "
                             "loads. Relax the targets, broaden the grade search, or reduce the load.")

        # 5) Reference dimensions
        with st.expander("🧰 Reference dimensions"):
            hw = bolt_hardware_reference(bolt_size, d_mm, p_mm)
            st.markdown(f"Typical wrench/hole dimensions for **{thread_desig}** (§20). "
                        "Tap drill follows the selected pitch (d − p).")

            def _disp(x: Optional[float]) -> str:
                return f"{x*out_length_factor:,.2f} {len_unit}" if x is not None else "—"

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Hex across-flats", _disp(hw["hex_af_mm"]))
            h2.metric("Hex-key (socket)", _disp(hw["socket_af_mm"]))
            h3.metric("Clearance hole", _disp(hw["clearance_hole_mm"]))
            h4.metric("Tap drill (d−p)", _disp(hw["tap_drill_mm"]))
            st.caption("Wrench/key sizes are standard tool sizes; clearance is a typical free fit; "
                       "tap drill uses the nominal-minus-pitch rule. Verify against the relevant "
                       "standard for critical work.")

    # 6. Export
    def to_export_value(v: Any) -> Any:
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            return None
        return v

    export_results = {
        "preload": round(results["recommended_preload_N"] * out_force_factor, 2),
        "operating_preload": round(results["operating_preload_N"] * out_force_factor, 2),
        "torque": round(results["torque_Nm"] * out_torque_factor, 3),
        "joint_constant_C": round(results["joint_constant_C"], 4),
        "kb": round(results["kb_N_mm"] * out_stiffness_factor, 1),
        "km": round(results["km_N_mm"] * out_stiffness_factor, 1),
        "fatigue_fos": (None if results["fatigue_fos"] == float('inf')
                        else round(results["fatigue_fos"], 3)),
        "proof_fos": to_export_value(results["proof_fos"]),
        "separation_fos": to_export_value(results["separation_fos"]) if ext_max_N > 0 else None,
        "thread_shear_fos": to_export_value(results["thread_shear_fos"]) if internal_props else None,
        "slip_fos": to_export_value(slip_fos) if slip_fos is not None else None,
        "force_unit": force_unit,
        "torque_unit": torque_unit,
        "stiffness_unit": stiff_unit,
    }
    export_inputs: dict = {
        "unit_system": "Metric" if is_metric else "Imperial",
        "bolt": thread_desig,
        "bolt_material": bolt_material,
        "bolt_type": bolt_type,
        "num_bolts": n_bolts,
        "layers": edited_layers,
        "tightening_method": tightening_method,
    }
    if group_active and g_res_max is not None:
        export_inputs["bolt_group"] = {
            "pattern": pattern_type,
            "governing_tension": round(ext_max_N * out_force_factor, 2),
            "governing_shear": round(governing_shear_N * out_force_factor, 2),
        }
    export_data = {"inputs": export_inputs, "results": export_results}

    # Build the report rows / figures shared by Markdown and PDF.
    _gov_txt = "∞" if _gov == float("inf") else f"{_gov:.2f}"
    _verdict_txt = (f"{_status.upper()} — governing FoS {_gov_txt}"
                    + (f" ({_gov_label})" if _gov_label else "")
                    + f"; required {required_fos:.2f}")
    result_rows = [
        ("Design verdict", _verdict_txt),
        ("Installation preload", f"{preload_disp*out_force_factor:,.0f} {force_unit}"),
        ("Installation torque", f"{results['torque_Nm']*out_torque_factor:,.1f} {torque_unit}"),
        ("Operating preload", f"{results['operating_preload_N']*out_force_factor:,.0f} {force_unit}"),
        ("Joint constant C", f"{results['joint_constant_C']:.3f}"),
        (f"Fatigue FOS ({fatigue_criterion})" if use_fatigue else "Fatigue FOS",
         fos_str(results['fatigue_fos']) if use_fatigue else "off (not evaluated)"),
        ("Proof FOS (static)", fos_str(results['proof_fos'])),
        ("Separation FOS", fos_str(results['separation_fos']) if ext_max_N > 0 else "N/A"),
        ("Thread shear FOS", fos_str(results['thread_shear_fos']) if internal_props else "N/A"),
        ("Tightening stress (von Mises)",
         f"{results['tightening_stress_MPa']*out_stress_factor:,.0f} {stress_unit} "
         f"({results['tightening_utilization']*100:.0f}% of yield)"),
        ("Embedment loss", f"{results['embedment_loss_N']*out_force_factor:,.0f} {force_unit}"),
        ("Bolt stiffness kb", f"{results['kb_N_mm']*out_stiffness_factor:,.0f} {stiff_unit}"),
        ("Joint stiffness km", f"{results['km_N_mm']*out_stiffness_factor:,.0f} {stiff_unit}"),
        ("Stress area At", f"{results['tensile_stress_area_mm2']*out_area_factor:,.3f} {area_unit}"),
        ("Grip length L", f"{results['total_grip_length_mm']*out_length_factor:,.2f} {len_unit}"),
    ]
    bolt_group_rows: Optional[List[Tuple[str, str]]] = None
    if group_active and g_res_max is not None:
        bolt_group_rows = [
            ("Pattern", f"{pattern_type}, {n_bolts} bolts"),
            ("Governing bolt tension (max → min)",
             f"{ext_max_N*out_force_factor:,.0f} → {ext_min_N*out_force_factor:,.0f} {force_unit}"),
            ("Max bolt shear", f"{governing_shear_N*out_force_factor:,.0f} {force_unit}"),
            ("Slip FOS", fos_str(slip_fos) if slip_fos is not None else "N/A"),
        ]

    report = (
        f"# Bolt Joint Analysis Report\n\n"
        f"**Bolt:** {thread_desig} {bolt_material} ({bolt_type}) | "
        f"**Units:** {'Metric' if is_metric else 'Imperial'} | **Bolts:** {n_bolts} | "
        f"**Required FoS:** {required_fos:.2f}\n\n"
        f"| Result | Value |\n|---|---|\n"
        + "".join(f"| {k} | {v} |\n" for k, v in result_rows)
    )
    if bolt_group_rows:
        report += ("\n## Bolt Group\n\n| Item | Value |\n|---|---|\n"
                   + "".join(f"| {k} | {v} |\n" for k, v in bolt_group_rows))

    assumptions = [
        "Methodology: Shigley's Mechanical Engineering Design (Ch. 8) and the Rotscher / VDI 2230 "
        "frustum-cone member-stiffness model.",
        f"Acceptance criterion: required factor of safety = {required_fos:.2f}. A check passes at or "
        "above this value, is marginal between 1.0 and it, and fails below 1.0; the design verdict is "
        "the governing (minimum) factor of safety.",
    ]
    if use_fatigue:
        assumptions.append(
            f"Fatigue: {fatigue_criterion} mean-stress criterion with the load line from the preload "
            "point (Shigley Ch. 8-12). Endurance limits are for rolled threads and already include the "
            "thread fatigue stress concentration.")
    assumptions += [
        "Bolt-group tension uses the elastic centroidal model; shear uses the elastic vector method.",
        "All values are shown in the selected display units; the analysis is computed internally in SI.",
    ]

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.download_button("📥 Export JSON", data=json.dumps(export_data, indent=4, default=to_export_value),
                           file_name="bolt_analysis.json", mime="application/json")
    with exp_col2:
        st.download_button("📄 Export Report (Markdown)", data=report,
                           file_name="bolt_analysis.md", mime="text/markdown")
    with exp_col3:
        # The PDF (and its light-themed figures) is built by a cached helper, so it
        # is only regenerated when an input actually changes -- not on every rerun.
        pdf_bytes = generate_pdf_report(
            "Bolt Joint Analysis Report",
            (f"{thread_desig} {bolt_material} ({bolt_type})  |  "
             f"{'Metric' if is_metric else 'Imperial'}  |  Bolts: {n_bolts}  |  "
             f"Required FoS: {required_fos:.2f}  |  v{__version__}"),
            tuple(result_rows),
            tuple(report_warnings),
            tuple(bolt_group_rows) if bolt_group_rows else None,
            tuple(assumptions),
            results, ext_max_N,
            units,
            tuple(group_coords) if group_active and g_res_max is not None else (),
            tuple(g_res_max["tensions_N"]) if group_active and g_res_max is not None else (),
            g_res_max["governing_index"] if group_active and g_res_max is not None else -1,
            tuple((str(r["Material"]), float(r["thickness"])) for r in layers_metric),
            d_mm, dw_mm, use_washer,
            tuple(g_res_max["shear_vectors_N"]) if group_active and g_res_max is not None else (),
        )
        st.download_button("📑 Export PDF", data=pdf_bytes,
                           file_name="bolt_analysis.pdf", mime="application/pdf")

    st.download_button(
        "💾 Save project (JSON)", data=json.dumps(build_project(), indent=2, default=str),
        file_name="bolt_project.json", mime="application/json",
        help="Saves every input (incl. layers, bolt-group and custom materials). Reload it from the "
             "sidebar's Save / load project panel to restore the whole setup.")

    st.markdown("---")
    with st.expander("📘 View Engineering Equations & Assumptions"):
        st.download_button(
            "📑 Download theory manual (PDF)", data=generate_theory_pdf(),
            file_name="bolt_theory_manual.pdf", mime="application/pdf",
            help="The equations and assumptions below as a standalone PDF.")
        for _kind, _content in THEORY_BLOCKS:
            if _kind == "md":
                st.markdown(_content)
            elif _kind == "eq":
                st.latex(_content)
            else:
                st.caption(_content)

except Exception as e:
    # Control-flow signals (st.stop / st.rerun) subclass BaseException, so they are
    # not caught here. Any real error is logged with its traceback and shown in full
    # rather than collapsed to a one-line message, so field issues are diagnosable.
    logger.exception("Bolt analysis failed for the current inputs")
    st.error("❌ The analysis could not be completed for the current inputs. "
             "Check the inputs (or reset via the sidebar).")
    # In production, do not expose the raw stack trace on the UI.
    # st.exception(e)

st.markdown("---")
st.caption(f"Bolt Preload & Joint Analysis · v{__version__} · methodology per Shigley Ch. 8 and "
           "VDI 2230 · values are nominal — verify before design use.")
