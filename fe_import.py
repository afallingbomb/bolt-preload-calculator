"""FE-results import tab: evaluate per-bolt finite-element results from a CSV and
report a factor-of-safety verdict per bolt.

Self-contained Streamlit tab -- it has its own inputs, population graphics and
CSV/PDF report and does not affect the analysis or exports on the other tabs. The
plotting lives in charts.py; this module owns the tab's I/O, layout and report.
"""
import io
import random
from typing import List, Mapping

import pandas as pd
import streamlit as st

from mechanics import (
    BOLT_MATERIALS_METRIC, FATIGUE_CRITERIA, BoltMaterial,
    calculate_stress_area, evaluate_fe_rows,
)
from report import build_fe_report
from charts import (
    _fe_results_df, make_fe_dashboard_figure, fig_to_png,
    fe_chart_histogram, fe_chart_governing, fe_chart_worst, fe_chart_haigh,
    fe_chart_tension_shear, fe_chart_box, fe_chart_ecdf,
)


MAX_FE_ROWS = 50000   # refuse absurdly large uploads so a rerun cannot hang the app


def _fe_sample_csv(n: int = 100) -> str:
    """A representative ``n``-bolt FE-results CSV (SI units) spanning several sizes/
    grades with a realistic spread of factors of safety -- mostly passing, with a
    marginal/failing tail and a mix of governing modes. Deterministic (seeded) so
    the download and the analysis always agree."""
    rng = random.Random(7)
    combos = [
        (8.0, 1.25, "Grade 8.8"), (10.0, 1.5, "Grade 8.8"),
        (12.0, 1.75, "Grade 10.9"), (16.0, 2.0, "Grade 10.9"),
        (20.0, 2.5, "Grade 12.9"),
    ]
    lines = ["bolt_id,diameter_mm,pitch_mm,bolt_grade,preload_N,"
             "axial_force_max_N,axial_force_min_N,shear_force_max_N"]
    for i in range(1, n + 1):
        d, p, grade = combos[i % len(combos)]
        sp = BOLT_MATERIALS_METRIC[grade]["Sp"]
        fp = sp * calculate_stress_area(d, p)              # proof load
        proof_target = rng.uniform(1.25, 3.0)
        if i % 17 == 0:                                    # a few clear overloads
            proof_target = rng.uniform(0.9, 1.3)
        axial_max = fp / proof_target
        preload = axial_max * rng.uniform(0.72, 0.90)
        axial_min = max(0.0, min(axial_max, preload * rng.uniform(0.96, 1.03)))
        shear_max = fp * rng.uniform(0.03, 0.14)
        lines.append(f"B{i:03d},{d:g},{p:g},{grade},{preload:.0f},"
                     f"{axial_max:.0f},{axial_min:.0f},{shear_max:.0f}")
    return "\n".join(lines) + "\n"


def _fos_fmt(v: float) -> str:
    """Factor-of-safety text; an unbounded (no-load) check shows as a dash."""
    return "—" if v == float('inf') else f"{v:.2f}"


@st.cache_data(show_spinner=False, max_entries=4)
def generate_fe_report_pdf(ok_results: List[dict], criterion: str, target: float) -> bytes:
    """Build the FE report PDF (population dashboard + per-bolt table), cached on its
    inputs so the matplotlib panels and reportlab table are not rebuilt every rerun."""
    df = _fe_results_df(ok_results)
    png = fig_to_png(make_fe_dashboard_figure(df, target))
    n_fail = int((~df["passes"]).sum())
    min_fos = float(df["min_fos"].min())
    pdf_cols = ["Bolt", "d", "σa", "σm", "Proof", "Fatigue", "Shear", "Min", "Result"]
    pdf_rows = [[r["bolt_id"], f"{r['diameter_mm']:g}", f"{r['sigma_a_MPa']:.0f}",
                 f"{r['sigma_m_MPa']:.0f}", _fos_fmt(r["proof_fos"]), _fos_fmt(r["fatigue_fos"]),
                 _fos_fmt(r["shear_fos"]), f"{r['min_fos']:.2f}",
                 "PASS" if r["passes"] else "FAIL"] for r in ok_results]
    summary = [
        f"Bolts evaluated: {len(ok_results)}  |  passing: {len(ok_results) - n_fail}  |  failing: {n_fail}",
        f"Target factor of safety: {target:g}  |  fatigue criterion: {criterion}",
        f"Lowest factor of safety: {_fos_fmt(min_fos)}",
    ]
    notes = [
        "Inputs are SI (N, mm, MPa); axial forces are the total bolt tension (incl. preload).",
        "Shear uses the tensile-stress area (threads assumed in the shear plane); combined is an "
        "elliptic tension-shear interaction. The Haigh panel is normalised (Goodman reference line); "
        "pass/fail reflects the selected criterion.",
    ]
    return build_fe_report(title="FE Bolt Results — Factor of Safety",
                           subtitle=f"{len(ok_results)} bolts  |  criterion {criterion}  |  target {target:g}",
                           columns=pdf_cols, rows=pdf_rows, summary=summary, notes=notes, figures=[png])


def render_fe_import_tab(bolt_grades: Mapping[str, BoltMaterial]) -> None:
    """Self-contained tab: evaluate per-bolt FE results from a CSV and report FOS.

    Independent of the rest of the app -- its own inputs, results and report."""
    st.subheader("FE Results Import (CSV)")
    st.markdown(
        "Import per-bolt results from an external finite-element calculation and get a "
        "factor-of-safety verdict per bolt. This tab is **self-contained**: it has its own report "
        "and does **not** affect the analysis or exports on the other tabs.")
    st.info(
        "**Units — SI only.** The CSV is read in SI: forces in **N**, lengths in **mm**, stresses "
        "in **MPa**, tension positive. The results below are shown in SI as well.")
    st.info(
        "**Load convention.** `axial_force_max_N` / `axial_force_min_N` are the **total bolt tension** "
        "over the duty cycle — already including preload and the FE contact load-sharing. The factors "
        "of safety are computed directly from these forces; the joint-stiffness (joint-constant *C*) "
        "model used on the other tabs is **not** re-applied here.")

    with st.expander("📋 Accepted columns"):
        st.markdown(
            "**Required:** `bolt_id`, `diameter_mm`, `pitch_mm`, `axial_force_max_N`, and the bolt "
            "strength — either `bolt_grade` (a built-in grade, e.g. *Grade 10.9*) **or** explicit "
            "`proof_MPa`, `yield_MPa`, `ultimate_MPa`, `endurance_MPa`.\n\n"
            "**Recommended:** `axial_force_min_N` (fatigue range), `preload_N` (fatigue load line), "
            "`shear_force_max_N` (shear + combined checks).\n\n"
            "Common header aliases are accepted (e.g. `d`, `p`, `Sp`, `axial_max`, `shear`). Rows that "
            "share a `bolt_id` are **enveloped** (max tension / min tension / max shear). When "
            "`bolt_grade` is given it overrides the explicit strengths.")

    cset1, cset2, cset3 = st.columns([1.3, 1, 1])
    with cset1:
        st.download_button("⬇️ Download sample CSV", data=_fe_sample_csv(),
                           file_name="fe_bolts_sample.csv", mime="text/csv", key="fe_sample_dl")
        use_sample = st.checkbox("Use the sample data", value=False, key="fe_use_sample")
    with cset2:
        fe_criterion = st.selectbox("Fatigue criterion", list(FATIGUE_CRITERIA), key="fe_criterion")
    with cset3:
        fe_target = st.number_input("Target FOS", min_value=1.0, value=1.5, step=0.1, key="fe_target")

    uploaded = st.file_uploader("Upload FE results (CSV)", type="csv", key="fe_uploader")

    df_fe = None
    if use_sample:
        df_fe = pd.read_csv(io.StringIO(_fe_sample_csv()))
        st.caption("Showing results for the built-in sample data.")
    elif uploaded is not None:
        try:
            df_fe = pd.read_csv(uploaded, nrows=MAX_FE_ROWS + 1)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")

    if df_fe is None:
        st.info("Upload a CSV or tick **Use the sample data** to run the analysis.")
        return

    if len(df_fe) > MAX_FE_ROWS:
        st.error(f"The file has {len(df_fe):,} rows, above the {MAX_FE_ROWS:,}-row limit. "
                 "Reduce it (e.g. envelope the load cases per bolt) and re-upload.")
        return
    rows = df_fe.where(pd.notnull(df_fe), None).to_dict("records")
    results = evaluate_fe_rows(rows, bolt_grades, fe_criterion, fe_target)
    ok = [r for r in results if not r.get("error")]
    bad = [r for r in results if r.get("error")]
    if not results:
        st.warning("No rows with a `bolt_id` were found in the file.")
        return

    n_fail = sum(1 for r in ok if not r["passes"])
    min_fos = min((r["min_fos"] for r in ok), default=float('inf'))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bolts evaluated", f"{len(ok)}")
    m2.metric("Passing", f"{len(ok) - n_fail}")
    m3.metric("Failing", f"{n_fail}")
    m4.metric("Lowest FOS", _fos_fmt(min_fos))
    if n_fail:
        st.error(f"❌ {n_fail} of {len(ok)} bolts are below the target FOS of {fe_target:g}.")
    elif ok:
        st.success(f"✅ All {len(ok)} bolts meet the target FOS of {fe_target:g}.")
    if bad:
        st.warning("Some rows were skipped:\n"
                   + "\n".join(f"- **{r.get('bolt_id', '?')}**: {r['error']}" for r in bad))

    table_rows = [{
        "Bolt": r["bolt_id"], "d (mm)": r["diameter_mm"],
        "σa (MPa)": round(r["sigma_a_MPa"], 1), "σm (MPa)": round(r["sigma_m_MPa"], 1),
        "Proof FOS": _fos_fmt(r["proof_fos"]), "Fatigue FOS": _fos_fmt(r["fatigue_fos"]),
        "Shear FOS": _fos_fmt(r["shear_fos"]), "Combined FOS": _fos_fmt(r["combined_fos"]),
        "Min FOS": round(r["min_fos"], 2), "Governing": r["governing"],
        "Result": "✅ PASS" if r["passes"] else "❌ FAIL",
    } for r in ok]
    if table_rows:
        st.dataframe(table_rows, hide_index=True, width="stretch")

    # --- Population graphics (interactive; live only in this tab) ---
    if ok:
        dfres = _fe_results_df(ok)
        st.markdown("### 📈 Result graphics")
        st.caption("Population view of the bolt fleet — these charts and the dashboard in the FE "
                   "report are exported only by this tab.")
        ga, gb = st.columns(2)
        with ga:
            st.markdown("**Min-FOS distribution**")
            st.altair_chart(fe_chart_histogram(dfres, fe_target))
        with gb:
            st.markdown("**Governing check** (what drives the design)")
            st.altair_chart(fe_chart_governing(dfres))

        st.markdown("**Worst bolts (lowest min FOS)**")
        st.altair_chart(fe_chart_worst(dfres, fe_target))

        gc, gd = st.columns(2)
        with gc:
            st.markdown("**Fatigue — normalised Haigh** (Goodman reference)")
            st.altair_chart(fe_chart_haigh(dfres))
        with gd:
            st.markdown("**Tension–shear interaction** (unit-ellipse limit)")
            st.altair_chart(fe_chart_tension_shear(dfres))

        ge, gf = st.columns(2)
        with ge:
            st.markdown("**FOS by check**")
            st.altair_chart(fe_chart_box(dfres))
        with gf:
            st.markdown("**Cumulative distribution**")
            st.altair_chart(fe_chart_ecdf(dfres, fe_target))

    # --- Separate report: annotated CSV + dedicated PDF (independent of other tabs) ---
    export_rows = [{
        "bolt_id": r["bolt_id"], "diameter_mm": r["diameter_mm"], "pitch_mm": r["pitch_mm"],
        "stress_area_mm2": round(r["stress_area_mm2"], 2),
        "axial_max_N": r["axial_max_N"], "axial_min_N": r["axial_min_N"],
        "shear_max_N": r["shear_max_N"], "preload_N": r["preload_N"],
        "sigma_a_MPa": round(r["sigma_a_MPa"], 2), "sigma_m_MPa": round(r["sigma_m_MPa"], 2),
        "proof_fos": r["proof_fos"], "fatigue_fos": r["fatigue_fos"], "shear_fos": r["shear_fos"],
        "combined_fos": r["combined_fos"], "min_fos": round(r["min_fos"], 3),
        "governing": r["governing"], "passes": r["passes"],
    } for r in ok]
    csv_out = pd.DataFrame(export_rows).to_csv(index=False) if export_rows else ""
    pdf_bytes = generate_fe_report_pdf(ok, fe_criterion, fe_target) if ok else b""

    dl1, dl2 = st.columns(2)
    dl1.download_button("📥 Download results CSV", data=csv_out, file_name="fe_bolt_results.csv",
                        mime="text/csv", key="fe_csv_dl", disabled=not export_rows)
    dl2.download_button("📑 Download FE report (PDF + graphics)", data=pdf_bytes,
                        file_name="fe_bolt_report.pdf", mime="application/pdf",
                        key="fe_pdf_dl", disabled=not ok)
