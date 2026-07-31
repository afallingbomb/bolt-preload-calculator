from __future__ import annotations
"""Pure presentation logic derived from a calculation result.

Factor-of-safety formatting and the design "findings" (the warnings/errors shown
under the verdict banner and copied into the report). No Streamlit dependency, so
this branch-heavy logic is unit-testable and the page module (app.py) only has to
render what these return.
"""
from typing import List, NamedTuple, Optional, Tuple

from mechanics import PreloadResult


class DisplayUnits(NamedTuple):
    """Display-unit conversion factors (internal SI -> display) and their labels.

    Bundled so the functions that format result text take one argument instead of
    six. ``*_factor`` multiplies an SI value to get the display value. Hashable
    (a NamedTuple), so it is safe as a cached-function argument."""
    force_factor: float
    length_factor: float
    stress_factor: float
    force_unit: str
    len_unit: str
    stress_unit: str


def fos_str(value: float) -> str:
    """Factor-of-safety text; an unbounded (no-load) check shows as the infinity sign."""
    return "∞" if value == float('inf') else f"{value:.2f}"


from dataclasses import dataclass

@dataclass
class AnalysisContext:
    scatter: float
    preload_disp: float
    preload_hi: float
    required_fos: float
    fatigue_criterion: str
    has_internal: bool
    ext_max_N: float
    slip_fos: Optional[float]
    combined_fos: Optional[float]
    embedment_um: float
    use_fatigue: bool = True


def collect_findings(
    results: PreloadResult, units: DisplayUnits, ctx: AnalysisContext
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Design findings for the current result.

    Returns ``(findings, report_warnings)``. ``findings`` is a list of
    ``(severity, message)`` pairs (severity in {"warn", "error"}) for on-screen
    rendering; ``report_warnings`` is the same messages with markdown bold stripped,
    for the PDF/markdown report. ``preload_disp``/``preload_hi`` are the target and
    upper-scatter-band preloads (SI N). ``combined_fos`` is the governing bolt's
    tension-shear interaction FoS (None when not applicable). Pure: no Streamlit.
    """
    findings: List[Tuple[str, str]] = []
    report_warnings: List[str] = []

    def add(severity: str, msg: str) -> None:
        findings.append((severity, msg))
        report_warnings.append(msg.replace("**", ""))

    if results["crushing_warning_material"]:
        add("warn", f"**Material Crushing Risk:** Preload clamped to keep bearing stress below the "
            f"compressive yield of {results['crushing_warning_material']}. Consider washers or a "
            f"larger head.")

    if ctx.preload_hi > results["proof_load_N"]:
        add("warn", f"**Tightening Overload Risk:** With ±{ctx.scatter*100:.0f}% scatter the preload may reach "
            f"{ctx.preload_hi*units.force_factor:,.0f} {units.force_unit}, exceeding the proof load "
            f"({results['proof_load_N']*units.force_factor:,.0f} {units.force_unit}). Use a more accurate "
            f"tightening method or lower the target.")

    if ctx.has_internal and results["thread_shear_fos"] < 1.0:
        add("error", f"**Thread Stripping Risk:** thread-shear FoS is below 1 "
            f"({results['thread_shear_fos']:.2f}) — the internal threads strip before the bolt "
            f"yields. Increase engagement length (≥ "
            f"{results['required_engagement_mm']*units.length_factor:,.2f} {units.len_unit}) or use a "
            f"stronger internal material.")
    elif ctx.has_internal and results["thread_shear_fos"] < ctx.required_fos:
        add("warn", f"**Thread engagement below required FoS:** thread-shear FoS is "
            f"{results['thread_shear_fos']:.2f}, under the required {ctx.required_fos:.2f}. Increase "
            f"engagement (≥ {results['required_engagement_mm']*units.length_factor:,.2f} "
            f"{units.len_unit}) or use a stronger internal material.")

    if ctx.use_fatigue and results["fatigue_fos"] < 1.0:
        add("error", f"**Fatigue Failure Risk:** Bolt is expected to fail under cyclic loading "
            f"({ctx.fatigue_criterion} FOS = {results['fatigue_fos']:.2f}).")

    if ctx.ext_max_N > 0 and results["separation_fos"] < 1.0:
        add("error", f"**Joint Separation Risk:** The external load exceeds the separation load "
            f"(FOS = {results['separation_fos']:.2f}). The joint will gap open.")

    if results["proof_fos"] < 1.0:
        add("error", f"**Bolt Yield Risk:** Maximum service load exceeds the proof load "
            f"(Proof FOS = {results['proof_fos']:.2f}).")

    if ctx.slip_fos is not None and ctx.slip_fos < 1.0:
        add("error", f"**Joint Slip Risk:** Friction from the clamp load cannot carry the bolt shear "
            f"(Slip FOS = {ctx.slip_fos:.2f}). The joint relies on bolts in bearing.")

    if ctx.combined_fos is not None and ctx.combined_fos < 1.0:
        add("error", f"**Combined Tension-Shear Risk:** the governing bolt's elliptic tension-shear "
            f"interaction FoS is {ctx.combined_fos:.2f} (< 1). Reduce the load, add bolts, or use a "
            f"larger size/grade.")

    if results["tightening_utilization"] > 1.0:
        add("error", f"**Tightening Yield Risk:** the von Mises stress while torquing "
            f"({results['tightening_stress_MPa']*units.stress_factor:,.0f} {units.stress_unit}) exceeds "
            f"yield. Lower the target preload, lubricate, or use angle/tension control.")
    elif results["tightening_utilization"] > 0.9:
        add("warn", f"**High Tightening Utilization:** the combined (axial + torsion) stress while "
            f"torquing is {results['tightening_utilization']*100:.0f}% of yield (guideline ≤ 90%).")

    if ctx.embedment_um > 0 and results["embedment_loss_N"] >= results["recommended_preload_N"]:
        add("error", "**Embedment Loss Risk:** the relaxation loss meets or exceeds the installation "
            "preload — the joint may lose clamp entirely. Re-check the embedment estimate.")

    return findings, report_warnings

