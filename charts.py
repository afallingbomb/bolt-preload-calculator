from __future__ import annotations
"""Figure and chart builders for the bolt preload calculator.

Pure plotting helpers (matplotlib for the PDF, Altair for the interactive UI) with
no Streamlit dependency, so they can be reused by both the on-screen app and the
PDF/report exports and unit-tested in isolation. All inputs are already in display
units (the caller owns unit conversion); the ``dark`` flag switches between the
on-screen dark theme and the light theme used in the PDF.
"""
import io
import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import altair as alt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg

from mechanics import bolt_member_forces, morrow_sigma_f, PreloadResult


# =============================================================================
# Figure helpers (re-used for both the dark on-screen theme and the light PDF)
# =============================================================================
def _style_axes(fig: Figure, ax: Any, dark: bool) -> None:
    if dark:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        c = '#e2e8f0'
        ax.xaxis.label.set_color(c)
        ax.yaxis.label.set_color(c)
        ax.title.set_color(c)
        ax.tick_params(colors=c)
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_color('black')


def _new_figure(figsize: Tuple[float, float]) -> Tuple[Figure, Any]:
    """Create a stand-alone Agg figure (no pyplot global state) and one axes.

    Using the object-oriented API keeps every figure local to the call instead of
    the shared pyplot figure manager, which is not thread-safe across Streamlit's
    per-session script threads and otherwise needs manual plt.close() to avoid
    leaks. The figure is garbage-collected normally once it goes out of scope.
    """
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    return fig, ax


def fig_to_png(fig: Figure) -> bytes:
    """Render a figure to PNG bytes (white background)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    return buf.getvalue()


def make_joint_diagram(kb: float, km: float, Fi: float, F_ext: float, C: float,
                       ff: float, lf: float, force_unit: str, len_unit: str,
                       dark: bool = True) -> Optional[Figure]:
    """Force-vs-deflection joint diagram. Returns None if stiffness is degenerate."""
    if not (kb > 0 and km > 0 and Fi > 0 and kb != float('inf') and km != float('inf')):
        return None
    fig, ax = _new_figure((10, 5))
    delta_b = Fi / kb
    delta_m = Fi / km

    ax.plot([0, delta_b * lf], [0, Fi * ff], 'b-', linewidth=2, label='Bolt Stiffness')
    ax.plot([(delta_b + delta_m) * lf, delta_b * lf], [0, Fi * ff], 'r-', linewidth=2, label='Joint Stiffness')
    ax.axhline(y=Fi * ff, color='gray', linestyle='--', label='Operating Preload (Fi)')

    if F_ext > 0:
        delta_ext = F_ext / (kb + km)
        Fb_max = Fi + C * F_ext
        Fm_min = Fi - (1 - C) * F_ext
        ax.plot([0, (delta_b + delta_ext) * lf], [0, Fb_max * ff], 'b:', linewidth=1.5)
        ax.plot([(delta_b + delta_m) * lf, (delta_b + delta_ext) * lf], [0, Fm_min * ff], 'r:', linewidth=1.5)
        ax.plot([(delta_b + delta_ext) * lf, (delta_b + delta_ext) * lf], [Fm_min * ff, Fb_max * ff],
                'g-', linewidth=3, label="External Load (F_ext)")
        ax.axhline(y=Fb_max * ff, color='purple', linestyle=':', alpha=0.5, label="Max Bolt Force")

    ax.set_xlabel(f"Deflection ({len_unit})")
    ax.set_ylabel(f"Force ({force_unit})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _style_axes(fig, ax, dark)
    return fig


def make_bolt_group_figure(coords: List[Tuple[float, float]], tensions: List[float], gov_idx: int,
                           lf: float, ff: float, len_unit: str, force_unit: str,
                           shear_vectors: Optional[List[Tuple[float, float]]] = None,
                           dark: bool = True) -> Figure:
    """Scatter of the bolt pattern coloured by per-bolt tension.

    If ``shear_vectors`` (per-bolt (Fx, Fy) in N) are supplied, the resultant shear
    of each bolt is overlaid as an arrow (direct + torsional). The arrows are scaled
    only for legibility, not to data units. The axis window is a square sized to
    enclose every marker and arrow tip (plus padding) so nothing is clipped, mainly
    for the PDF export -- the on-screen plot uses the interactive Altair version.
    """
    fig, ax = _new_figure((6.2, 5.6))
    fig.set_layout_engine("constrained")           # let the colorbar/labels fit
    xs = [x * lf for x, _ in coords]
    ys = [y * lf for _, y in coords]
    t_disp = [t * ff for t in tensions]

    sc = ax.scatter(xs, ys, c=t_disp, cmap='coolwarm', s=240, edgecolors='black', zorder=3)
    if 0 <= gov_idx < len(coords):
        ax.scatter([xs[gov_idx]], [ys[gov_idx]], s=520, facecolors='none',
                   edgecolors='#16a34a', linewidths=2.5, zorder=4, label='Governing bolt')
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    ax.plot([cx], [cy], 'k+', markersize=12, markeredgewidth=2, zorder=5)

    tip_x: List[float] = []
    tip_y: List[float] = []
    if shear_vectors and any(math.hypot(vx, vy) > 0.0 for vx, vy in shear_vectors):
        max_mag = max(math.hypot(vx, vy) for vx, vy in shear_vectors)
        span = max((max(xs) - min(xs)), (max(ys) - min(ys)), 1.0)
        target = 0.22 * span                       # display length of the largest arrow
        scale = max_mag / target if target > 0 else 1.0
        us = [vx for vx, _ in shear_vectors]
        vs = [vy for _, vy in shear_vectors]
        ax.quiver(xs, ys, us, vs, angles='xy', scale_units='xy', scale=scale,
                  color='#15803d', width=0.006, zorder=6)
        tip_x = [x + u / scale for x, u in zip(xs, us)]
        tip_y = [y + v / scale for y, v in zip(ys, vs)]
        ax.text(0.02, 0.02, f"arrows: shear (max {max_mag*ff:,.0f} {force_unit})",
                transform=ax.transAxes, fontsize=8, color='#15803d', va='bottom', ha='left')

    # Square window enclosing every marker and arrow tip, with padding, so the
    # equal-aspect plot never clips an element.
    all_x = xs + tip_x
    all_y = ys + tip_y
    mid_x = (min(all_x) + max(all_x)) / 2.0
    mid_y = (min(all_y) + max(all_y)) / 2.0
    half = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) / 2.0
    half = half * 1.18 + 0.08 * max(half, 1.0) + 1e-6
    ax.set_xlim(mid_x - half, mid_x + half)
    ax.set_ylim(mid_y - half, mid_y + half)
    ax.set_aspect('equal')

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label(f"Bolt tension ({force_unit})")
    ax.set_xlabel(f"x ({len_unit})")
    ax.set_ylabel(f"y ({len_unit})")
    ax.grid(True, alpha=0.3)
    if 0 <= gov_idx < len(coords):
        ax.legend(loc='upper left', fontsize=8, framealpha=0.6)
    _style_axes(fig, ax, dark)
    if dark:
        cbar.ax.yaxis.label.set_color('#e2e8f0')
        cbar.ax.tick_params(colors='#e2e8f0')
    return fig


def make_haigh_diagram(res: PreloadResult, sf: float, stress_unit: str, dark: bool = True) -> Optional[Figure]:
    """Haigh (constant-life) diagram: every mean-stress failure locus, the VDI
    amplitude limits, the load line from the preload point and the operating point.
    sf converts MPa to the display stress unit."""
    sa = res['fatigue_sigma_a_MPa']
    sm = res['fatigue_sigma_m_MPa']
    si = res['preload_stress_MPa']
    Se, Sut = res['endurance_Se_MPa'], res['ultimate_Sut_MPa']
    Sp, Sy = res['proof_strength_MPa'], res['yield_Sy_MPa']
    if Se <= 0 or Sut <= 0:
        return None

    fig, ax = _new_figure((8, 5))
    n = 60

    def ramp(stop: float) -> List[float]:
        return [stop * i / n for i in range(n + 1)]

    ax.plot([0, Sut * sf], [Se * sf, 0], '-', color='#2563eb', lw=1.6, label='Goodman')
    ax.plot([0, Sy * sf], [Se * sf, 0], '--', color='#16a34a', lw=1.4, label='Soderberg')
    sigf = morrow_sigma_f(Sut)
    ax.plot([0, sigf * sf], [Se * sf, 0], ':', color='#9333ea', lw=1.6, label='Morrow')
    gx = ramp(Sut)
    ax.plot([x * sf for x in gx], [Se * (1 - (x / Sut) ** 2) * sf for x in gx],
            '-', color='#ea580c', lw=1.4, label='Gerber')
    ex = ramp(Sp)
    ax.plot([x * sf for x in ex], [Se * math.sqrt(max(0.0, 1 - (x / Sp) ** 2)) * sf for x in ex],
            '-.', color='#0891b2', lw=1.4, label='ASME-elliptic')
    wx = ramp(Sut)
    ax.plot([x * sf for x in wx], [(-x + math.sqrt(x * x + 4 * Se * Se)) / 2.0 * sf for x in wx],
            '-', color='#db2777', lw=1.2, label='SWT')
    asv, asg = res.get('vdi_sigma_asv_MPa', 0.0), res.get('vdi_sigma_asg_MPa', 0.0)
    if asv > 0:
        ax.axhline(asv * sf, color='#64748b', ls=(0, (5, 3)), lw=1.0, label='VDI 2230 (before HT)')
    if asg > asv:
        ax.axhline(asg * sf, color='#94a3b8', ls=(0, (1, 2)), lw=1.0, label='VDI 2230 (after HT)')
    ax.plot([0, Sy * sf], [Sy * sf, 0], '-', color='#dc2626', lw=0.8, alpha=0.6, label='Yield (Langer)')

    if sa > 0:
        ax.plot([si * sf, sm * sf], [0, sa * sf], color='gray', lw=1.0, label='Load line')
    ax.plot([sm * sf], [sa * sf], 'o', color='#111111', ms=8, zorder=6, label='Operating point')

    ax.set_xlabel(r"Mean stress $\sigma_m$ (%s)" % stress_unit)
    ax.set_ylabel(r"Alternating stress $\sigma_a$ (%s)" % stress_unit)
    ax.set_xlim(0, max(Sut, sm * 1.1) * sf)
    ax.set_ylim(0, max(Se, sa) * 1.25 * sf)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    _style_axes(fig, ax, dark)
    return fig


_MAT_PALETTE = ['#60a5fa', '#f59e0b', '#34d399', '#f472b6', '#a78bfa',
                '#fbbf24', '#22d3ee', '#fb7185', '#4ade80', '#c084fc']


def make_cross_section_diagram(layers: Sequence[Mapping[str, Any]], d: float, dw: float, use_washer: bool,
                               lf: float, len_unit: str, dark: bool = True) -> Optional[Figure]:
    """Scaled cross-section of the bolted stack with the Rötscher compression cones.

    Geometry is drawn to scale in mm; the grip and bearing diameter are annotated in
    display units (factor ``lf``). The translucent double cone is the member
    compression zone of §8. Returns None when there is no grip to draw.
    """
    thick = [float(r.get("thickness", 0.0) or 0.0) for r in layers]
    L = sum(thick)
    if L <= 0 or not layers:
        return None
    fig, ax = _new_figure((7, 6))

    rc = 0.55 * d                       # clearance-hole radius (~1.1 d)
    W = max(dw * 0.75, d * 2.2)         # half-width of the drawn members
    tan30 = math.tan(math.radians(30.0))

    mats: List[str] = []
    for r in layers:
        m = str(r.get("Material", "—"))
        if m not in mats:
            mats.append(m)
    cmap = {m: _MAT_PALETTE[i % len(_MAT_PALETTE)] for i, m in enumerate(mats)}

    # Clamped layers: two rectangles each, leaving the clearance hole open.
    y0 = 0.0
    for r, t in zip(layers, thick):
        if t <= 0:
            continue
        col = cmap[str(r.get("Material", "—"))]
        for x_left in (-W, rc):
            ax.add_patch(Rectangle((x_left, y0), W - rc, t, facecolor=col,
                                   edgecolor='#334155', linewidth=0.8, zorder=2))
        y0 += t

    # Compression frustum (right half + mirror), shaded translucent.
    r_mid = dw / 2.0 + (L / 2.0) * tan30
    right = [(rc, 0.0), (dw / 2.0, 0.0), (r_mid, L / 2.0), (dw / 2.0, L), (rc, L)]
    for poly in (right, [(-x, y) for x, y in right]):
        ax.add_patch(Polygon(poly, closed=True, facecolor='#3b82f6', alpha=0.12,
                             edgecolor='#3b82f6', linewidth=1.0, linestyle='--', zorder=3))

    # Hardware (shank, head, washers, nut).
    wt = 0.15 * d if use_washer else 0.0
    hh, nh = 0.7 * d, 0.8 * d
    head_w, nut_w = max(dw, 1.6 * d), 1.8 * d
    shank_bottom, shank_top = -(nh + wt), L + wt + hh
    ax.add_patch(Rectangle((-d / 2.0, shank_bottom), d, shank_top - shank_bottom,
                           facecolor='#9ca3af', edgecolor='#374151', linewidth=0.8, zorder=4))
    ax.add_patch(Rectangle((-head_w / 2.0, L + wt), head_w, hh, facecolor='#6b7280',
                           edgecolor='#374151', linewidth=0.8, zorder=5))
    ax.add_patch(Rectangle((-nut_w / 2.0, -(nh + wt)), nut_w, nh, facecolor='#6b7280',
                           edgecolor='#374151', linewidth=0.8, zorder=5))
    if use_washer:
        for yw in (L, -wt):
            ax.add_patch(Rectangle((-dw / 2.0, yw), dw, wt, facecolor='#d1d5db',
                                   edgecolor='#374151', linewidth=0.6, zorder=5))

    txt_c = '#e2e8f0' if dark else '#111111'
    xdim = W + 0.6 * d
    ax.annotate('', xy=(xdim, 0), xytext=(xdim, L),
                arrowprops=dict(arrowstyle='<->', color='#ef4444', lw=1.2))
    ax.text(xdim + 0.15 * d, L / 2.0, f"Grip L = {L*lf:,.2f} {len_unit}",
            rotation=90, va='center', ha='left', color='#ef4444', fontsize=9)
    ytop = L + wt + hh + 0.5 * d
    ax.annotate('', xy=(-dw / 2.0, ytop), xytext=(dw / 2.0, ytop),
                arrowprops=dict(arrowstyle='<->', color='#22c55e', lw=1.2))
    ax.text(0, ytop + 0.2 * d, f"Bearing ø {dw*lf:,.2f} {len_unit}",
            va='bottom', ha='center', color='#22c55e', fontsize=9)

    handles = [Rectangle((0, 0), 1, 1, facecolor=cmap[m], edgecolor='#334155') for m in mats]
    leg = ax.legend(handles, mats, loc='lower left', fontsize=8, title='Layers')
    if leg.get_title() is not None:
        leg.get_title().set_color(txt_c)
    for t_obj in leg.get_texts():
        t_obj.set_color(txt_c)

    ax.set_xlim(-W - 2.0 * d, W + 3.4 * d)
    ax.set_ylim(shank_bottom - 0.5 * d, ytop + 1.2 * d)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    if dark:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    else:
        fig.patch.set_facecolor('white')
    return fig


def alt_joint_diagram(kb: float, km: float, Fi: float, F_ext: float, C: float,
                      ff: float, lf: float, force_unit: str, len_unit: str) -> Optional[Any]:
    """Interactive (Altair) force-vs-deflection joint diagram. None if degenerate."""
    if not (kb > 0 and km > 0 and Fi > 0 and kb != float('inf') and km != float('inf')):
        return None
    delta_b, delta_m = Fi / kb, Fi / km
    df = pd.DataFrame([
        {"Deflection": 0.0, "Force": 0.0, "Member": "Bolt"},
        {"Deflection": delta_b * lf, "Force": Fi * ff, "Member": "Bolt"},
        {"Deflection": (delta_b + delta_m) * lf, "Force": 0.0, "Member": "Joint (members)"},
        {"Deflection": delta_b * lf, "Force": Fi * ff, "Member": "Joint (members)"},
    ])
    base = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Deflection:Q", title=f"Deflection ({len_unit})"),
        y=alt.Y("Force:Q", title=f"Force ({force_unit})"),
        color=alt.Color("Member:N", title=None, scale=alt.Scale(
            domain=["Bolt", "Joint (members)"], range=["#2563eb", "#dc2626"])),
        tooltip=[alt.Tooltip("Member:N"), alt.Tooltip("Deflection:Q", format=",.4f"),
                 alt.Tooltip("Force:Q", format=",.0f")])
    layers = [base, alt.Chart(pd.DataFrame({"Force": [Fi * ff]})).mark_rule(
        color="gray", strokeDash=[5, 4]).encode(y="Force:Q")]
    if F_ext > 0:
        fb_max = Fi + C * F_ext
        layers.append(alt.Chart(pd.DataFrame({"Force": [fb_max * ff]})).mark_rule(
            color="#7c3aed", strokeDash=[2, 3]).encode(
            y="Force:Q", tooltip=[alt.Tooltip("Force:Q", format=",.0f", title="Max bolt force")]))
    return alt.layer(*layers).properties(height=340, width="container")


def alt_bolt_force_chart(operating_preload: float, C: float, separation_load: float,
                         P_max: float, ff: float, force_unit: str) -> Optional[Any]:
    """Interactive bolt-force / member-force vs external-load chart with the
    separation point and current load marked."""
    if operating_preload <= 0 or C >= 1.0:
        return None
    finite_sep = separation_load if separation_load != float('inf') else 0.0
    p_top = max(P_max * 1.3, finite_sep * 1.3, operating_preload * 1.5, 1.0)
    n = 41
    rows = []
    for i in range(n):
        P = p_top * i / (n - 1)
        fb, fm = bolt_member_forces(operating_preload, C, P)
        rows.append({"P": P * ff, "Force": fb * ff, "Series": "Bolt force F_b"})
        rows.append({"P": P * ff, "Force": fm * ff, "Series": "Member clamp F_m"})
    chart = alt.Chart(pd.DataFrame(rows)).mark_line().encode(
        x=alt.X("P:Q", title=f"External load per bolt ({force_unit})"),
        y=alt.Y("Force:Q", title=f"Force ({force_unit})"),
        color=alt.Color("Series:N", title=None, scale=alt.Scale(
            domain=["Bolt force F_b", "Member clamp F_m"], range=["#2563eb", "#dc2626"])),
        tooltip=[alt.Tooltip("Series:N"), alt.Tooltip("P:Q", format=",.0f"),
                 alt.Tooltip("Force:Q", format=",.0f")])
    layers = [chart]
    if 0 < finite_sep <= p_top:
        layers.append(alt.Chart(pd.DataFrame({"P": [finite_sep * ff]})).mark_rule(
            color="#16a34a", strokeDash=[5, 4]).encode(
            x="P:Q", tooltip=[alt.Tooltip("P:Q", format=",.0f", title="Separation load")]))
    if P_max > 0:
        layers.append(alt.Chart(pd.DataFrame({"P": [P_max * ff]})).mark_rule(
            color="#6b7280", strokeDash=[2, 2]).encode(x="P:Q"))
    return alt.layer(*layers).properties(height=320, width="container")


def alt_clamp_waterfall(steps: List[dict], ff: float, force_unit: str) -> Any:
    """Interactive clamp-load budget waterfall from a clamp_load_budget() step list."""
    order = [s["label"] for s in steps]
    rows = []
    prev = 0.0
    for s in steps:
        cum = s["cumulative"] * ff
        if s["kind"] in ("start", "total"):
            base, top, kind = 0.0, cum, "Total"
        else:
            base, top = min(prev, cum), max(prev, cum)
            kind = "Increase" if s["delta"] >= 0 else "Decrease"
        rows.append({"Step": s["label"], "base": base, "top": top,
                     "delta": s["delta"] * ff, "cumulative": cum, "Type": kind})
        prev = cum
    return alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
        x=alt.X("Step:N", sort=order, title=None),
        y=alt.Y("base:Q", title=f"Clamp force ({force_unit})"), y2="top:Q",
        color=alt.Color("Type:N", title=None, scale=alt.Scale(
            domain=["Total", "Increase", "Decrease"], range=["#2563eb", "#16a34a", "#dc2626"])),
        tooltip=[alt.Tooltip("Step:N"), alt.Tooltip("delta:Q", format=",.0f"),
                 alt.Tooltip("cumulative:Q", format=",.0f")]).properties(height=320, width="container")


def alt_load_sharing(C: float, P: float, ff: float, force_unit: str) -> Any:
    """Interactive normalized bar of the external-load split (bolt C vs members 1-C)."""
    df = pd.DataFrame([
        {"Path": "Bolt (C·P)", "Fraction": C, "Force": C * P * ff, "o": 0},
        {"Path": "Members relieved ((1−C)·P)", "Fraction": 1.0 - C,
         "Force": (1.0 - C) * P * ff, "o": 1},
    ])
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Fraction:Q", stack="normalize", title="Share of external load",
                axis=alt.Axis(format="%")),
        color=alt.Color("Path:N", title=None, scale=alt.Scale(range=["#2563eb", "#94a3b8"])),
        order=alt.Order("o:Q"),
        tooltip=[alt.Tooltip("Path:N"), alt.Tooltip("Fraction:Q", format=".1%"),
                 alt.Tooltip("Force:Q", format=",.0f", title=f"Force ({force_unit})")]
    ).properties(height=90, width="container")


def alt_bolt_group_chart(coords: List[Tuple[float, float]], tensions: List[float],
                         shear_vectors: Optional[List[Tuple[float, float]]], gov_idx: int,
                         lf: float, ff: float, len_unit: str, force_unit: str) -> Optional[Any]:
    """Interactive (Altair) bolt-pattern plot: markers coloured by tension, the
    governing bolt ringed, the centroid marked, and per-bolt shear resultants as
    arrows. A square, padded domain encloses every element (no clipping); hover
    gives per-bolt values and the chart pans/zooms."""
    n = len(coords)
    if n == 0:
        return None
    xs = [x * lf for x, _ in coords]
    ys = [y * lf for _, y in coords]
    cx, cy = sum(xs) / n, sum(ys) / n

    pts = pd.DataFrame([
        {"idx": i + 1, "x": xs[i], "y": ys[i], "tension": tensions[i] * ff,
         "shear": (math.hypot(*shear_vectors[i]) * ff
                   if shear_vectors and i < len(shear_vectors) else 0.0),
         "role": "Governing" if i == gov_idx else "Bolt"}
        for i in range(n)])

    seg_rows: List[dict] = []
    if shear_vectors and any(math.hypot(vx, vy) > 0.0 for vx, vy in shear_vectors):
        max_mag = max(math.hypot(vx, vy) for vx, vy in shear_vectors)
        max_t = max(abs(t) for t in tensions) if tensions else 0.0
        ref_force = max(max_mag, max_t, 1e-6)
        span = max((max(xs) - min(xs)), (max(ys) - min(ys)), 1.0)
        # Cap visual arrow length relative to max force in the system (tension or shear)
        factor = (0.22 * span) / ref_force if ref_force > 0 else 0.0
        for i, (vx, vy) in enumerate(shear_vectors):
            ux, uy = vx * factor, vy * factor
            seg_rows.append({"x": xs[i], "y": ys[i], "x2": xs[i] + ux, "y2": ys[i] + uy,
                             "ang": math.degrees(math.atan2(ux, uy)) if (ux or uy) else 0.0,
                             "shear": math.hypot(vx, vy) * ff})

    all_x = xs + [r["x2"] for r in seg_rows] + [cx]
    all_y = ys + [r["y2"] for r in seg_rows] + [cy]
    mid_x, mid_y = (min(all_x) + max(all_x)) / 2.0, (min(all_y) + max(all_y)) / 2.0
    half = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) / 2.0 * 1.18 + 1e-6
    dom_x, dom_y = [mid_x - half, mid_x + half], [mid_y - half, mid_y + half]
    x_enc = alt.X("x:Q", title=f"x ({len_unit})", scale=alt.Scale(domain=dom_x, nice=False))
    y_enc = alt.Y("y:Q", title=f"y ({len_unit})", scale=alt.Scale(domain=dom_y, nice=False))

    layers: List[Any] = []
    if seg_rows:
        segs = pd.DataFrame(seg_rows)
        layers.append(alt.Chart(segs).mark_rule(color="#15803d", strokeWidth=2).encode(
            x=x_enc, y=y_enc, x2="x2:Q", y2="y2:Q",
            tooltip=[alt.Tooltip("shear:Q", title=f"Shear ({force_unit})", format=",.0f")]))
        layers.append(alt.Chart(segs).mark_point(
            shape="triangle-up", filled=True, color="#15803d", size=70).encode(
            x=alt.X("x2:Q", scale=alt.Scale(domain=dom_x, nice=False)),
            y=alt.Y("y2:Q", scale=alt.Scale(domain=dom_y, nice=False)),
            angle=alt.Angle("ang:Q", scale=None)))

    layers.append(alt.Chart(pts).mark_circle(size=260, stroke="black", strokeWidth=0.8).encode(
        x=x_enc, y=y_enc,
        color=alt.Color("tension:Q", title=f"Tension ({force_unit})",
                        scale=alt.Scale(scheme="redblue", reverse=True)),
        tooltip=[alt.Tooltip("idx:Q", title="Bolt #"),
                 alt.Tooltip("x:Q", title=f"x ({len_unit})", format=",.1f"),
                 alt.Tooltip("y:Q", title=f"y ({len_unit})", format=",.1f"),
                 alt.Tooltip("tension:Q", title=f"Tension ({force_unit})", format=",.0f"),
                 alt.Tooltip("shear:Q", title=f"Shear ({force_unit})", format=",.0f")]))

    gov = pts[pts["role"] == "Governing"]
    if not gov.empty:
        layers.append(alt.Chart(gov).mark_point(
            shape="circle", size=560, stroke="#16a34a", strokeWidth=2.5, filled=False).encode(
            x=x_enc, y=y_enc))
    layers.append(alt.Chart(pd.DataFrame([{"x": cx, "y": cy}])).mark_point(
        shape="cross", size=180, color="#475569", strokeWidth=2).encode(x=x_enc, y=y_enc))

    return alt.layer(*layers).properties(width=560, height=560).interactive()


# =============================================================================
# FE-results population graphics (on-screen Altair + a matplotlib PDF dashboard)
# =============================================================================
# Okabe-Ito colour-blind-safe pair (bluish-green / vermillion) for the pass/fail
# charts, where the verdict is encoded by colour alone.
_PASS_FAIL = alt.Scale(domain=["PASS", "FAIL"], range=["#009E73", "#D55E00"])


def _fe_results_df(ok_results: List[dict]) -> Any:
    """Per-bolt FE results as a DataFrame with the derived columns the charts need."""
    df = pd.DataFrame(ok_results)
    df["Result"] = ["PASS" if bool(pz) else "FAIL" for pz in df["passes"]]
    df["sm_n"] = df["sigma_m_MPa"] / df["Sut_MPa"]          # normalised mean stress
    df["sa_n"] = df["sigma_a_MPa"] / df["Se_MPa"]           # normalised alternating
    df["tau"] = df["shear_max_N"] / df["stress_area_mm2"]
    df["tn"] = df["sigma_max_MPa"] / df["Sp_MPa"]           # tension utilisation
    df["vn"] = df["tau"] / (0.577 * df["Sy_MPa"])           # shear utilisation
    return df


def _quarter_circle() -> Any:
    pts = [i / 40.0 for i in range(41)]
    return pd.DataFrame({"x": pts, "y": [math.sqrt(max(0.0, 1.0 - x * x)) for x in pts]})


def fe_chart_histogram(df: Any, target: float) -> Any:
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("min_fos:Q", bin=alt.Bin(maxbins=30), title="Min FOS"),
        y=alt.Y("count()", title="Bolts"),
        color=alt.Color("Result:N", scale=_PASS_FAIL, title=None))
    rule = alt.Chart(pd.DataFrame({"t": [target]})).mark_rule(
        color="#111", strokeDash=[4, 4]).encode(x="t:Q")
    return (bars + rule).properties(height=260, width="container")


def fe_chart_governing(df: Any) -> Any:
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("governing:N", title="Governing check", sort="-y"),
        y=alt.Y("count()", title="Bolts"),
        color=alt.Color("governing:N", legend=None),
        tooltip=["governing:N", alt.Tooltip("count()", title="Bolts")]
    ).properties(height=260, width="container")


def fe_chart_worst(df: Any, target: float, n: int = 20) -> Any:
    worst = df.nsmallest(n, "min_fos")
    bars = alt.Chart(worst).mark_bar().encode(
        y=alt.Y("bolt_id:N", sort="x", title=None),
        x=alt.X("min_fos:Q", title="Min FOS"),
        color=alt.Color("Result:N", scale=_PASS_FAIL, title=None),
        tooltip=["bolt_id:N", alt.Tooltip("min_fos:Q", format=".2f"), "governing:N"])
    rule = alt.Chart(pd.DataFrame({"t": [target]})).mark_rule(
        color="#111", strokeDash=[4, 4]).encode(x="t:Q")
    return (bars + rule).properties(height=max(260, 18 * len(worst)), width="container")


def fe_chart_haigh(df: Any) -> Any:
    line = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [1, 0]})).mark_line(
        color="#2563eb").encode(x="x:Q", y="y:Q")
    pts = alt.Chart(df).mark_circle(size=70, opacity=0.75).encode(
        x=alt.X("sm_n:Q", title="σm / Sut"), y=alt.Y("sa_n:Q", title="σa / Se"),
        color=alt.Color("Result:N", scale=_PASS_FAIL, title=None),
        tooltip=["bolt_id:N", alt.Tooltip("sm_n:Q", format=".2f"),
                 alt.Tooltip("sa_n:Q", format=".2f")])
    return (line + pts).properties(height=300, width="container")


def fe_chart_tension_shear(df: Any) -> Any:
    line = alt.Chart(_quarter_circle()).mark_line(color="#2563eb").encode(
        x=alt.X("x:Q", title="σmax / Sp"), y=alt.Y("y:Q", title="τ / (0.577 Sy)"))
    pts = alt.Chart(df).mark_circle(size=70, opacity=0.75).encode(
        x="tn:Q", y="vn:Q", color=alt.Color("Result:N", scale=_PASS_FAIL, title=None),
        tooltip=["bolt_id:N", alt.Tooltip("tn:Q", format=".2f"),
                 alt.Tooltip("vn:Q", format=".2f")])
    return (line + pts).properties(height=300, width="container")


def fe_chart_box(df: Any) -> Any:
    long_rows = []
    for _, r in df.iterrows():
        for label, col in (("Proof", "proof_fos"), ("Fatigue", "fatigue_fos"),
                           ("Shear", "shear_fos"), ("Combined", "combined_fos")):
            v = r[col]
            long_rows.append({"check": label, "fos": 5.0 if v == float('inf') else min(v, 5.0)})
    return alt.Chart(pd.DataFrame(long_rows)).mark_boxplot(extent="min-max").encode(
        x=alt.X("check:N", title=None, sort=["Proof", "Fatigue", "Shear", "Combined"]),
        y=alt.Y("fos:Q", title="FOS (capped at 5)")).properties(height=260, width="container")


def fe_chart_ecdf(df: Any, target: float) -> Any:
    base = alt.Chart(df).transform_window(
        ecdf="cume_dist()", sort=[{"field": "min_fos"}]).mark_line(
        interpolate="step-after", color="#2563eb").encode(
        x=alt.X("min_fos:Q", title="Min FOS"),
        y=alt.Y("ecdf:Q", title="Fraction of bolts ≤", axis=alt.Axis(format="%")))
    rule = alt.Chart(pd.DataFrame({"t": [target]})).mark_rule(
        color="#111", strokeDash=[4, 4]).encode(x="t:Q")
    return (base + rule).properties(height=260, width="container")


def make_fe_dashboard_figure(df: Any, target: float) -> Figure:
    """Light-themed matplotlib multi-panel of the FE population graphics for the PDF."""
    fig = Figure(figsize=(10, 11.5))
    FigureCanvasAgg(fig)
    fig.set_layout_engine("constrained")
    axes = fig.subplots(4, 2)
    pf_colors = ["#009E73" if bool(pz) else "#D55E00" for pz in df["passes"]]

    ax = axes[0][0]
    ax.hist(df["min_fos"], bins=20, color="#3b82f6", edgecolor="white")
    ax.axvline(target, color="red", ls="--")
    ax.set_title("Min-FOS distribution")
    ax.set_xlabel("Min FOS")
    ax.set_ylabel("Bolts")

    ax = axes[0][1]
    counts = df["governing"].value_counts()
    ax.bar([str(i) for i in counts.index], counts.values, color="#0891b2")
    ax.set_title("Governing check")
    ax.set_ylabel("Bolts")

    ax = axes[1][0]
    worst = df.nsmallest(15, "min_fos")
    wcolors = ["#009E73" if bool(pz) else "#D55E00" for pz in worst["passes"]]
    ax.barh([str(b) for b in worst["bolt_id"]], worst["min_fos"], color=wcolors)
    ax.axvline(target, color="#111", ls="--")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title("Worst 15 bolts")
    ax.set_xlabel("Min FOS")

    ax = axes[1][1]
    xs = sorted(df["min_fos"])
    ys = [(i + 1) / len(xs) for i in range(len(xs))]
    ax.step(xs, ys, where="post", color="#2563eb")
    ax.axvline(target, color="#111", ls="--")
    ax.set_title("Cumulative distribution")
    ax.set_xlabel("Min FOS")
    ax.set_ylabel("Fraction ≤")

    ax = axes[2][0]
    ax.plot([0, 1], [1, 0], color="#2563eb")
    ax.scatter(df["sm_n"], df["sa_n"], c=pf_colors, s=18, edgecolors="black", linewidths=0.3)
    ax.set_title("Fatigue (normalised Haigh, Goodman ref.)")
    ax.set_xlabel("σm / Sut")
    ax.set_ylabel("σa / Se")

    ax = axes[2][1]
    qc = _quarter_circle()
    ax.plot(qc["x"], qc["y"], color="#2563eb")
    ax.scatter(df["tn"], df["vn"], c=pf_colors, s=18, edgecolors="black", linewidths=0.3)
    ax.set_title("Tension-shear interaction")
    ax.set_xlabel("σmax / Sp")
    ax.set_ylabel("τ / (0.577 Sy)")

    ax = axes[3][0]
    box_data = [[5.0 if v == float('inf') else min(v, 5.0) for v in df[col]]
                for col in ("proof_fos", "fatigue_fos", "shear_fos", "combined_fos")]
    ax.boxplot(box_data, tick_labels=["Proof", "Fatigue", "Shear", "Comb."])
    ax.set_title("FOS by check (capped at 5)")
    ax.set_ylabel("FOS")

    axes[3][1].axis("off")
    return fig

